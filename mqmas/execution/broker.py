"""Order execution: paper simulator and live CCXT spot broker.

Both brokers implement the same ``Broker`` protocol so the rest of the system
is mode-agnostic. The live broker:

* sends **LIMIT** entry orders only (marketable at the best ask, capped by
  the slippage tolerance), with a fill timeout and cancel-on-timeout;
* rounds quantities *down* to the exchange step (risk can only shrink);
* places an **exchange-native stop-limit** right after the fill, so the
  downside is protected even if this process crashes;
* manages the take-profit in software (spot accounts cannot lock the same
  BTC in two resting sell orders without OCO).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from mqmas.config import Settings
from mqmas.data.market_feed import MarketFeed, with_backoff

log = logging.getLogger(__name__)

STOP_LIMIT_OFFSET = 0.002  # stop-limit sells 0.2% below trigger to guarantee a fill


@dataclass(frozen=True)
class Fill:
    order_id: str | None
    quantity: float  # base-asset quantity we actually hold / sold
    avg_price: float
    fee_usdt: float


class Broker(Protocol):
    async def min_notional(self) -> float: ...
    async def available_cash(self, mandate_equity: float) -> float: ...
    def normalize_quantity(self, qty: float) -> float: ...
    async def buy_limit(self, qty: float, limit_price: float, timeout_sec: int) -> Fill: ...
    async def place_protective_stop(self, qty: float, stop_price: float) -> str | None: ...
    async def stop_was_triggered(self, order_id: str) -> Fill | None: ...
    async def sell_now(self, qty: float, stop_order_id: str | None, urgent: bool) -> Fill: ...


class PaperBroker:
    """Fills against the live (or synthetic) order book, charging real fees."""

    def __init__(self, settings: Settings, feed: MarketFeed) -> None:
        self._s = settings
        self._feed = feed

    async def min_notional(self) -> float:
        return 5.0

    async def available_cash(self, mandate_equity: float) -> float:
        return mandate_equity

    def normalize_quantity(self, qty: float) -> float:
        return float(f"{qty:.5f}") if qty >= 1e-5 else 0.0

    async def buy_limit(self, qty: float, limit_price: float, timeout_sec: int) -> Fill:
        deadline = asyncio.get_running_loop().time() + timeout_sec
        while True:
            _, asks = self._feed.book()
            if asks and float(asks[0][0]) <= limit_price:
                px = float(asks[0][0])
                return Fill("paper-entry", qty, px, qty * px * self._s.fee_rate)
            if asyncio.get_running_loop().time() >= deadline:
                return Fill(None, 0.0, 0.0, 0.0)
            await asyncio.sleep(1)

    async def place_protective_stop(self, qty: float, stop_price: float) -> str | None:
        return None  # paper stops are enforced by the position manager

    async def stop_was_triggered(self, order_id: str) -> Fill | None:
        return None

    async def sell_now(self, qty: float, stop_order_id: str | None, urgent: bool) -> Fill:
        bids, _ = self._feed.book()
        px = float(bids[0][0])
        if urgent:
            px *= 1 - self._s.max_slippage_pct  # pessimistic slippage on stop exits
        return Fill("paper-exit", qty, px, qty * px * self._s.fee_rate)


class LiveBroker:
    def __init__(self, settings: Settings, feed: MarketFeed) -> None:
        self._s = settings
        self._feed = feed
        self._ex: Any = feed.exchange
        if self._ex is None:
            raise RuntimeError("live broker requires an exchange-backed feed")
        self._market: dict[str, Any] = self._ex.market(settings.symbol)
        self._base: str = self._market["base"]
        self._quote: str = self._market["quote"]

    async def min_notional(self) -> float:
        cost_min = (self._market.get("limits", {}).get("cost", {}) or {}).get("min") or 5.0
        return float(cost_min)

    async def available_cash(self, mandate_equity: float) -> float:
        bal = await with_backoff(self._ex.fetch_balance, "fetch_balance")
        free_quote = float((bal.get(self._quote) or {}).get("free") or 0.0)
        # The firm only ever deploys its own mandate, even if the account holds more.
        return min(free_quote, mandate_equity)

    def normalize_quantity(self, qty: float) -> float:
        return float(self._ex.amount_to_precision(self._s.symbol, qty))  # TRUNCATE by default

    def _fee_usdt(self, order: dict[str, Any], price: float) -> tuple[float, float]:
        """Return (fee in USDT, fee charged in base asset)."""
        usdt, base = 0.0, 0.0
        for fee in order.get("fees") or ([order["fee"]] if order.get("fee") else []):
            cost = float(fee.get("cost") or 0.0)
            cur = fee.get("currency")
            if cur == self._base:
                base += cost
                usdt += cost * price
            elif cur == self._quote:
                usdt += cost
            else:  # e.g. BNB discount: approximate at the configured rate
                usdt += float(order.get("cost") or 0.0) * self._s.fee_rate
        return usdt, base

    async def buy_limit(self, qty: float, limit_price: float, timeout_sec: int) -> Fill:
        sym = self._s.symbol
        amount = self.normalize_quantity(qty)
        price = float(self._ex.price_to_precision(sym, limit_price))
        order = await self._ex.create_order(sym, "limit", "buy", amount, price, {"timeInForce": "GTC"})
        oid = str(order["id"])
        deadline = asyncio.get_running_loop().time() + timeout_sec
        while True:
            order = await with_backoff(lambda: self._ex.fetch_order(oid, sym), "fetch_order")
            if order["status"] == "closed" or asyncio.get_running_loop().time() >= deadline:
                break
            await asyncio.sleep(1)
        if order["status"] != "closed":
            try:
                await self._ex.cancel_order(oid, sym)
            except Exception as exc:  # already filled/cancelled races are fine
                log.info("cancel after timeout: %s", exc)
            order = await with_backoff(lambda: self._ex.fetch_order(oid, sym), "fetch_order")
        filled = float(order.get("filled") or 0.0)
        if filled <= 0:
            return Fill(oid, 0.0, 0.0, 0.0)
        avg = float(order.get("average") or order.get("price") or price)
        fee_usdt, fee_base = self._fee_usdt(order, avg)
        net_qty = self.normalize_quantity(filled - fee_base)
        return Fill(oid, net_qty, avg, fee_usdt)

    async def place_protective_stop(self, qty: float, stop_price: float) -> str | None:
        sym = self._s.symbol
        trigger = float(self._ex.price_to_precision(sym, stop_price))
        limit = float(self._ex.price_to_precision(sym, stop_price * (1 - STOP_LIMIT_OFFSET)))
        try:
            order = await self._ex.create_order(
                sym, "limit", "sell", self.normalize_quantity(qty), limit, {"triggerPrice": trigger}
            )
            return str(order["id"])
        except Exception:
            log.exception("exchange stop rejected — falling back to software stop monitoring")
            return None

    async def stop_was_triggered(self, order_id: str) -> Fill | None:
        order = await with_backoff(lambda: self._ex.fetch_order(order_id, self._s.symbol), "fetch_order")
        if order["status"] != "closed":
            return None
        avg = float(order.get("average") or order.get("price") or 0.0)
        fee_usdt, _ = self._fee_usdt(order, avg)
        return Fill(order_id, float(order.get("filled") or 0.0), avg, fee_usdt)

    async def sell_now(self, qty: float, stop_order_id: str | None, urgent: bool) -> Fill:
        import ccxt

        sym = self._s.symbol
        if stop_order_id:
            try:
                await self._ex.cancel_order(stop_order_id, sym)
            except ccxt.OrderNotFound:
                pass
        amount = self.normalize_quantity(qty)
        legs: list[dict[str, Any]] = []
        if not urgent:
            # Take-profit: marketable IOC limit at the bid, remainder at market.
            bids, _ = self._feed.book()
            px = float(self._ex.price_to_precision(sym, float(bids[0][0])))
            legs.append(await self._ex.create_order(sym, "limit", "sell", amount, px, {"timeInForce": "IOC"}))
        sold = sum(float(o.get("filled") or 0.0) for o in legs)
        remaining = self.normalize_quantity(amount - sold)
        if remaining > 0:
            legs.append(await self._ex.create_order(sym, "market", "sell", remaining))
        await asyncio.sleep(0.5)
        qty_total = cost_total = fee_total = 0.0
        for leg in legs:
            o = await with_backoff(lambda oid=str(leg["id"]): self._ex.fetch_order(oid, sym), "fetch_order")
            filled = float(o.get("filled") or 0.0)
            avg = float(o.get("average") or o.get("price") or 0.0)
            qty_total += filled
            cost_total += filled * avg
            fee_total += self._fee_usdt(o, avg)[0]
        avg_px = cost_total / qty_total if qty_total > 0 else 0.0
        return Fill(str(legs[-1]["id"]), qty_total, avg_px, fee_total)


def build_broker(settings: Settings, feed: MarketFeed) -> Broker:
    return LiveBroker(settings, feed) if settings.is_live else PaperBroker(settings, feed)
