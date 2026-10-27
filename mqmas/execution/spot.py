"""Live spot execution for any symbol (the strategy book's executor).

Guarantees, in order of importance:

* **The stop lives on the exchange.** Right after a fill a STOP_LOSS_LIMIT sell is placed on Binance,
  so the downside is protected even if this process, Railway or the network dies.
* **Idempotency.** Every order carries our own ``clientOrderId`` (stored in the database *before*
  the order is sent); after a crash or a timeout the order is looked up by that id instead of being
  sent again, so a retry can never double a position.
* **Exchange filters.** Quantity is truncated to the lot step, prices rounded to the tick, and an
  order below the symbol's minimum notional is refused here instead of rejected by the exchange.
* **Fees in the right currency.** A fee charged in the base asset reduces the quantity we hold (and
  can sell); a fee in USDT or BNB is converted to USDT.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Any

from mqmas.data.market_feed import with_backoff

log = logging.getLogger(__name__)

STOP_LIMIT_OFFSET = 0.005  # the stop's limit sits 0.5 % below its trigger so it fills in a fast drop
FILL_POLL_SEC = 1.0


class OrderRefused(Exception):
    """Refused locally (filters, balance); nothing was sent to the exchange."""


@dataclass(frozen=True)
class Fill:
    order_id: str | None
    quantity: float  # base quantity we now hold (buy) or sold (sell), net of base-asset fees
    avg_price: float
    fee_usdt: float
    status: str = "closed"


def client_id(tag: str) -> str:
    """Binance accepts ``[.A-Za-z0-9:/_-]{1,36}``; unique per order."""
    return f"mq{tag}{int(time.time())}{secrets.token_hex(3)}"[:36]


class SpotExecutor:
    def __init__(self, exchange: Any, fee_rate: float) -> None:
        self._ex = exchange
        self._fee_rate = fee_rate

    # ------------------------------------------------------------------ filters

    def min_cost(self, symbol: str) -> float:
        m = self._ex.market(symbol)
        return float(((m.get("limits") or {}).get("cost") or {}).get("min") or 5.0)

    def amount(self, symbol: str, qty: float) -> float:
        return float(self._ex.amount_to_precision(symbol, qty)) if qty > 0 else 0.0  # truncates

    def price(self, symbol: str, px: float) -> float:
        return float(self._ex.price_to_precision(symbol, px))

    def _fees(self, symbol: str, order: dict[str, Any], px: float) -> tuple[float, float]:
        m = self._ex.market(symbol)
        usdt = base = 0.0
        for fee in order.get("fees") or ([order["fee"]] if order.get("fee") else []):
            cost, cur = float(fee.get("cost") or 0.0), fee.get("currency")
            if cur == m["base"]:
                base += cost
                usdt += cost * px
            elif cur == m["quote"]:
                usdt += cost
            elif cost:
                usdt += float(order.get("cost") or 0.0) * self._fee_rate  # BNB discount: approximate
        return usdt, base

    def _fill(self, symbol: str, order: dict[str, Any], side: str) -> Fill:
        filled = float(order.get("filled") or 0.0)
        avg = float(order.get("average") or order.get("price") or 0.0)
        fee_usdt, fee_base = self._fees(symbol, order, avg)
        qty = self.amount(symbol, filled - fee_base) if side == "buy" else filled
        return Fill(str(order.get("id")) if order.get("id") else None, qty if filled else 0.0, avg, fee_usdt,
                    str(order.get("status") or ""))

    # ------------------------------------------------------------------ queries

    async def free_quote(self, quote: str = "USDT") -> float:
        bal = await with_backoff(self._ex.fetch_balance, "fetch_balance")
        return float((bal.get(quote) or {}).get("free") or 0.0)

    async def find(self, symbol: str, cid: str) -> dict[str, Any] | None:
        """Look an order up by our clientOrderId (None if the exchange never received it)."""
        import ccxt

        try:
            return dict(await with_backoff(
                lambda: self._ex.fetch_order(None, symbol, {"origClientOrderId": cid}), f"find {cid}"))
        except ccxt.OrderNotFound:
            return None

    async def order(self, symbol: str, order_id: str) -> dict[str, Any]:
        return dict(await with_backoff(lambda: self._ex.fetch_order(order_id, symbol), "fetch_order"))

    async def fill_of(self, symbol: str, order: dict[str, Any], side: str) -> Fill:
        return self._fill(symbol, order, side)

    # ------------------------------------------------------------------ orders

    async def buy(self, symbol: str, notional: float, limit_price: float, cid: str, timeout_sec: int) -> Fill:
        """Marketable limit buy; whatever is unfilled at the timeout is cancelled."""
        px = self.price(symbol, limit_price)
        qty = self.amount(symbol, notional / px)
        if qty <= 0 or qty * px < self.min_cost(symbol):
            raise OrderRefused(f"{symbol}: US$ {qty * px:.2f} abaixo do mínimo de US$ {self.min_cost(symbol):g}")
        try:
            order = await self._ex.create_order(symbol, "limit", "buy", qty, px,
                                                {"timeInForce": "GTC", "newClientOrderId": cid})
        except Exception:
            # timeout/disconnect: the order may or may not exist — ask by our id, never resend blindly
            log.exception("buy %s: create_order failed, looking it up by clientOrderId", symbol)
            found = await self.find(symbol, cid)
            if found is None:
                raise
            order = found
        oid = str(order["id"])
        deadline = asyncio.get_running_loop().time() + timeout_sec
        while order.get("status") not in ("closed", "canceled", "expired", "rejected"):
            if asyncio.get_running_loop().time() >= deadline:
                try:
                    await self._ex.cancel_order(oid, symbol)
                except Exception as exc:  # filled/cancelled in between: fine
                    log.info("cancel %s after timeout: %s", oid, exc)
                order = await self.order(symbol, oid)
                break
            await asyncio.sleep(FILL_POLL_SEC)
            order = await self.order(symbol, oid)
        return self._fill(symbol, order, "buy")

    async def place_stop(self, symbol: str, qty: float, stop_price: float, cid: str) -> str:
        """Exchange-native STOP_LOSS_LIMIT sell. Raises if the exchange refuses it."""
        trigger = self.price(symbol, stop_price)
        limit = self.price(symbol, stop_price * (1 - STOP_LIMIT_OFFSET))
        amount = self.amount(symbol, qty)
        if amount <= 0 or amount * limit < self.min_cost(symbol):
            raise OrderRefused(f"{symbol}: stop de US$ {amount * limit:.2f} abaixo do mínimo da Binance")
        try:
            order = await self._ex.create_order(symbol, "limit", "sell", amount, limit,
                                                {"stopLossPrice": trigger, "timeInForce": "GTC",
                                                 "newClientOrderId": cid})
        except Exception:
            found = await self.find(symbol, cid)
            if found is None:
                raise
            order = found
        return str(order["id"])

    async def cancel(self, symbol: str, order_id: str) -> None:
        import ccxt

        try:
            await self._ex.cancel_order(order_id, symbol)
        except ccxt.OrderNotFound:
            pass

    async def sell_all(self, symbol: str, qty: float, stop_order_id: str | None, urgent: bool) -> Fill:
        """Cancel the resting stop, then sell: IOC at the bid first (unless urgent), rest at market."""
        legs: list[dict[str, Any]] = []
        if stop_order_id:
            await self.cancel(symbol, stop_order_id)
            stop = await self.order(symbol, stop_order_id)
            if stop.get("status") == "closed":  # it fired while we were cancelling
                return self._fill(symbol, stop, "sell")
            if float(stop.get("filled") or 0.0) > 0:  # partly filled before the cancel
                legs.append(stop)
                qty -= float(stop["filled"])
        amount = self.amount(symbol, qty)
        if not urgent and amount > 0:
            t = await with_backoff(lambda: self._ex.fetch_ticker(symbol), f"ticker {symbol}")
            bid = self.price(symbol, float(t.get("bid") or t.get("last")))
            legs.append(await self._ex.create_order(symbol, "limit", "sell", amount, bid,
                                                    {"timeInForce": "IOC", "newClientOrderId": client_id("x")}))
        sold = sum(float(o.get("filled") or 0.0) for o in legs if str(o.get("id")) != stop_order_id)
        rest = self.amount(symbol, amount - sold)
        if rest > 0:
            legs.append(await self._ex.create_order(symbol, "market", "sell", rest, None,
                                                    {"newClientOrderId": client_id("m")}))
        qty_total = cost = fees = 0.0
        last_id = None
        for leg in legs:
            o = await self.order(symbol, str(leg["id"]))
            f = self._fill(symbol, o, "sell")
            qty_total += f.quantity
            cost += f.quantity * f.avg_price
            fees += f.fee_usdt
            last_id = f.order_id
        return Fill(last_id, qty_total, cost / qty_total if qty_total else 0.0, fees)
