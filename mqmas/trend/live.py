"""Live trend allocator: one tick per run, built to be started by a cron job and exit.

Every run (hourly is plenty):

1. **Stops first.** Paper: the hourly lows since the last run are checked against each crash stop. Live: the
   exchange-side stop orders are queried; a filled one closes the position, a missing one is re-placed.
2. **Safety.** ``TREND_KILL_SWITCH`` or a drawdown of ``TREND_MAX_DRAWDOWN`` from the equity peak sells
   everything and halts (a halt is only cleared by hand: ``python -m mqmas.runner --reset-halt``).
3. **Rebalance once a day**, on the first run after the daily close: the same ``weights_frame`` and
   ``plan_orders`` as the backtest, then every holding's crash stop is ratcheted up and re-placed on Binance.

All state (cash, holdings, stops, blocked coins, the trade log) is one JSON document in the SQLite ``kv``
table, so the process can die at any point between runs; the stop orders resting on Binance protect the
positions while nothing is running.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from mqmas.config import Settings
from mqmas.data.market_feed import with_backoff
from mqmas.execution.spot import Fill, OrderRefused, SpotExecutor, client_id
from mqmas.persistence.db import Database
from mqmas.research.data import is_crypto
from mqmas.trend.committee import Committee
from mqmas.trend.model import CRASH_STOP_ATR, TrendParams, atr, weights_frame
from mqmas.trend.rebalance import MIN_ORDER_USDT, Order, next_stop, plan_orders
from mqmas.trend.specialists import Proposal

log = logging.getLogger(__name__)

STATE_KEY = "trend:state"
DAY_MS = 86_400_000
HOUR_MS = 3_600_000
DAILY_BARS = 1000  # 360-day channel + 365 days of history for eligibility
EVAL_GRACE_MS = 120_000  # wait for Binance to publish the closed daily candle
DUST_USDT = 1.0  # a holding worth less than this is ignored (Binance cannot sell it)
STOP_REPLACE_MIN_MOVE = 0.002  # re-place a stop order only when its level rose by more than 0.2 %
KEEP_TRIPS, KEEP_EQUITY = 300, 800

Notify = Callable[[str], Awaitable[Any]]


@dataclass
class Holding:
    qty: float
    spent: float  # USDT paid into the position (all its buys)
    received: float = 0.0  # USDT taken out of it (all its sells)
    stop: float | None = None
    stop_order: str | None = None
    opened: str = ""


@dataclass
class TrendState:
    cash: float
    holdings: dict[str, Holding] = field(default_factory=dict)
    last_close: str | None = None  # date of the daily close last rebalanced on
    last_check_ms: int = 0
    blocked: dict[str, float] = field(default_factory=dict)  # coin -> signal at its stop-out
    signals: dict[str, float] = field(default_factory=dict)
    targets: dict[str, float] = field(default_factory=dict)
    reviews: dict[str, Any] = field(default_factory=dict)  # last committee review per coin
    peak: float = 0.0
    halted: str | None = None
    trips: list[dict[str, Any]] = field(default_factory=list)
    equity_log: list[list[Any]] = field(default_factory=list)

    @classmethod
    def from_json(cls, raw: str) -> TrendState:
        d = json.loads(raw)
        d["holdings"] = {s: Holding(**h) for s, h in d.get("holdings", {}).items()}
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


def closed_day(now_ms: int) -> str:
    """Date of the most recent daily candle that has closed (and been published) at ``now_ms``."""
    return str(pd.Timestamp((now_ms - EVAL_GRACE_MS) // DAY_MS * DAY_MS - DAY_MS, unit="ms", tz="UTC").date().isoformat())


class TrendAllocator:
    def __init__(self, settings: Settings, db: Database, exchange: Any, notify: Notify | None = None,
                 committee: Committee | None = None) -> None:
        self._s = settings
        self.committee = committee
        self._db = db
        self._ex = exchange
        self._notify = notify
        self.live = settings.is_live and settings.trend_live_enabled
        self._spot = SpotExecutor(exchange, settings.fee_rate)
        self.params = TrendParams(settings.trend_vol_target, settings.trend_top_n, settings.trend_max_weight)
        self.state = TrendState(cash=settings.trend_capital_usdt, peak=settings.trend_capital_usdt)
        self.notes: list[str] = []

    @property
    def mode(self) -> str:
        return "live" if self.live else "paper"

    # ------------------------------------------------------------------ state

    async def load(self) -> TrendState:
        raw = await self._db.get_kv(f"{STATE_KEY}:{self.mode}")
        if raw:
            self.state = TrendState.from_json(raw)
        return self.state

    async def save(self) -> None:
        st = self.state
        st.trips = st.trips[-KEEP_TRIPS:]
        st.equity_log = st.equity_log[-KEEP_EQUITY:]
        await self._db.set_kv(f"{STATE_KEY}:{self.mode}", st.to_json())

    def _note(self, text: str) -> None:
        log.info("trend: %s", text)
        self.notes.append(text)

    # ------------------------------------------------------------------ market data

    async def _prices(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for sym in symbols:
            t = await with_backoff(lambda s=sym: self._ex.fetch_ticker(s), f"ticker {sym}")
            last = float(t.get("last") or t.get("close") or 0.0)
            out[sym] = {"bid": float(t.get("bid") or last), "ask": float(t.get("ask") or last), "last": last}
        return out

    async def _candles(self, symbol: str, tf: str, limit: int, now_ms: int, since: int | None = None) -> pd.DataFrame:
        rows = await with_backoff(lambda: self._ex.fetch_ohlcv(symbol, tf, since=since, limit=limit),
                                  f"ohlcv {symbol} {tf}")
        step = DAY_MS if tf == "1d" else HOUR_MS
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df = df[df["ts"] + step <= now_ms]  # closed candles only
        df.index = pd.to_datetime(df.pop("ts").astype("int64"), unit="ms", utc=True)
        return df.astype(float)

    async def _pool(self) -> list[str]:
        if not self._ex.markets:
            await with_backoff(self._ex.load_markets, "load_markets")
        tickers = await with_backoff(self._ex.fetch_tickers, "fetch_tickers")
        rows: list[tuple[float, str]] = []
        for sym, m in self._ex.markets.items():
            if m.get("spot") and m.get("active") and m.get("quote") == "USDT" and is_crypto(str(m.get("base", ""))):
                rows.append((float((tickers.get(sym) or {}).get("quoteVolume") or 0.0), sym))
        rows.sort(reverse=True)
        pool = [s for _, s in rows[: self._s.trend_pool_size]]
        for sym in ["BTC/USDT", *self.state.holdings]:
            if sym not in pool:
                pool.append(sym)
        return pool

    # ------------------------------------------------------------------ orders

    async def _buy(self, sym: str, value: float, px: dict[str, float]) -> None:
        st = self.state
        if self.live:
            limit = px["ask"] * (1 + self._s.max_slippage_pct)
            try:
                fill = await self._spot.buy(sym, value, limit, client_id("tb"), self._s.entry_order_timeout_sec)
            except OrderRefused as exc:
                self._note(f"compra recusada {sym}: {exc}")
                return
            if fill.quantity <= 0:
                self._note(f"compra de {sym} não executou no limite {limit:.6g}")
                return
            qty, paid = fill.quantity, fill.quantity * fill.avg_price + fill.fee_usdt
            price = fill.avg_price
        else:
            price = px["ask"] * (1 + self._s.max_slippage_pct)
            qty, paid = value * (1 - self._s.fee_rate) / price, value
        st.cash -= paid
        h = st.holdings.get(sym)
        if h is None:
            st.holdings[sym] = Holding(qty, paid, opened=datetime.now(timezone.utc).isoformat())
        else:
            h.qty += qty
            h.spent += paid
        self._note(f"COMPRA {sym} US$ {paid:.2f} @ {price:.6g}")

    async def _sell(self, sym: str, qty: float | None, px: dict[str, float], reason: str) -> None:
        """Sell ``qty`` (None = all) of a holding at market, cancelling its resting stop first."""
        st = self.state
        h = st.holdings[sym]
        q = h.qty if qty is None else min(qty, h.qty)
        if self.live:
            try:
                if qty is None:
                    fill = await self._spot.sell_all(sym, h.qty, h.stop_order, urgent=reason != "rebalance")
                else:
                    if h.stop_order:
                        await self._spot.cancel(sym, h.stop_order)
                        h.stop_order = None
                    fill = await self._spot.sell_part(sym, q)
            except OrderRefused as exc:
                self._note(f"venda recusada {sym}: {exc}")
                return
            h.stop_order = None
            q, got, price = fill.quantity, fill.quantity * fill.avg_price - fill.fee_usdt, fill.avg_price
        else:
            price = px["bid"] * (1 - self._s.max_slippage_pct)
            got = q * price * (1 - self._s.fee_rate)
        self._book_sale(sym, q, got, price, reason)

    def _book_sale(self, sym: str, qty: float, got: float, price: float, reason: str) -> None:
        st = self.state
        h = st.holdings[sym]
        st.cash += got
        h.received += got
        h.qty -= qty
        self._note(f"VENDA {sym} US$ {got:.2f} @ {price:.6g} ({reason})")
        if h.qty * price < DUST_USDT:
            ret = h.received / h.spent - 1 if h.spent > 0 else 0.0
            st.trips.append({"symbol": sym, "opened": h.opened, "closed": datetime.now(timezone.utc).isoformat(),
                             "spent": round(h.spent, 4), "received": round(h.received, 4), "ret": round(ret, 5),
                             "reason": reason})
            del st.holdings[sym]

    # ------------------------------------------------------------------ stops

    async def check_stops(self, now_ms: int) -> None:
        st = self.state
        since = st.last_check_ms or now_ms - HOUR_MS
        for sym in list(st.holdings):
            h = st.holdings[sym]
            if h.stop is None:
                continue
            if self.live:
                await self._check_live_stop(sym, h)
                continue
            bars = await self._candles(sym, "1h", 200, now_ms, since=since // HOUR_MS * HOUR_MS)
            hit = bars[bars["low"] <= h.stop]
            if hit.empty:
                continue
            px = min(float(hit["open"].iloc[0]), h.stop) * (1 - self._s.max_slippage_pct)
            got = h.qty * px * (1 - self._s.fee_rate)
            st.blocked[sym] = st.signals.get(sym, 0.0)
            self._book_sale(sym, h.qty, got, px, "stop")

    async def _check_live_stop(self, sym: str, h: Holding) -> None:
        if not h.stop_order:
            await self._place_stop(sym, h)
            return
        o = await self._spot.order(sym, h.stop_order)
        status = o.get("status")
        if status == "closed":
            fill: Fill = await self._spot.fill_of(sym, o, "sell")
            self.state.blocked[sym] = self.state.signals.get(sym, 0.0)
            h.stop_order = None
            self._book_sale(sym, fill.quantity, fill.quantity * fill.avg_price - fill.fee_usdt, fill.avg_price, "stop")
        elif status in ("canceled", "expired", "rejected"):
            self._note(f"stop de {sym} sumiu da corretora ({status}): recolocando")
            h.stop_order = None
            await self._place_stop(sym, h)

    async def _place_stop(self, sym: str, h: Holding) -> None:
        if h.stop is None:
            return
        px = (await self._prices([sym]))[sym]
        if px["last"] <= h.stop:  # already through the stop: get out now
            await self._sell(sym, None, px, "stop")
            return
        try:
            h.stop_order = await self._spot.place_stop(sym, h.qty, h.stop, client_id("ts"))
        except OrderRefused as exc:
            self._note(f"stop de {sym} não colocado na corretora: {exc}; vigiado a cada execução")

    async def _ratchet_stops(self, daily: dict[str, pd.DataFrame]) -> None:
        for sym, h in list(self.state.holdings.items()):
            d = daily.get(sym)
            if d is None or len(d) < 20:
                continue
            a = float(atr(d).iloc[-1])
            if not np.isfinite(a) or a <= 0:
                continue
            new = next_stop(h.stop, float(d["close"].iloc[-1]), a, CRASH_STOP_ATR)
            moved = h.stop is None or new > h.stop * (1 + STOP_REPLACE_MIN_MOVE)
            h.stop = new
            if self.live and (moved or not h.stop_order):
                if h.stop_order:
                    await self._spot.cancel(sym, h.stop_order)
                    h.stop_order = None
                await self._place_stop(sym, h)

    # ------------------------------------------------------------------ the daily rebalance

    async def rebalance(self, now_ms: int) -> bool:
        st = self.state
        day = closed_day(now_ms)
        pool = await self._pool()
        daily: dict[str, pd.DataFrame] = {}
        for sym in pool:
            try:
                d = await self._candles(sym, "1d", DAILY_BARS, now_ms)
            except Exception:
                log.warning("trend: daily candles unavailable for %s", sym)
                continue
            if len(d) > 60:
                daily[sym] = d
        btc = daily.get("BTC/USDT")
        if btc is None or btc.index[-1].date().isoformat() != day:
            self._note(f"candle diário de {day} ainda não publicado: rebalanceamento adiado")
            return False
        w, sig = weights_frame(daily, self.params)
        last_w, last_sig = w.iloc[-1], sig.iloc[-1]
        for sym in list(st.blocked):  # a fresh breakout (stronger signal) lifts the block
            s = float(last_sig.get(sym, 0.0) or 0.0)
            if s > st.blocked[sym] + 1e-9:
                st.blocked.pop(sym)
            else:
                st.blocked[sym] = min(st.blocked[sym], s)
        st.signals = {s: round(float(x), 4) for s, x in last_sig.items() if np.isfinite(x) and x > 0}
        st.targets = {s: round(float(x), 4) for s, x in last_w.items() if x > 0 and s not in st.blocked}
        px = await self._prices(sorted(set(st.targets) | set(st.holdings)))
        if self.live:
            await self._sync_live_balances()
        values = {s: h.qty * px[s]["last"] for s, h in st.holdings.items()}
        orders = plan_orders(st.targets, values, st.cash, min_order=MIN_ORDER_USDT)
        if self.committee is not None:
            orders = await self._review(day, orders, daily, values, px)
        for o in orders:
            if o.side == "sell":
                qty = None if o.close_all else o.value / px[o.symbol]["last"]
                await self._sell(o.symbol, qty, px[o.symbol], "rebalance")
        for o in orders:
            if o.side == "buy":
                await self._buy(o.symbol, min(o.value, st.cash), px[o.symbol])
        await self._ratchet_stops(daily)
        st.last_close = day
        if not orders:
            self._note(f"rebalanceamento de {day}: nada a ajustar")
        return True

    async def _review(self, day: str, orders: list[Order], daily: dict[str, pd.DataFrame], values: dict[str, float],
                      px: dict[str, dict[str, float]]) -> list[Order]:
        """Send each proposed buy to the committee; only active specialists can shrink or stop it."""
        assert self.committee is not None
        st = self.state
        equity = st.cash + sum(values.values())
        buys = [o.symbol for o in orders if o.side == "buy"]
        await self.committee.fill_outcomes({s: d["close"] for s, d in daily.items()}, day)
        if not buys or equity <= 0:
            return orders
        props = [Proposal(s, st.targets[s], values.get(s, 0.0) / equity, st.signals.get(s, 0.0), px[s]["last"],
                          daily[s]) for s in buys if s in daily]
        reviews = await self.committee.review(day, props)
        st.reviews = {s: {"proposto": r.proposed, "final": round(r.final, 4),
                          "pareceres": [{"especialista": o.specialist, "voto": o.verdict, "motivo": o.reason}
                                        for o in r.opinions]} for s, r in reviews.items()}
        for s, r in reviews.items():
            self._note("COMITÊ " + r.summary)
            st.targets[s] = round(r.final, 4)
        return plan_orders(st.targets, values, st.cash, min_order=MIN_ORDER_USDT)

    async def _sync_live_balances(self) -> None:
        """Never act on more than the account holds: cap cash and holdings by the exchange balances."""
        st = self.state
        bal = await with_backoff(self._ex.fetch_balance, "fetch_balance")
        st.cash = min(st.cash, float((bal.get("USDT") or {}).get("free") or 0.0))
        for sym, h in st.holdings.items():
            base = sym.split("/")[0]
            held = float((bal.get(base) or {}).get("total") or 0.0)
            if held < h.qty * 0.999:
                self._note(f"saldo de {base} menor que o registrado ({held:.8g} < {h.qty:.8g}): ajustado")
                h.qty = held

    # ------------------------------------------------------------------ one run

    async def equity(self) -> tuple[float, dict[str, dict[str, float]]]:
        px = await self._prices(sorted(self.state.holdings))
        return self.state.cash + sum(h.qty * px[s]["last"] for s, h in self.state.holdings.items()), px

    async def liquidate(self, reason: str) -> None:
        px = await self._prices(sorted(self.state.holdings))
        for sym in list(self.state.holdings):
            await self._sell(sym, None, px[sym], reason)

    async def run_once(self, now_ms: int | None = None) -> dict[str, Any]:
        now_ms = now_ms or int(time.time() * 1000)
        st = await self.load()
        if self.live and not self._ex.markets:
            await with_backoff(self._ex.load_markets, "load_markets")
        await self.check_stops(now_ms)
        eq, _ = await self.equity()
        st.peak = max(st.peak, eq)
        dd = 1 - eq / st.peak if st.peak > 0 else 0.0
        if st.halted == "kill_switch" and not self._s.trend_kill_switch:
            st.halted = None
            self._note("kill switch desligado: alocador volta a operar")
        if self._s.trend_kill_switch and (st.holdings or st.halted != "kill_switch"):
            await self.liquidate("kill_switch")
            st.halted = "kill_switch"
            self._note("KILL SWITCH: tudo vendido, alocador parado")
        elif dd >= self._s.trend_max_drawdown and not st.halted:
            await self.liquidate("max_drawdown")
            st.halted = f"max_drawdown {dd:.1%}"
            self._note(f"queda de {dd:.1%} desde o pico: tudo vendido, alocador parado até reset manual")
        elif not st.halted and st.last_close != closed_day(now_ms):
            await self.rebalance(now_ms)
        st.last_check_ms = now_ms
        eq, _ = await self.equity()
        day = closed_day(now_ms)
        if not st.equity_log or st.equity_log[-1][0] != day:
            st.equity_log.append([day, round(eq, 4)])
        else:
            st.equity_log[-1][1] = round(eq, 4)
        await self.save()
        report = self.report(eq)
        if self._notify and self.notes:
            await self._notify(self.render(report))
        return report

    def report(self, equity: float) -> dict[str, Any]:
        st = self.state
        return {"mode": self.mode, "equity": round(equity, 4), "cash": round(st.cash, 4),
                "capital": self._s.trend_capital_usdt, "peak": round(st.peak, 4), "halted": st.halted,
                "last_close": st.last_close, "targets": st.targets, "blocked": st.blocked,
                "holdings": {s: {"qty": h.qty, "stop": h.stop, "stop_order": h.stop_order} for s, h in st.holdings.items()},
                "trips": len(st.trips), "notes": self.notes}

    def render(self, r: dict[str, Any]) -> str:
        ret = r["equity"] / r["capital"] - 1
        lines = [f"Alocador de tendência ({r['mode']}): banca US$ {r['equity']:.2f} ({ret:+.1%}), caixa US$ {r['cash']:.2f}"]
        if r["halted"]:
            lines.append(f"PARADO: {r['halted']}")
        lines += [f"• {n}" for n in self.notes]
        if r["holdings"]:
            lines.append("Posições: " + ", ".join(f"{s} (stop {h['stop']:.6g})" if h["stop"] else s
                                                  for s, h in r["holdings"].items()))
        return "\n".join(lines)
