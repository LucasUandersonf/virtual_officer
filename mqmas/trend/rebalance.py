"""From target weights to orders: the one place where a small account meets the exchange's rules.

Shared by the backtest and the live allocator, so the simulated account trades exactly like the real one:

* Binance refuses orders below US$ 5 (``min_order``): a buy smaller than that is skipped, and a sell that
  would leave less than that behind sells the whole position instead (dust cannot be sold later).
* Tiny weight drifts are not traded (``band``): a position is resized only when it is off its target by
  more than ``band`` of the target. This cuts turnover, and fees, by an order of magnitude.
* Sells go first, so their cash funds the buys; buys never use more cash than there is.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_ORDER_USDT = 5.0
BAND = 0.25


@dataclass(frozen=True)
class Order:
    symbol: str
    side: str  # "buy" | "sell"
    value: float  # USDT notional
    close_all: bool = False


def plan_orders(targets: dict[str, float], values: dict[str, float], cash: float, min_order: float = MIN_ORDER_USDT,
                band: float = BAND) -> list[Order]:
    """Orders that move ``values`` (USDT per coin) towards ``targets`` (weights of total equity)."""
    equity = cash + sum(values.values())
    if equity <= 0:
        return []
    sells: list[Order] = []
    buys: list[Order] = []
    for sym in sorted(set(targets) | set(values)):
        want = max(targets.get(sym, 0.0), 0.0) * equity
        have = values.get(sym, 0.0)
        if want < min_order:  # too small to hold: flat
            if have > 0:
                sells.append(Order(sym, "sell", have, close_all=True))
            continue
        diff = want - have
        if abs(diff) < max(min_order, band * want):
            continue
        if diff < 0:
            close_all = have + diff < min_order
            sells.append(Order(sym, "sell", have if close_all else -diff, close_all))
        else:
            buys.append(Order(sym, "buy", diff))
    budget = cash + sum(o.value for o in sells)
    out = list(sells)
    for o in sorted(buys, key=lambda o: -o.value):
        v = min(o.value, budget)
        if v >= min_order:
            out.append(Order(o.symbol, "buy", v))
            budget -= v
    return out


def next_stop(prev: float | None, close: float, atr: float, mult: float) -> float:
    """Crash stop: ``mult`` ATR under the close, ratcheted so it only moves up."""
    level = close - mult * atr
    return level if prev is None else max(prev, level)
