"""Order-book microstructure: imbalance, depth delta, microprice, slippage."""

from __future__ import annotations

from collections.abc import Sequence

from mqmas.models import OrderBookMetrics

Level = Sequence[float]  # [price, amount, ...] as returned by ccxt


def _top(levels: Sequence[Level], n: int) -> list[tuple[float, float]]:
    return [(float(lv[0]), float(lv[1])) for lv in levels[:n]]


def orderbook_metrics(
    bids: Sequence[Level],
    asks: Sequence[Level],
    levels: int = 5,
    previous_imbalance: float | None = None,
) -> OrderBookMetrics:
    """Compute top-of-book pressure.

    * imbalance  = (Vb - Va) / (Vb + Va) over the first ``levels`` levels.
      +1 → only bids (buy pressure), -1 → only asks.
    * microprice = (Pa*Vb1 + Pb*Va1) / (Vb1 + Va1): the fair price implied by
      level-1 queue sizes (Stoikov, 2018).
    * depth_delta = imbalance_t - imbalance_{t-1}: acceleration of pressure.
    """
    b = _top(bids, levels)
    a = _top(asks, levels)
    if not b or not a:
        raise ValueError("order book side is empty")
    best_bid, best_ask = b[0][0], a[0][0]
    if best_ask <= best_bid:
        raise ValueError(f"crossed book: bid={best_bid} ask={best_ask}")
    vb = sum(q for _, q in b)
    va = sum(q for _, q in a)
    total = vb + va
    imbalance = (vb - va) / total if total > 0 else 0.0
    mid = (best_bid + best_ask) / 2.0
    q_b1, q_a1 = b[0][1], a[0][1]
    micro = (best_ask * q_b1 + best_bid * q_a1) / (q_b1 + q_a1) if (q_b1 + q_a1) > 0 else mid
    return OrderBookMetrics(
        best_bid=best_bid,
        best_ask=best_ask,
        mid=mid,
        spread_pct=(best_ask - best_bid) / mid,
        bid_volume=vb,
        ask_volume=va,
        imbalance=imbalance,
        bid_ask_ratio=vb / va if va > 0 else float("inf"),
        depth_delta=0.0 if previous_imbalance is None else imbalance - previous_imbalance,
        microprice=micro,
    )


def estimate_buy_slippage(asks: Sequence[Level], quantity: float) -> tuple[float, float]:
    """Walk the ask side to fill ``quantity``.

    Returns ``(average_fill_price, slippage_pct_vs_best_ask)``. Raises if the
    visible book cannot absorb the order (a liquidity red flag on its own).
    """
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    remaining = quantity
    cost = 0.0
    best = float(asks[0][0])
    for lv in asks:
        price, size = float(lv[0]), float(lv[1])
        take = min(size, remaining)
        cost += take * price
        remaining -= take
        if remaining <= 1e-15:
            break
    if remaining > 1e-15:
        raise ValueError("insufficient visible liquidity for requested size")
    avg = cost / quantity
    return avg, (avg - best) / best
