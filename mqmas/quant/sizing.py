"""Position sizing: fractional Kelly bounded by fixed-fractional risk.

Spot-only arithmetic (no leverage). Per unit of notional:

    loss_at_stop = stop% + fee_in + fee_out + slippage
    gain_at_tp   = tp%   - fee_in - fee_out

Kelly for a binary bet with win prob p and payoff ratio b = gain/loss:

    f* = p - (1 - p) / b

``f*`` is the fraction of equity to put *at risk*. We apply a fraction of it
(quarter-Kelly by default, robust to estimation error in p) and clip it to the
mandate's hard ceiling of 0.6% of equity:

    risk_fraction = min(kelly_fraction · f*, max_risk_per_trade)
    notional      = equity · risk_fraction / loss_at_stop   (≤ cash; no leverage)
"""

from __future__ import annotations

import math

from mqmas.config import Settings
from mqmas.models import SizingPlan

CASH_BUFFER = 0.98  # never commit 100% of cash: rounding + fee headroom


def kelly_fraction(p: float, payoff_ratio: float) -> float:
    if payoff_ratio <= 0 or not 0.0 <= p <= 1.0:
        return 0.0
    return p - (1.0 - p) / payoff_ratio


def stop_pct_from_atr(atr_pct: float, s: Settings) -> float:
    """ATR-dynamic stop, clamped into [stop_loss_min_pct, stop_loss_max_pct]."""
    raw = s.stop_atr_multiplier * atr_pct
    return min(max(raw, s.stop_loss_min_pct), s.stop_loss_max_pct)


def take_profit_pct(price: float, vwap: float, s: Settings) -> float:
    """Target the reversion to VWAP, bounded by the mandate's [0.6%, 0.8%] band."""
    to_vwap = (vwap - price) / price if price > 0 else 0.0
    return min(max(to_vwap, s.take_profit_min_pct), s.take_profit_max_pct)


def unit_economics(stop_pct: float, tp_pct: float, s: Settings) -> tuple[float, float]:
    """(loss_per_unit_notional, gain_per_unit_notional) net of all costs."""
    loss = stop_pct + 2 * s.fee_rate + s.max_slippage_pct
    gain = tp_pct - 2 * s.fee_rate
    return loss, gain


def build_sizing_plan(
    equity: float,
    entry_price: float,
    stop_pct: float,
    tp_pct: float,
    kelly_applied: float,
    s: Settings,
) -> SizingPlan:
    if equity <= 0 or entry_price <= 0:
        raise ValueError("equity and entry price must be positive")
    loss_u, gain_u = unit_economics(stop_pct, tp_pct, s)
    risk_fraction = max(0.0, min(kelly_applied, s.max_risk_per_trade_pct))
    risk_budget = equity * risk_fraction
    notional = min(risk_budget / loss_u, equity * CASH_BUFFER) if risk_budget > 0 else 0.0
    quantity = notional / entry_price
    return SizingPlan(
        equity=equity,
        risk_budget_usdt=risk_budget,
        entry_price=entry_price,
        stop_price=entry_price * (1 - stop_pct),
        take_profit_price=entry_price * (1 + tp_pct),
        stop_pct=stop_pct,
        take_profit_pct=tp_pct,
        quantity=quantity,
        notional=notional,
        max_loss_usdt=notional * loss_u,
        expected_gain_usdt=notional * gain_u,
    )


def trade_sharpe(p: float, gain: float, loss: float) -> tuple[float, float]:
    """Per-trade expected return and Sharpe of a two-outcome bet.

    E = p·g − (1−p)·l,  σ = sqrt(p(1−p))·(g + l),  SR = E/σ.
    """
    expected = p * gain - (1 - p) * loss
    sigma = math.sqrt(max(p * (1 - p), 0.0)) * (gain + loss)
    return expected, (expected / sigma if sigma > 0 else 0.0)
