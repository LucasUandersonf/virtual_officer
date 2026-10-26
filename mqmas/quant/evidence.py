"""Statistical evidence for the mean-reversion setup.

1. Event study (walk-forward, no look-ahead) on the cached candle history:
   every *fresh* cross of z ≤ zscore_entry is treated as an entry at that
   bar's close, with the exact same ATR stop / VWAP target rules used live.
   Each event is resolved bar-by-bar; if stop and target fall inside the same
   candle the **stop is assumed first** (pessimistic convention).
2. Realised trades from SQLite (paper or live, both are forward
   out-of-sample) are merged in.
3. The hit rate is a Beta(1+wins, 1+losses) posterior — a uniform prior that
   shrinks small samples towards 50% instead of trusting 3/3 = 100%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from mqmas.config import Settings
from mqmas.models import SetupStats
from mqmas.quant.indicators import atr, rolling_zscore, session_vwap
from mqmas.quant.sizing import kelly_fraction, stop_pct_from_atr, take_profit_pct, trade_sharpe, unit_economics

MAX_HOLD_BARS = 32  # 8h on 15m candles; unresolved trades are marked to market


@dataclass(frozen=True)
class BacktestResult:
    trades: int
    wins: int
    net_returns: tuple[float, ...]


def event_study(df: pd.DataFrame, s: Settings) -> BacktestResult:
    z = rolling_zscore(df["close"], s.zscore_window)["zscore"].to_numpy()
    vw = session_vwap(df)["vwap"].to_numpy()
    a = atr(df).to_numpy()
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(df)
    returns: list[float] = []
    i = s.zscore_window
    while i < n - 1:
        fresh_cross = z[i] <= s.zscore_entry and not (z[i - 1] <= s.zscore_entry)
        if not fresh_cross or np.isnan(a[i]) or np.isnan(vw[i]):
            i += 1
            continue
        entry = close[i]
        sp = stop_pct_from_atr(a[i] / entry, s)
        tp = take_profit_pct(entry, vw[i], s)
        loss_u, gain_u = unit_economics(sp, tp, s)
        stop_px, tp_px = entry * (1 - sp), entry * (1 + tp)
        outcome: float | None = None
        j = i + 1
        last = min(i + MAX_HOLD_BARS, n - 1)
        while j <= last:
            if low[j] <= stop_px:
                outcome = -loss_u
                break
            if high[j] >= tp_px:
                outcome = gain_u
                break
            j += 1
        if outcome is None:
            j = last
            outcome = (close[j] - entry) / entry - 2 * s.fee_rate - s.max_slippage_pct
        returns.append(outcome)
        i = j + 1  # one position at a time: no overlapping events
    wins = sum(1 for r in returns if r > 0)
    return BacktestResult(trades=len(returns), wins=wins, net_returns=tuple(returns))


def compute_setup_stats(
    backtest: BacktestResult,
    live_trades: int,
    live_wins: int,
    stop_pct: float,
    tp_pct: float,
    s: Settings,
) -> SetupStats:
    wins = backtest.wins + live_wins
    losses = (backtest.trades - backtest.wins) + (live_trades - live_wins)
    a, b = 1 + wins, 1 + losses
    p = a / (a + b)
    p_lo = float(stats.beta.ppf(0.025, a, b))
    loss_u, gain_u = unit_economics(stop_pct, tp_pct, s)
    payoff = gain_u / loss_u if loss_u > 0 else 0.0
    expected, sharpe = trade_sharpe(p, gain_u, loss_u)
    k_full = kelly_fraction(p, payoff)
    return SetupStats(
        backtest_trades=backtest.trades,
        backtest_wins=backtest.wins,
        live_trades=live_trades,
        live_wins=live_wins,
        win_rate=p,
        win_rate_lower_95=p_lo,
        payoff_ratio=payoff,
        expected_return_pct=expected,
        trade_sharpe=sharpe,
        kelly_full=k_full,
        kelly_applied=max(0.0, k_full * s.kelly_fraction),
    )
