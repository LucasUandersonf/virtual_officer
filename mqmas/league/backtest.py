"""Vectorised bracket backtest and the walk-forward tournament.

Execution model (mirrors live trading, pessimistic where it is ambiguous):

* Signal on the close of bar ``i`` → entry at the **open of bar i+1**.
* The bar range from the entry bar onward is scanned for the stop and the
  target; if both fall in the same candle the **stop counts first**.
* After ``max_hold`` bars the trade is closed at that bar's close (time stop,
  also enforced live by the position manager).
* Costs per trade: two taker fees, plus slippage on stop and time exits
  (same arithmetic as :func:`mqmas.quant.sizing.unit_economics`).
* One position at a time: a signal while a trade is open is skipped.

Walk-forward (anchored): the history after warm-up is cut into ``K`` equal
segments. For fold ``k`` the exit profile is chosen on segments ``0..k-1``
and the setup is scored on segment ``k`` only. Only these out-of-sample trades
decide approval, which is the main defence against overfitting when many
setups × profiles are tried.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

from mqmas.config import Settings
from mqmas.league.setups import SETUPS, WARMUP_BARS, Setup, features
from mqmas.models import ExitProfile, LeagueEntry, LeagueReport, SetupMetrics

TP_GRID = (0.006, 0.007, 0.008)
STOP_MULT_GRID = (1.0, 1.5, 2.5)
DEFAULT_PROFILE = ExitProfile(tp_pct=0.007, stop_atr_mult=1.5)
MIN_TRAIN_TRADES = 10
PF_CAP = 99.0  # keeps JSON finite when a sample has no losses
TIMEFRAME_MIN = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}


def max_hold_bars(s: Settings) -> int:
    return max(1, s.max_hold_minutes // TIMEFRAME_MIN.get(s.timeframe, 15))


@dataclass(frozen=True)
class Trades:
    entry_idx: np.ndarray
    returns: np.ndarray

    def between(self, lo: int, hi: int) -> np.ndarray:
        return self.returns[(self.entry_idx >= lo) & (self.entry_idx < hi)]


def simulate(f: pd.DataFrame, sig: np.ndarray, profile: ExitProfile, s: Settings, hold: int) -> Trades:
    o = f["open"].to_numpy()
    h = f["high"].to_numpy()
    lo = f["low"].to_numpy()
    c = f["close"].to_numpy()
    atr_pct = f["atr_pct"].to_numpy()
    n = len(f)
    idx = np.flatnonzero(sig)
    idx = idx[(idx + hold < n) & ~np.isnan(atr_pct[idx])]
    if idx.size == 0:
        return Trades(np.array([], dtype=int), np.array([]))
    e = idx + 1
    sp = np.clip(profile.stop_atr_mult * atr_pct[idx], s.stop_loss_min_pct, s.stop_loss_max_pct)
    tp = profile.tp_pct
    entry = o[e]
    win_l = sliding_window_view(lo, hold)[e]
    win_h = sliding_window_view(h, hold)[e]
    hit_s = win_l <= (entry * (1 - sp))[:, None]
    hit_t = win_h >= (entry * (1 + tp))[:, None]
    first_s = np.where(hit_s.any(axis=1), hit_s.argmax(axis=1), hold)
    first_t = np.where(hit_t.any(axis=1), hit_t.argmax(axis=1), hold)
    stopped = (first_s < hold) & (first_s <= first_t)
    targeted = ~stopped & (first_t < hold)
    mtm = c[e + hold - 1] / entry - 1
    fees = 2 * s.fee_rate
    ret = np.where(stopped, -sp - fees - s.max_slippage_pct,
                   np.where(targeted, tp - fees, mtm - fees - s.max_slippage_pct))
    exit_idx = e + np.where(stopped, first_s, np.where(targeted, first_t, hold - 1))
    keep: list[int] = []
    busy_until = -1
    for k in range(e.size):  # one position at a time
        if e[k] > busy_until:
            keep.append(k)
            busy_until = int(exit_idx[k])
    sel = np.asarray(keep, dtype=int)
    return Trades(e[sel], ret[sel])


def metrics(returns: np.ndarray) -> SetupMetrics:
    n = int(returns.size)
    if n == 0:
        return SetupMetrics()
    wins_mask = returns > 0
    w = int(wins_mask.sum())
    a, b = 1 + w, 1 + (n - w)
    gains, losses = returns[wins_mask], -returns[~wins_mask]
    gross_win, gross_loss = float(gains.sum()), float(losses.sum())
    avg_win = float(gains.mean()) if gains.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0
    breakeven = avg_loss / (avg_win + avg_loss) if (avg_win + avg_loss) > 0 else 1.0
    equity = np.cumsum(returns)
    drawdown = float((np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:] - equity).max())
    sd = float(returns.std(ddof=1)) if n > 1 else 0.0
    return SetupMetrics(
        trades=n, wins=w, win_rate=a / (a + b),
        win_rate_lower_95=float(stats.beta.ppf(0.025, a, b)),
        expectancy_pct=float(returns.mean()),
        profit_factor=min(gross_win / gross_loss, PF_CAP) if gross_loss > 0 else (PF_CAP if gross_win > 0 else 0.0),
        max_drawdown_pct=drawdown,
        sharpe=float(returns.mean()) / sd if sd > 0 else 0.0,
        total_return_pct=float(returns.sum()),
        p_edge=float(1 - stats.beta.cdf(breakeven, a, b)),
    )


def _best_profile(sims: dict[ExitProfile, Trades], lo: int, hi: int, min_trades: int) -> ExitProfile:
    best, best_mean = DEFAULT_PROFILE, -math.inf
    for prof, tr in sims.items():
        r = tr.between(lo, hi)
        if r.size >= min_trades and r.mean() > best_mean:
            best, best_mean = prof, float(r.mean())
    return best


def _verdict(oos: SetupMetrics, folds: list[float], s: Settings) -> list[str]:
    reasons: list[str] = []
    if oos.trades < s.min_sample_size:
        reasons.append(f"amostra fora da amostra n={oos.trades} < {s.min_sample_size}")
    if oos.expectancy_pct <= 0:
        reasons.append(f"expectativa líquida {oos.expectancy_pct:+.3%} por trade (precisa > 0)")
    if oos.profit_factor < s.league_min_profit_factor:
        reasons.append(f"profit factor {oos.profit_factor:.2f} < {s.league_min_profit_factor}")
    if oos.win_rate < s.min_win_rate:
        reasons.append(f"taxa de acerto {oos.win_rate:.1%} < mandato {s.min_win_rate:.0%}")
    traded = [x for x in folds if x != 0.0]
    if traded and sum(1 for x in traded if x > 0) < math.ceil(len(traded) / 2):
        reasons.append(f"positivo em só {sum(1 for x in traded if x > 0)}/{len(traded)} janelas de teste")
    return reasons


def evaluate_setup(setup: Setup, f: pd.DataFrame, s: Settings, edges: np.ndarray, hold: int) -> LeagueEntry:
    sig = setup.signals(f)
    sims = {ExitProfile(tp_pct=tp, stop_atr_mult=m): simulate(f, sig, ExitProfile(tp_pct=tp, stop_atr_mult=m), s, hold)
            for tp in TP_GRID for m in STOP_MULT_GRID}
    oos_parts: list[np.ndarray] = []
    folds: list[float] = []
    for k in range(1, len(edges) - 1):
        prof = _best_profile(sims, int(edges[0]), int(edges[k]), MIN_TRAIN_TRADES)
        r = sims[prof].between(int(edges[k]), int(edges[k + 1]))
        oos_parts.append(r)
        folds.append(float(r.sum()))
    oos = metrics(np.concatenate(oos_parts) if oos_parts else np.array([]))
    live_profile = _best_profile(sims, int(edges[0]), int(edges[-1]), s.min_sample_size)
    in_sample = metrics(sims[live_profile].between(int(edges[0]), int(edges[-1])))
    reasons = _verdict(oos, folds, s)
    score = oos.sharpe * math.sqrt(oos.trades) if oos.trades > 1 else 0.0
    fired = np.flatnonzero(sig)
    last = f.index[fired[-1]].to_pydatetime() if fired.size else None
    return LeagueEntry(key=setup.key, name=setup.name, family=setup.family, thesis=setup.thesis,
                       profile=live_profile, oos=oos, in_sample=in_sample, folds=folds,
                       approved=not reasons, reasons=reasons, score=score, last_signal=last)


def run_tournament(df: pd.DataFrame, s: Settings) -> LeagueReport:
    hold = max_hold_bars(s)
    f = features(df)
    start = min(WARMUP_BARS, len(f))
    edges = np.linspace(start, len(f), s.league_folds + 1).astype(int)
    entries = [evaluate_setup(st, f, s, edges, hold) for st in SETUPS]
    entries.sort(key=lambda e: (e.approved, e.score), reverse=True)
    for i, e in enumerate(entries, 1):
        e.rank = i
    return LeagueReport(symbol=s.symbol, timeframe=s.timeframe, bars=len(df),
                        start=df.index[0].to_pydatetime() if len(df) else None,
                        end=df.index[-1].to_pydatetime() if len(df) else None,
                        folds=s.league_folds - 1, max_hold_bars=hold, entries=entries)
