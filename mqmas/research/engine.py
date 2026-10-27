"""Trade generation and portfolio simulation.

Per-asset bracket simulation (vectorised, path-aware):

* signal on the close of bar ``i`` → entry at the open of ``i+1``;
* initial stop ``entry − k_s·ATR`` and optional target ``entry + k_t·ATR``;
* optional chandelier trail ``max(high since entry, excluding the current
  bar) − k_tr·ATR``, which only ever tightens the stop;
* **gaps are honoured**: if a bar opens beyond the stop (flash crash) the
  fill is the open, not the stop price;
* stop and target in the same bar → stop first (pessimistic);
* time stop at the close of the last bar of the holding window;
* one position per asset at a time.

The portfolio simulator then replays every asset's trades on one shared
equity, applying fixed-fractional risk sizing, max concurrent positions,
cash limits, minimum notional, a daily loss limit and the "2 losses in a row
→ 24 h lockout" rule.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

REASON_STOP, REASON_TARGET, REASON_TIME = 0, 1, 2


@dataclass(frozen=True)
class Exit:
    stop_atr: float
    target_atr: float | None
    trail_atr: float | None
    hold: int


@dataclass(frozen=True)
class Costs:
    fee: float = 0.001  # per side, Binance spot standard tier
    slippage: float = 0.0005  # per side

    @property
    def round_trip(self) -> float:
        return 2 * (self.fee + self.slippage)


def bracket_trades(o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray, atr: np.ndarray,
                   sig: np.ndarray, ex: Exit, costs: Costs) -> dict[str, np.ndarray]:
    n = o.size
    hold = ex.hold
    idx = np.flatnonzero(sig)
    idx = idx[(idx + hold < n) & np.isfinite(atr[idx]) & (atr[idx] > 0)]
    empty = {k: np.array([]) for k in ("entry", "exit", "ret", "stop_pct", "reason")}
    if idx.size == 0:
        return empty
    e = idx + 1
    a = atr[idx]
    entry = o[e]
    stop0 = entry - ex.stop_atr * a
    ok = stop0 > 0
    e, a, entry, stop0 = e[ok], a[ok], entry[ok], stop0[ok]
    if e.size == 0:
        return empty
    wo = sliding_window_view(o, hold)[e]
    wh = sliding_window_view(h, hold)[e]
    wl = sliding_window_view(lo, hold)[e]
    stop = np.repeat(stop0[:, None], hold, axis=1)
    if ex.trail_atr is not None:
        prev_max = np.maximum.accumulate(wh, axis=1)
        prev_max = np.concatenate([entry[:, None], prev_max[:, :-1]], axis=1)
        stop = np.maximum(stop, np.maximum.accumulate(prev_max, axis=1) - ex.trail_atr * a[:, None])
    hit_s = wl <= stop
    first_s = np.where(hit_s.any(axis=1), hit_s.argmax(axis=1), hold)
    if ex.target_atr is not None:
        tgt = entry + ex.target_atr * a
        hit_t = wh >= tgt[:, None]
        first_t = np.where(hit_t.any(axis=1), hit_t.argmax(axis=1), hold)
    else:
        tgt = np.full(e.size, np.inf)
        first_t = np.full(e.size, hold)
    stopped = (first_s < hold) & (first_s <= first_t)
    targeted = ~stopped & (first_t < hold)
    rows = np.arange(e.size)
    js = np.minimum(first_s, hold - 1)
    jt = np.minimum(first_t, hold - 1)
    stop_fill = np.minimum(wo[rows, js], stop[rows, js])  # gap through the stop fills at the open
    tgt_fill = np.maximum(wo[rows, jt], tgt)
    time_fill = c[e + hold - 1]
    px = np.where(stopped, stop_fill, np.where(targeted, tgt_fill, time_fill))
    exit_bar = e + np.where(stopped, first_s, np.where(targeted, first_t, hold - 1))
    ret = px / entry - 1 - costs.round_trip
    reason = np.where(stopped, REASON_STOP, np.where(targeted, REASON_TARGET, REASON_TIME))
    keep: list[int] = []
    busy = -1
    for k in range(e.size):
        if e[k] > busy:
            keep.append(k)
            busy = int(exit_bar[k])
    s = np.asarray(keep, dtype=int)
    return {"entry": e[s], "exit": exit_bar[s], "ret": ret[s], "stop_pct": ((entry - stop0) / entry)[s],
            "reason": reason[s]}


# --------------------------------------------------------------------------- portfolio


@dataclass(frozen=True)
class RiskRules:
    equity0: float = 50.0
    risk_per_trade: float = 0.006
    max_positions: int = 3
    daily_loss_limit: float = 0.015
    loss_streak_lock: int = 2
    lock_hours: float = 24.0
    min_notional: float = 5.0
    # Dynamic sizing: halve risk while equity sits more than ``dd_throttle`` below its peak (None = off).
    # Per-trade scaling (e.g. volatility targeting) comes from an optional ``risk_mult`` trade column.
    dd_throttle: float | None = None


@dataclass
class PortfolioResult:
    equity: pd.Series
    trades: pd.DataFrame
    skipped: int = 0
    metrics: dict[str, float] = field(default_factory=dict)


def simulate_portfolio(trades: pd.DataFrame, rules: RiskRules, costs: Costs,
                       start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> PortfolioResult:
    """Replay trades (columns: coin, t_in, t_out, ret, stop_pct) on one shared equity."""
    t = trades
    if start is not None:
        t = t[t["t_in"] >= start]
    if end is not None:
        t = t[t["t_in"] < end]
    t = t.sort_values("t_in", kind="stable")
    equity = rules.equity0
    open_heap: list[tuple[pd.Timestamp, int, float, float, str]] = []
    open_coins: set[str] = set()
    committed = 0.0
    lock_until: pd.Timestamp | None = None
    streak = 0
    day_pnl: dict[object, float] = {}
    day_start_eq: dict[object, float] = {}
    events: list[tuple[pd.Timestamp, float]] = []
    peak = equity
    has_mult = "risk_mult" in t.columns
    taken: list[tuple[str, pd.Timestamp, pd.Timestamp, float, float, float]] = []
    skipped = 0
    seq = 0

    def close_until(ts: pd.Timestamp) -> None:
        nonlocal equity, committed, streak, lock_until, peak
        while open_heap and open_heap[0][0] <= ts:
            t_out, _, notional, ret, coin = heapq.heappop(open_heap)
            pnl = notional * ret
            equity += pnl
            peak = max(peak, equity)
            committed -= notional
            open_coins.discard(coin)
            events.append((t_out, equity))
            d = t_out.normalize()
            day_pnl[d] = day_pnl.get(d, 0.0) + pnl
            if pnl < 0:
                streak += 1
                if streak >= rules.loss_streak_lock:
                    lock_until = t_out + pd.Timedelta(hours=rules.lock_hours)
                    streak = 0
            else:
                streak = 0

    for row in t.itertuples(index=False):
        close_until(row.t_in)
        d = row.t_in.normalize()
        day_start_eq.setdefault(d, equity)
        if (lock_until is not None and row.t_in < lock_until) or row.coin in open_coins \
                or len(open_heap) >= rules.max_positions \
                or day_pnl.get(d, 0.0) <= -rules.daily_loss_limit * day_start_eq[d]:
            skipped += 1
            continue
        loss_per_unit = row.stop_pct + costs.round_trip
        mult = float(row.risk_mult) if has_mult else 1.0
        if rules.dd_throttle is not None and equity < peak * (1 - rules.dd_throttle):
            mult *= 0.5
        notional = min(equity * rules.risk_per_trade * mult / loss_per_unit, equity - committed)
        if notional < rules.min_notional or equity <= 0:
            skipped += 1
            continue
        seq += 1
        heapq.heappush(open_heap, (row.t_out, seq, notional, row.ret, row.coin))
        open_coins.add(row.coin)
        committed += notional
        taken.append((row.coin, row.t_in, row.t_out, row.ret, notional, notional * row.ret))
    close_until(pd.Timestamp.max.tz_localize("UTC"))
    tr = pd.DataFrame(taken, columns=["coin", "t_in", "t_out", "ret", "notional", "pnl"])
    first = start if start is not None else (t["t_in"].iloc[0] if len(t) else pd.Timestamp.now(tz="UTC"))
    eq = pd.Series([rules.equity0] + [x[1] for x in events],
                   index=pd.DatetimeIndex([first] + [x[0] for x in events])).sort_index()
    res = PortfolioResult(equity=eq, trades=tr, skipped=skipped)
    res.metrics = equity_metrics(eq, tr, rules.equity0, end)
    return res


def daily_curve(eq: pd.Series, end: pd.Timestamp | None = None) -> pd.Series:
    daily = eq.groupby(eq.index.normalize()).last()
    last = (end if end is not None else daily.index[-1]).normalize()
    idx = pd.date_range(daily.index[0], max(last, daily.index[0]), freq="D", tz="UTC")
    return daily.reindex(idx).ffill()


def equity_metrics(eq: pd.Series, tr: pd.DataFrame, equity0: float,
                   end: pd.Timestamp | None = None) -> dict[str, float]:
    daily = daily_curve(eq, end)
    rets = daily.pct_change().dropna()
    months = daily.resample("ME").last()
    monthly = months.pct_change()
    monthly.iloc[0] = months.iloc[0] / equity0 - 1 if len(months) else 0.0
    peak = daily.cummax()
    dd = float(((peak - daily) / peak).max()) if len(daily) else 0.0
    sd = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    downside = float(np.sqrt((np.minimum(rets, 0) ** 2).mean())) if len(rets) else 0.0
    wins = tr[tr["pnl"] > 0]
    losses = tr[tr["pnl"] <= 0]
    gross_w, gross_l = float(wins["pnl"].sum()), float(-losses["pnl"].sum())
    avg_w = float(wins["ret"].mean()) if len(wins) else 0.0
    avg_l = float(-losses["ret"].mean()) if len(losses) else 0.0
    years = max(len(daily) / 365.0, 1e-9)
    final = float(daily.iloc[-1]) if len(daily) else equity0
    return {
        "trades": float(len(tr)),
        "win_rate": float(len(wins) / len(tr)) if len(tr) else 0.0,
        "profit_factor": min(gross_w / gross_l, 99.0) if gross_l > 0 else (99.0 if gross_w > 0 else 0.0),
        "payoff": avg_w / avg_l if avg_l > 0 else 0.0,
        "expectancy_pct": float(tr["ret"].mean()) if len(tr) else 0.0,
        "max_drawdown": dd,
        "sharpe": float(rets.mean()) / sd * math.sqrt(365) if sd > 0 else 0.0,
        "sortino": float(rets.mean()) / downside * math.sqrt(365) if downside > 0 else 0.0,
        "total_return": final / equity0 - 1,
        "cagr": (final / equity0) ** (1 / years) - 1 if final > 0 else -1.0,
        "monthly_mean": float(monthly.mean()) if len(monthly) else 0.0,
        "monthly_median": float(monthly.median()) if len(monthly) else 0.0,
        "monthly_worst": float(monthly.min()) if len(monthly) else 0.0,
        "monthly_best": float(monthly.max()) if len(monthly) else 0.0,
        "positive_months": float((monthly > 0).mean()) if len(monthly) else 0.0,
        "months": float(len(monthly)),
        "daily_sr": float(rets.mean()) / sd if sd > 0 else 0.0,
        "daily_skew": float(rets.skew()) if len(rets) > 2 else 0.0,
        "daily_kurt": float(rets.kurt()) + 3 if len(rets) > 3 else 3.0,
        "days": float(len(rets)),
    }
