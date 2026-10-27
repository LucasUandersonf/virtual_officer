"""Team setup templates, translated literally from the "top Binance setups" playbook and adapted to
the mandate (spot, long-only, no leverage). Each template is an exact rule that the walk-forward lab
can test; nothing here is assumed to work until the out-of-sample numbers say so.

Templates need three things the plain bracket simulator lacks, provided by :func:`structured_trades`:

* **structural stops** (channel midpoint, swing low, broken level − ATR) instead of ATR multiples;
* **limit entries** that wait for a re-test (support-flip) and may never fill;
* **indicator exits** (e.g. daily close below the EMA200), checked at the bar close.

Execution conventions match :mod:`mqmas.research.engine`: market entries at the next open, stop
before target inside a bar, gaps fill at the open, costs on both sides, one position per coin.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from mqmas.research.engine import REASON_STOP, REASON_TARGET, REASON_TIME, Costs

REASON_SIGNAL = 3


@dataclass
class Plan:
    """What a template wants to do at each signal bar (arrays aligned with ``sig_idx``)."""

    sig_idx: np.ndarray
    stop: np.ndarray  # absolute stop price; NaN → stop_atr × ATR below the fill
    target_r: float | None = None  # take profit at R multiples of the initial risk
    stop_atr: float = 2.0
    limit: np.ndarray | None = None  # absolute limit price; None → market at next open
    wait: int = 0  # bars a limit order stays alive
    trail_atr: float | None = None
    exit_sig: np.ndarray | None = None  # bool per bar: exit at that bar's close
    hold: int = 40


def structured_trades(o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray, atr: np.ndarray,
                      plan: Plan, costs: Costs) -> dict[str, np.ndarray]:
    n, hold = o.size, plan.hold
    empty = {k: np.array([]) for k in ("entry", "exit", "ret", "stop_pct", "reason")}
    i = plan.sig_idx.astype(int)
    keep = (i + 1 + plan.wait + hold < n) & np.isfinite(atr[i]) & (atr[i] > 0)
    i = i[keep]
    if i.size == 0:
        return empty
    stop_in = plan.stop[keep]
    a = atr[i]
    if plan.limit is None:
        e = i + 1
        entry = o[e]
    else:
        lim = plan.limit[keep]
        wl = sliding_window_view(lo, plan.wait)[i + 1]
        hit = wl <= lim[:, None]
        filled = hit.any(axis=1)
        j = hit.argmax(axis=1)
        i, a, stop_in, lim, j = i[filled], a[filled], stop_in[filled], lim[filled], j[filled]
        if i.size == 0:
            return empty
        e = i + 1 + j
        entry = np.minimum(o[e], lim)  # gap below the limit fills at the open
    stop0 = np.where(np.isfinite(stop_in), stop_in, entry - plan.stop_atr * a)
    ok = (stop0 > 0) & (stop0 < entry)
    e, a, entry, stop0 = e[ok], a[ok], entry[ok], stop0[ok]
    if e.size == 0:
        return empty
    risk = entry - stop0
    wo = sliding_window_view(o, hold)[e]
    wh = sliding_window_view(h, hold)[e]
    wl = sliding_window_view(lo, hold)[e]
    wc = sliding_window_view(c, hold)[e]
    stop = np.repeat(stop0[:, None], hold, axis=1)
    if plan.trail_atr is not None:
        prev_max = np.concatenate([entry[:, None], np.maximum.accumulate(wh, axis=1)[:, :-1]], axis=1)
        stop = np.maximum(stop, prev_max - plan.trail_atr * a[:, None])
    hit_s = wl <= stop
    first_s = np.where(hit_s.any(axis=1), hit_s.argmax(axis=1), hold)
    if plan.target_r is not None:
        tgt = entry + plan.target_r * risk
        hit_t = wh >= tgt[:, None]
        first_t = np.where(hit_t.any(axis=1), hit_t.argmax(axis=1), hold)
    else:
        tgt = np.full(e.size, np.inf)
        first_t = np.full(e.size, hold)
    if plan.exit_sig is not None:
        ws = sliding_window_view(plan.exit_sig, hold)[e]
        first_x = np.where(ws.any(axis=1), ws.argmax(axis=1), hold)
    else:
        first_x = np.full(e.size, hold)
    stopped = (first_s < hold) & (first_s <= first_t) & (first_s <= first_x)
    targeted = ~stopped & (first_t < hold) & (first_t <= first_x)
    signalled = ~stopped & ~targeted & (first_x < hold)
    rows = np.arange(e.size)
    js, jt, jx = (np.minimum(x, hold - 1) for x in (first_s, first_t, first_x))
    px = np.where(stopped, np.minimum(wo[rows, js], stop[rows, js]),
                  np.where(targeted, np.maximum(wo[rows, jt], tgt),
                           np.where(signalled, wc[rows, jx], c[e + hold - 1])))
    exit_bar = e + np.where(stopped, first_s, np.where(targeted, first_t, np.where(signalled, first_x, hold - 1)))
    ret = px / entry - 1 - costs.round_trip
    reason = np.where(stopped, REASON_STOP, np.where(targeted, REASON_TARGET,
                                                     np.where(signalled, REASON_SIGNAL, REASON_TIME)))
    sel: list[int] = []
    busy = -1
    for k in np.argsort(e, kind="stable"):
        if e[k] > busy:
            sel.append(int(k))
            busy = int(exit_bar[k])
    s = np.asarray(sel, dtype=int)
    return {"entry": e[s], "exit": exit_bar[s], "ret": ret[s], "stop_pct": (risk / entry)[s], "reason": reason[s]}


# --------------------------------------------------------------------------- templates

PlanFn = Callable[[pd.DataFrame, dict[str, Any], np.ndarray], Plan]
HOLD = {"1h": 72, "4h": 90, "1d": 60}


def _idx(mask: pd.Series | np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(mask, dtype=bool))


def t1_range_breakout(f: pd.DataFrame, p: dict[str, Any], regime: np.ndarray) -> Plan:
    """#1 High-volume 20-bar range breakout, stop at the channel midpoint."""
    hh, ll = f["hh20"], f["low"].shift(1).rolling(20).min()
    width = (hh - ll) / f["close"]
    sig = (f["close"] > hh) & (f["volume"] > p["vol"] * f["vol20"]) & (width <= p["max_width"])
    if p["trend"]:
        sig &= f["close"] > f["ema200"]
    i = _idx(sig.to_numpy(dtype=bool) & regime)
    mid = ((hh + ll) / 2).to_numpy()
    return Plan(i, mid[i], target_r=p["r"], trail_atr=p["trail"], hold=HOLD[p["tf"]])


def t2_golden_cross(f: pd.DataFrame, p: dict[str, Any], regime: np.ndarray) -> Plan:
    """#2 EMA50/200 golden cross; optional entries only on EMA50 pullbacks; exit on close < EMA200."""
    up = f["ema50"] > f["ema200"]
    if p["entry"] == "cross":
        sig = up & ~up.shift(1, fill_value=False)
    else:
        sig = up & (f["low"] <= f["ema50"]) & (f["close"] > f["ema50"])
    i = _idx(sig.to_numpy(dtype=bool) & regime)
    exit_sig = (f["close"] < f["ema200"]).to_numpy(dtype=bool)
    return Plan(i, np.full(i.size, np.nan), stop_atr=p["stop_atr"], exit_sig=exit_sig, hold=p["hold"])


def t3_rsi_divergence(f: pd.DataFrame, p: dict[str, Any], regime: np.ndarray) -> Plan:
    """#3 (long side only — spot): bullish RSI divergence + bullish engulfing, stop under the swing low."""
    ll20 = f["low"].shift(1).rolling(20).min()
    rsi_min = f["rsi14"].shift(1).rolling(20).min()
    swing_low = f["low"].rolling(5).min()
    div = (swing_low <= ll20) & (f["rsi14"] > rsi_min + 3) & (rsi_min < 35)
    engulf = (f["close"] > f["open"]) & (f["close"] > f["open"].shift(1)) & (f["open"] <= f["close"].shift(1)) \
        & (f["close"].shift(1) < f["open"].shift(1))
    sig = div & engulf
    if p["trend"]:
        sig &= f["d_trend"].astype(bool)
    i = _idx(sig.to_numpy(dtype=bool) & (regime if p["trend"] else True))
    stop = (swing_low - 0.2 * f["atr"]).to_numpy()
    return Plan(i, stop[i], target_r=p["r"], hold=HOLD[p["tf"]])


def t7_support_flip(f: pd.DataFrame, p: dict[str, Any], regime: np.ndarray) -> Plan:
    """#7 Breakout with volume, then a limit buy at the broken resistance (re-test); stop level − ATR."""
    level = f["high"].shift(1).rolling(p["n"]).max()
    sig = (f["close"] > level) & (f["volume"] > 1.5 * f["vol20"]) & (f["close"] > f["ema200"])
    i = _idx(sig.to_numpy(dtype=bool) & regime)
    lv = level.to_numpy()
    a = f["atr"].to_numpy()
    return Plan(i, (lv - p["stop_atr"] * a)[i], target_r=p["r"], limit=lv[i] * 1.002, wait=p["wait"],
                hold=HOLD[p["tf"]])


@dataclass(frozen=True)
class Template:
    name: str
    title: str
    playbook: int
    fn: PlanFn
    tfs: tuple[str, ...]
    grid: dict[str, list[Any]]

    def configs(self) -> list[tuple[str, str, dict[str, Any]]]:
        out = []
        for tf in self.tfs:
            for combo in itertools.product(*self.grid.values()):
                p = dict(zip(self.grid, combo, strict=True)) | {"tf": tf}
                key = f"tpl:{self.name}|{tf}|" + ",".join(f"{k}={v}" for k, v in p.items() if k != "tf")
                out.append((key, tf, p))
        return out


TEMPLATES: tuple[Template, ...] = (
    Template("range_breakout", "Rompimento de range 20 barras com volume (stop no meio do canal)", 1,
             t1_range_breakout, ("4h", "1d"),
             {"vol": [1.5, 2.0], "max_width": [0.15, 0.30], "trend": [False, True], "r": [2.0, None],
              "trail": [None, 3.0]}),
    Template("golden_cross", "Golden Cross EMA50/200 (saída no fechamento abaixo da EMA200)", 2,
             t2_golden_cross, ("4h", "1d"),
             {"entry": ["cross", "pullback"], "stop_atr": [3.0, 5.0], "hold": [60, 120]}),
    Template("rsi_divergence", "Divergência de RSI + engolfo de alta (lado comprado)", 3,
             t3_rsi_divergence, ("1h", "4h"), {"trend": [False, True], "r": [1.5, 2.5]}),
    Template("support_flip", "Reteste de resistência rompida (limit no nível)", 7,
             t7_support_flip, ("4h", "1d"), {"n": [20, 55], "wait": [5, 10], "stop_atr": [1.0, 1.5], "r": [2.0, 3.0]}),
)


# --------------------------------------------------------------------------- non-bracket templates


def grid_returns(df: pd.DataFrame, levels: int, stop_below: float, costs: Costs) -> pd.Series:
    """#6 Spot grid bot, re-armed every month on the previous 30 days' range.

    Capital is split into ``levels`` equal slots. A slot buys when price falls through its level and
    sells one level higher. Inventory is marked to market; if price closes ``stop_below`` under the
    grid floor everything is sold and the grid stays off until next month. Bars are walked
    open → extreme → extreme → close in the direction of the candle (the usual OHLC path assumption).
    Returns hourly portfolio returns.
    """
    o, h, lo, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    months = df.index.tz_localize(None).to_period("M")
    cash, last_val = 1.0, 1.0
    slots: dict[int, float] = {}
    rets = np.zeros(len(df))
    grid: np.ndarray | None = None
    halted = False
    cur_month = None
    fee = costs.fee + costs.slippage
    for t in range(len(df)):
        m = months[t]
        if m != cur_month:
            cur_month = m
            for lv, qty in list(slots.items()):  # flatten at the roll
                cash += qty * o[t] * (1 - fee)
                del slots[lv]
            past = df.iloc[max(0, t - 720):t]
            halted = len(past) < 240
            if not halted:
                grid = np.geomspace(past["low"].min(), past["high"].max(), levels + 1)
        if grid is not None and not halted:
            path = (o[t], lo[t], h[t], c[t]) if c[t] >= o[t] else (o[t], h[t], lo[t], c[t])
            slot_cash = 1.0 / levels
            for a, b in zip(path[:-1], path[1:], strict=True):
                for k in range(levels):
                    buy_px, sell_px = grid[k], grid[k + 1]
                    if k not in slots and a > buy_px >= b and cash >= slot_cash * 0.999:
                        slots[k] = slot_cash * (1 - fee) / buy_px
                        cash -= slot_cash
                    elif k in slots and a < sell_px <= b:
                        cash += slots.pop(k) * sell_px * (1 - fee)
            if c[t] < grid[0] * (1 - stop_below):
                for k, qty in list(slots.items()):
                    cash += qty * c[t] * (1 - fee)
                    del slots[k]
                halted = True
        value = cash + sum(q * c[t] for q in slots.values())
        rets[t] = value / last_val - 1
        last_val = value
    return pd.Series(rets, index=df.index)


def dca_benchmark(daily: dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp,
                  weights: dict[str, float]) -> dict[str, float]:
    """#10 Weekly DCA (every Monday 00:00 UTC) — a benchmark, not a trading setup."""
    closes = pd.DataFrame({s: daily[s]["close"] for s in weights}).loc[start:end].dropna()
    mondays = closes.index[closes.index.dayofweek == 0]
    units = pd.Series(0.0, index=closes.columns)
    invested = 0.0
    value_path = []
    for d in closes.index:
        if d in mondays:
            for s, w in weights.items():
                units[s] += w * (1 - 0.0015) / closes.at[d, s]
            invested += 1.0
        value_path.append((float((units * closes.loc[d]).sum()), invested))
    v = pd.DataFrame(value_path, index=closes.index, columns=["value", "invested"])
    pnl_ratio = v["value"] / v["invested"].replace(0, np.nan)
    peak = pnl_ratio.cummax()
    return {"weeks": float(len(mondays)), "total_return_on_invested": float(pnl_ratio.iloc[-1] - 1),
            "worst_drawdown_vs_cost": float(((peak - pnl_ratio) / peak).max())}
