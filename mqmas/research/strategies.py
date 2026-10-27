"""Strategy families and their parameter grids (long-only spot).

Every signal uses only the closed bar and earlier data. The BTC regime filter
uses the *previous* day's BTC close vs its SMA, so it is known before the
intraday bar opens.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mqmas.research.engine import Exit

WARMUP = 220


def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def features(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()
    c = f["close"]
    f["ema50"], f["ema200"] = ema(c, 50), ema(c, 200)
    f["rsi3"], f["rsi14"] = rsi(c, 3), rsi(c, 14)
    f["atr"] = atr(df)
    f["vol20"] = f["volume"].shift(1).rolling(20).mean()
    for n in (20, 55):
        f[f"hh{n}"] = f["high"].shift(1).rolling(n).max()
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    f["sma20"] = mid
    f["bbw"] = 4 * sd / mid
    for k in (2.0, 2.5):
        f[f"bblo{k}"] = mid - k * sd
    f["bbup2.0"] = mid + 2 * sd
    for q in (0.1, 0.2):
        f[f"bbwq{q}"] = f["bbw"].rolling(100).quantile(q)
    return f


SignalFn = Callable[[pd.DataFrame, dict[str, Any], np.ndarray], np.ndarray]


def _breakout(f: pd.DataFrame, p: dict[str, Any], regime: np.ndarray) -> np.ndarray:
    s = (f["close"] > f[f"hh{p['n']}"]) & (f["close"] > f["ema200"])
    if p["vol"]:
        s &= f["volume"] > 1.5 * f["vol20"]
    out: np.ndarray = s.to_numpy(dtype=bool) & regime
    return out


def _pullback(f: pd.DataFrame, p: dict[str, Any], regime: np.ndarray) -> np.ndarray:
    s = (f["ema50"] > f["ema200"]) & (f["close"] > f["ema200"]) & (f["rsi3"] < p["rsi"])
    out: np.ndarray = s.to_numpy(dtype=bool) & regime
    return out


def _meanrev(f: pd.DataFrame, p: dict[str, Any], regime: np.ndarray) -> np.ndarray:
    s = (f["close"] < f[f"bblo{p['k']}"]) & (f["rsi14"] < 30)
    out: np.ndarray = s.to_numpy(dtype=bool)
    return out & regime if p["regime"] else out


def _squeeze(f: pd.DataFrame, p: dict[str, Any], regime: np.ndarray) -> np.ndarray:
    s = (f["bbw"].shift(1) <= f[f"bbwq{p['q']}"].shift(1)) & (f["close"] > f["bbup2.0"]) & (f["close"] > f["ema200"])
    out: np.ndarray = s.to_numpy(dtype=bool) & regime
    return out


@dataclass(frozen=True)
class Config:
    family: str
    tf: str
    params: tuple[tuple[str, Any], ...]
    exit: Exit

    @property
    def key(self) -> str:
        p = ",".join(f"{k}={v}" for k, v in self.params)
        e = self.exit
        return (f"{self.family}|{self.tf}|{p}|stop={e.stop_atr}|tp={e.target_atr}|trail={e.trail_atr}"
                f"|hold={e.hold}")

    @property
    def p(self) -> dict[str, Any]:
        return dict(self.params)


FAMILIES: dict[str, SignalFn] = {
    "breakout": _breakout, "pullback": _pullback, "meanrev": _meanrev, "squeeze": _squeeze,
}
HOLD = {"1h": 120, "4h": 60}  # 5 and 10 days


def _grid(family: str, tf: str, params: dict[str, list[Any]], exits: list[Exit]) -> list[Config]:
    keys = list(params)
    return [Config(family, tf, tuple(zip(keys, combo, strict=True)), ex)
            for combo in itertools.product(*params.values()) for ex in exits]


def build_configs(timeframes: tuple[str, ...] = ("1h", "4h")) -> list[Config]:
    out: list[Config] = []
    for tf in timeframes:
        h = HOLD[tf]
        trend_exits = [Exit(s, None, tr, h) for s in (2.0, 3.0) for tr in (3.0, 4.5)]
        swing_exits = [Exit(s, t, None, h // 2) for s in (1.5, 2.5) for t in (2.0, 3.0, 4.5)]
        out += _grid("breakout", tf, {"n": [20, 55], "vol": [False, True]}, trend_exits)
        out += _grid("pullback", tf, {"rsi": [10, 20, 30]}, swing_exits)
        out += _grid("meanrev", tf, {"k": [2.0, 2.5], "regime": [False, True]},
                     [Exit(s, t, None, h // 2) for s in (1.5, 2.5) for t in (1.5, 2.5)])
        out += _grid("squeeze", tf, {"q": [0.1, 0.2]}, trend_exits + [Exit(2.0, 3.0, None, h // 2)])
    return out


# --------------------------------------------------------------------------- universe & regime


def btc_regime(btc_daily: pd.DataFrame, sma: int = 100) -> pd.Series:
    """True on day D when BTC closed above its SMA on day D-1 (known before D opens)."""
    c = btc_daily["close"]
    ok = (c > c.rolling(sma).mean()).shift(1)
    return ok.fillna(False).astype(bool)


def align_daily(flag: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    out: np.ndarray = flag.reindex(index.normalize()).fillna(False).to_numpy(dtype=bool)
    return out


def monthly_eligibility(daily: dict[str, pd.DataFrame], top_n: int, min_range: float,
                        max_range: float) -> dict[pd.Period, set[str]]:
    """Point-in-time universe: at each month start, top-N by trailing 30d dollar volume
    among coins whose trailing 30d average daily range sits inside [min_range, max_range]."""
    stats: dict[str, pd.DataFrame] = {}
    for coin, d in daily.items():
        dv = (d["close"] * d["volume"]).rolling(30).median()
        rng = ((d["high"] - d["low"]) / d["close"]).rolling(30).mean()
        age = pd.Series(np.arange(len(d)), index=d.index)
        stats[coin] = pd.DataFrame({"dv": dv, "rng": rng, "age": age}).shift(1)
    months = sorted({ts.tz_localize(None).to_period("M") for d in daily.values() for ts in d.index})
    out: dict[pd.Period, set[str]] = {}
    for m in months:
        day = m.start_time.tz_localize("UTC")
        rows = []
        for coin, st in stats.items():
            if day not in st.index:
                continue
            r = st.loc[day]
            if r["age"] >= 60 and min_range <= r["rng"] <= max_range and np.isfinite(r["dv"]):
                rows.append((float(r["dv"]), coin))
        rows.sort(reverse=True)
        out[m] = {c for _, c in rows[:top_n]}
    return out
