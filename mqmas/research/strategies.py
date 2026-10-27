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


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder's ADX: trend *strength* regardless of direction (>20–25 = trending)."""
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus = up.where((up > dn) & (up > 0), 0.0)
    minus = dn.where((dn > up) & (dn > 0), 0.0)
    a = atr(df, n)
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


REGIMES = ("alta", "queda", "lateral", "volatil")


def daily_regime(daily: pd.DataFrame) -> pd.Series:
    """Market phase of one coin, known at the open of each day (computed on D-1's close).

    volatil  ATR% above its own 1-year 90th percentile (crash / blow-off conditions)
    alta     close > EMA200, EMA50 > EMA200 and ADX ≥ 20
    queda    close < EMA200 and EMA50 < EMA200
    lateral  everything else (no confirmed trend, or a weak one: ADX < 20)
    """
    c = daily["close"]
    e50, e200 = ema(c, 50), ema(c, 200)
    strength = adx(daily)
    atr_pct = atr(daily) / c
    hot = atr_pct > atr_pct.rolling(365, min_periods=120).quantile(0.9)
    lab = pd.Series("lateral", index=daily.index)
    lab[(c < e200) & (e50 < e200)] = "queda"
    lab[(c > e200) & (e50 > e200) & (strength >= 20)] = "alta"
    lab[hot] = "volatil"
    lab[e200.isna()] = "lateral"
    return lab.shift(1).fillna("lateral")


def features(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()
    c = f["close"]
    f["ema20"], f["ema50"], f["ema200"] = ema(c, 20), ema(c, 50), ema(c, 200)
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
    tp = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.normalize()
    f["vwap"] = (tp * df["volume"]).groupby(day).cumsum() / df["volume"].groupby(day).cumsum().replace(0, np.nan)
    for q in (0.1, 0.2):
        f[f"bbwq{q}"] = f["bbw"].rolling(100).quantile(q)
    f["adx14"] = adx(df)
    f["atr_pct"] = f["atr"] / c
    # last 55-bar leg for Fibonacci: is the high more recent than the low (an up leg)?
    hi55 = f["high"].shift(1).rolling(55)
    lo55 = f["low"].shift(1).rolling(55)
    f["hh55"], f["ll55"] = hi55.max(), lo55.min()
    f["up_leg"] = hi55.apply(np.argmax, raw=True) > lo55.apply(np.argmin, raw=True)
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


def _confluence(f: pd.DataFrame, p: dict[str, Any], regime: np.ndarray) -> np.ndarray:
    """Layered confluence: everything mandatory must hold, plus a trigger, plus enough extras.

    mandatory  BTC regime (prev-day close > SMA100) · close > EMA200 · EMA50 > EMA200
    trigger    breakout of the 20-bar high | bounce off the EMA20 | either
    extras     relative strength vs the universe ≥ rs_min · volume > 1.3× average ·
               RSI(14) in the 50–70 momentum band (strong but not exhausted)
    """
    c = f["close"]
    mandatory = (c > f["ema200"]) & (f["ema50"] > f["ema200"])
    breakout = c > f["hh20"]
    bounce = (f["low"] <= f["ema20"]) & (c > f["ema20"]) & (c > f["open"])
    trigger = {"breakout": breakout, "pullback": bounce, "either": breakout | bounce}[p["trigger"]]
    extras = ((f["rs"] >= p["rs_min"]).astype(int) + (f["volume"] > 1.3 * f["vol20"]).astype(int)
              + f["rsi14"].between(50, 70).astype(int))
    out: np.ndarray = (mandatory & trigger & (extras >= p["min_extra"])).to_numpy(dtype=bool) & regime
    return out


def _mtf(f: pd.DataFrame, p: dict[str, Any], regime: np.ndarray) -> np.ndarray:
    """Multi-timeframe confluence: the *daily* chart says what to buy, the intraday chart says when.

    mandatory  BTC regime · coin's daily trend (prev-day close > EMA200 and EMA50 > EMA200) ·
               intraday close > intraday EMA200
    trigger    pullback bounce off the intraday EMA50 | RSI(3) dip | 20-bar breakout | VWAP reclaim
    extras     relative strength ≥ rs_min · volume > 1.3× average · RSI(14) in 45–70
    """
    c = f["close"]
    mandatory = f["d_trend"].astype(bool) & (c > f["ema200"])
    trig = {
        "pullback": (f["low"] <= f["ema50"]) & (c > f["ema50"]) & (c > f["open"]),
        "rsi_dip": f["rsi3"] < 15,
        "breakout": c > f["hh20"],
        "vwap": (c.shift(1) < f["vwap"].shift(1)) & (c > f["vwap"]),
    }[p["trigger"]]
    extras = ((f["rs"] >= p["rs_min"]).astype(int) + (f["volume"] > 1.3 * f["vol20"]).astype(int)
              + f["rsi14"].between(45, 70).astype(int))
    out: np.ndarray = (mandatory & trig & (extras >= p["min_extra"])).to_numpy(dtype=bool) & regime
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


# ---------------------------------------------------------------------- classic swing (no optimised parameters)
# Every number below is the textbook value, fixed before any test: EMA 20/50/200, ADX 20, Donchian 20,
# volume 1.5×, Bollinger 20/2, Fibonacci 50–61.8 % with 78.6 % as invalidation.
CLASSIC_SETUPS = ("tendencia", "rompimento", "bollinger", "fibonacci")
CLASSIC_ATR_BAND = (0.015, 0.08)  # daily ATR% outside this band = market dead or too wild → stay out


def classic_macro(f: pd.DataFrame, regime: np.ndarray) -> np.ndarray:
    """The gate before any setup: BTC above its SMA100 (regime), coin above EMA200 with EMA50 > EMA200,
    a trend that exists (ADX ≥ 20) and volatility in an acceptable band."""
    c = f["close"]
    ok = (c > f["ema200"]) & (f["ema50"] > f["ema200"]) & (f["adx14"] >= 20) & \
        f["atr_pct"].between(*CLASSIC_ATR_BAND)
    out: np.ndarray = ok.to_numpy(dtype=bool) & regime
    return out


def classic_triggers(f: pd.DataFrame) -> dict[str, pd.Series]:
    c, o, lo = f["close"], f["open"], f["low"]
    green = c > o
    leg = f["hh55"] - f["ll55"]
    fib50, fib618, fib786 = f["hh55"] - 0.5 * leg, f["hh55"] - 0.618 * leg, f["hh55"] - 0.786 * leg
    return {
        # moving averages: pullback to the EMA20 in an ordered trend, bullish close back above it
        "tendencia": (f["ema20"] > f["ema50"]) & (lo <= f["ema20"]) & (c > f["ema20"]) & green,
        # top/resistance: close above the prior 20-day high with volume 1.5× the 20-day average
        "rompimento": (c > f["hh20"]) & (f["volume"] > 1.5 * f["vol20"]),
        # Bollinger: close back inside after closing below the lower band (dip inside an uptrend)
        "bollinger": (c.shift(1) < f["bblo2.0"].shift(1)) & (c > f["bblo2.0"]) & green,
        # Fibonacci: an up leg retraced into the 50–61.8 % pocket, held 78.6 %, bullish close above 61.8 %
        "fibonacci": f["up_leg"].astype(bool) & (lo <= fib50) & (lo >= fib786) & (c > fib618) & green,
    }


def _classic(f: pd.DataFrame, p: dict[str, Any], regime: np.ndarray) -> np.ndarray:
    trig = classic_triggers(f)
    names = CLASSIC_SETUPS if p["setup"] == "todos" else (p["setup"],)
    fired = np.zeros(len(f), dtype=bool)
    for n in names:
        fired |= trig[n].fillna(False).to_numpy(dtype=bool)
    if p.get("macro", True):
        fired &= classic_macro(f, regime)
    return fired


CLASSIC_EXIT = Exit(stop_atr=2.0, target_atr=6.0, trail_atr=None, hold=20)  # stop 2 ATR, target 3R, 20 days


def classic_configs() -> list[Config]:
    out = [Config("classic", "1d", (("setup", s), ("macro", True)), CLASSIC_EXIT) for s in (*CLASSIC_SETUPS, "todos")]
    out.append(Config("classic", "1d", (("setup", "todos"), ("macro", False)), CLASSIC_EXIT))
    return out


FAMILIES: dict[str, SignalFn] = {
    "breakout": _breakout, "pullback": _pullback, "meanrev": _meanrev, "squeeze": _squeeze,
    "confluence": _confluence, "mtf": _mtf, "classic": _classic,
}
HOLD = {"1h": 120, "4h": 60, "1d": 40}  # 5, 10 and 40 days


def _grid(family: str, tf: str, params: dict[str, list[Any]], exits: list[Exit]) -> list[Config]:
    keys = list(params)
    return [Config(family, tf, tuple(zip(keys, combo, strict=True)), ex)
            for combo in itertools.product(*params.values()) for ex in exits]


def build_configs(timeframes: tuple[str, ...] = ("1h", "4h", "1d")) -> list[Config]:
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
    for tf in ("4h", "1d"):
        h = HOLD[tf]
        conf_exits = [Exit(2.0, None, 3.0, h), Exit(3.0, None, 4.5, h), Exit(2.0, 3.0, None, h // 2),
                      Exit(2.5, 4.5, None, h // 2), Exit(2.0, 6.0, 3.0, h)]
        out += _grid("confluence", tf, {"trigger": ["breakout", "pullback", "either"], "min_extra": [2, 3],
                                        "rs_min": [0.6, 0.8]}, conf_exits)
    for tf in ("1h", "4h"):
        h = HOLD[tf]
        mtf_exits = [Exit(2.0, None, 3.0, h), Exit(3.0, None, 4.5, h), Exit(2.0, 3.0, None, h // 2),
                     Exit(2.5, 4.5, None, h // 2), Exit(2.0, 6.0, 3.0, h)]
        out += _grid("mtf", tf, {"trigger": ["pullback", "rsi_dip", "breakout", "vwap"], "min_extra": [1, 2],
                                 "rs_min": [0.6, 0.8]}, mtf_exits)
    return out


# --------------------------------------------------------------------------- universe & regime


def btc_regime(btc_daily: pd.DataFrame, sma: int = 100) -> pd.Series:
    """True on day D when BTC closed above its SMA on day D-1 (known before D opens)."""
    c = btc_daily["close"]
    ok = (c > c.rolling(sma).mean()).shift(1)
    return ok.fillna(False).astype(bool)


def daily_trend(daily: pd.DataFrame) -> pd.Series:
    """True on day D when, at D-1's close, price > EMA200 and EMA50 > EMA200 (daily chart)."""
    c = daily["close"]
    e50 = c.ewm(span=50, adjust=False, min_periods=50).mean()
    e200 = c.ewm(span=200, adjust=False, min_periods=200).mean()
    return ((c > e200) & (e50 > e200)).shift(1).fillna(False).astype(bool)


def relative_strength(daily: dict[str, pd.DataFrame], eligible: dict[pd.Period, set[str]],
                      lookback: int = 30) -> pd.DataFrame:
    """Percentile rank (0–1) of each coin's trailing return inside that month's universe, known at the
    open of the day (shifted one day)."""
    closes = pd.DataFrame({c: d["close"] for c, d in daily.items()}).sort_index()
    mom = closes / closes.shift(lookback) - 1
    months = closes.index.tz_localize(None).to_period("M")
    mask = pd.DataFrame({c: [c in eligible.get(m, ()) for m in months] for c in closes.columns},
                        index=closes.index)
    return mom.where(mask).rank(axis=1, pct=True).shift(1)


def config_by_key() -> dict[str, Config]:
    return {c.key: c for c in [*build_configs(), *classic_configs()]}


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
