"""Classic indicators (trend, momentum, volatility, volume), pure pandas, no look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def wma(x: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return x.rolling(n, min_periods=n).apply(lambda a: float(np.dot(a, w) / w.sum()), raw=True)


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def macd(c: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(c, fast) - ema(c, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus = up.where((up > dn) & (up > 0), 0.0)
    minus = dn.where((dn > up) & (dn > 0), 0.0)
    a = atr(df, n)
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return pd.DataFrame({"adx": dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean(), "plus_di": pdi,
                         "minus_di": mdi})


def parabolic_sar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    h, lo = df["high"].to_numpy(), df["low"].to_numpy()
    n = len(df)
    out = np.full(n, np.nan)
    if n < 3:
        return pd.Series(out, index=df.index)
    up = True
    af = step
    ep = h[0]
    sar = lo[0]
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, lo[i - 1], lo[i - 2] if i > 1 else lo[i - 1])
            if lo[i] < sar:
                up, sar, ep, af = False, ep, lo[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, max_step)
        else:
            sar = max(sar, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > sar:
                up, sar, ep, af = True, ep, h[i], step
            elif lo[i] < ep:
                ep, af = lo[i], min(af + step, max_step)
        out[i] = sar
    return pd.Series(out, index=df.index)


def ichimoku(df: pd.DataFrame, t: int = 9, k: int = 26, s: int = 52) -> pd.DataFrame:
    """Tenkan, Kijun and the cloud *as it stands at each bar* (spans are already displaced forward,
    so the values aligned with bar i were computed from data up to i−26: no look-ahead)."""
    def mid(n: int) -> pd.Series:
        return (df["high"].rolling(n).max() + df["low"].rolling(n).min()) / 2

    tenkan, kijun = mid(t), mid(k)
    span_a = ((tenkan + kijun) / 2).shift(k)
    span_b = mid(s).shift(k)
    return pd.DataFrame({"tenkan": tenkan, "kijun": kijun, "span_a": span_a, "span_b": span_b})


def stochastic(df: pd.DataFrame, n: int = 14, d: int = 3) -> pd.DataFrame:
    lo, hi = df["low"].rolling(n).min(), df["high"].rolling(n).max()
    k = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    return pd.DataFrame({"stoch_k": k, "stoch_d": k.rolling(d).mean()})


def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    m = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda a: float(np.mean(np.abs(a - a.mean()))), raw=True)
    return (tp - m) / (0.015 * md.replace(0, np.nan))


def roc(c: pd.Series, n: int = 12) -> pd.Series:
    return 100 * (c / c.shift(n) - 1)


def bollinger(c: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(c, n)
    sd = c.rolling(n).std(ddof=0)
    return pd.DataFrame({"bb_mid": mid, "bb_up": mid + k * sd, "bb_lo": mid - k * sd,
                         "bb_width": 2 * k * sd / mid})


def keltner(df: pd.DataFrame, n: int = 20, mult: float = 1.5) -> pd.DataFrame:
    mid = ema(df["close"], n)
    a = atr(df, n)
    return pd.DataFrame({"kc_mid": mid, "kc_up": mid + mult * a, "kc_lo": mid - mult * a})


def ttm_squeeze(df: pd.DataFrame) -> pd.DataFrame:
    """Squeeze ON while Bollinger(20, 2) sits inside Keltner(20, 1.5); 'fired' on the bar it releases."""
    bb, kc = bollinger(df["close"]), keltner(df)
    on = (bb["bb_up"] < kc["kc_up"]) & (bb["bb_lo"] > kc["kc_lo"])
    mom = df["close"] - (df["high"].rolling(20).max() + df["low"].rolling(20).min() + sma(df["close"], 20) * 2) / 4
    return pd.DataFrame({"squeeze_on": on, "squeeze_fired": on.shift(1, fill_value=False) & ~on, "squeeze_mom": mom})


def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff()).fillna(0)
    return (sign * df["volume"]).cumsum()


def session_vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.normalize()
    return (tp * df["volume"]).groupby(day).cumsum() / df["volume"].groupby(day).cumsum().replace(0, np.nan)


def volume_profile(df: pd.DataFrame, bins: int = 40, area: float = 0.7) -> dict[str, float]:
    lo, hi = float(df["low"].min()), float(df["high"].max())
    if hi <= lo:
        return {"poc": lo, "vah": hi, "val": lo}
    edges = np.linspace(lo, hi, bins + 1)
    hist = np.zeros(bins)
    for lo_, hi_, v in zip(df["low"].to_numpy(), df["high"].to_numpy(), df["volume"].to_numpy(), strict=True):
        i0 = max(int(np.searchsorted(edges, lo_, side="right")) - 1, 0)
        i1 = max(min(int(np.searchsorted(edges, hi_, side="left")), bins), i0 + 1)
        hist[i0:i1] += v / (i1 - i0)
    poc = int(np.argmax(hist))
    left = right = poc
    acc, total = hist[poc], hist.sum()
    while acc < area * total and (left > 0 or right < bins - 1):
        lv = hist[left - 1] if left > 0 else -1
        rv = hist[right + 1] if right < bins - 1 else -1
        if rv >= lv:
            right += 1
            acc += hist[right]
        else:
            left -= 1
            acc += hist[left]
    centers = (edges[:-1] + edges[1:]) / 2
    return {"poc": float(centers[poc]), "vah": float(edges[right + 1]), "val": float(edges[left])}


def all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    f = pd.DataFrame(index=df.index)
    for n in (20, 50, 200):
        f[f"ema{n}"] = ema(c, n)
        f[f"sma{n}"] = sma(c, n)
    f["atr"] = atr(df)
    f["rsi"] = rsi(c)
    f = f.join(macd(c)).join(adx(df)).join(ichimoku(df)).join(stochastic(df)).join(bollinger(c)).join(keltner(df))
    f = f.join(ttm_squeeze(df))
    f["sar"] = parabolic_sar(df)
    f["cci"] = cci(df)
    f["roc"] = roc(c)
    f["obv"] = obv(df)
    f["obv_ema"] = ema(f["obv"], 20)
    f["vwap"] = session_vwap(df)
    return f
