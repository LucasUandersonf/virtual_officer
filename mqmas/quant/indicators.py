"""Vectorised price/volume statistics on OHLCV frames.

All functions take a pandas DataFrame indexed by UTC timestamp with columns
``open, high, low, close, volume`` and are pure (no I/O, no globals), which
makes them trivially unit-testable and safe to call from worker threads.

pandas-ta is intentionally not a hard dependency: its releases have lagged
NumPy 2.x and the few indicators needed here are a handful of lines each, so
they are implemented natively (and validated in ``tests/test_quant.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mqmas.models import VolumeProfile

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def validate_ohlcv(df: pd.DataFrame, min_rows: int) -> None:
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV frame missing columns: {missing}")
    if len(df) < min_rows:
        raise ValueError(f"need >= {min_rows} candles, got {len(df)}")
    if df[list(OHLCV_COLUMNS)].isna().any().any():
        raise ValueError("OHLCV frame contains NaN values")


def rolling_zscore(close: pd.Series, window: int = 50) -> pd.DataFrame:
    """Z = (P - mu_w) / sigma_w with sample std (ddof=1) over ``window`` bars."""
    mean = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=1)
    z = (close - mean) / std.replace(0.0, np.nan)
    return pd.DataFrame({"zscore": z, "mean": mean, "std": std})


def session_vwap(df: pd.DataFrame, band_mult: float = 2.0) -> pd.DataFrame:
    """Anchored VWAP reset at every UTC day, with volume-weighted sigma bands.

    VWAP_t = sum(TP_i * V_i) / sum(V_i) over the session, where TP is the
    typical price (H+L+C)/3. The band uses the volume-weighted variance
    sum(V_i * (TP_i - VWAP_t)^2) / sum(V_i).
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"]
    session = df.index.floor("D")
    cum_v = vol.groupby(session).cumsum()
    cum_pv = (tp * vol).groupby(session).cumsum()
    cum_p2v = (tp * tp * vol).groupby(session).cumsum()
    vwap = cum_pv / cum_v.replace(0.0, np.nan)
    variance = (cum_p2v / cum_v.replace(0.0, np.nan) - vwap**2).clip(lower=0.0)
    sigma = np.sqrt(variance)
    return pd.DataFrame(
        {"vwap": vwap, "upper": vwap + band_mult * sigma, "lower": vwap - band_mult * sigma}
    )


def volume_profile(df: pd.DataFrame, bins: int = 48, value_area: float = 0.70) -> VolumeProfile:
    """Volume-at-price histogram → Point of Control and Value Area (70%).

    Each candle's volume is spread uniformly across the bins its [low, high]
    range overlaps, which is far more faithful than bucketing by close only.
    """
    lo, hi = float(df["low"].min()), float(df["high"].max())
    if hi <= lo:
        return VolumeProfile(poc=lo, value_area_high=hi, value_area_low=lo)
    edges = np.linspace(lo, hi, bins + 1)
    hist = np.zeros(bins)
    lows = df["low"].to_numpy()
    highs = df["high"].to_numpy()
    vols = df["volume"].to_numpy()
    for lo_, hi_, v in zip(lows, highs, vols, strict=True):
        i0 = max(int(np.searchsorted(edges, lo_, side="right")) - 1, 0)
        i1 = min(int(np.searchsorted(edges, hi_, side="left")), bins)
        i1 = max(i1, i0 + 1)
        hist[i0:i1] += v / (i1 - i0)
    centers = (edges[:-1] + edges[1:]) / 2.0
    poc_idx = int(np.argmax(hist))

    # Grow the value area outward from the POC, always taking the heavier side.
    total = hist.sum()
    left = right = poc_idx
    acc = hist[poc_idx]
    while acc < value_area * total and (left > 0 or right < bins - 1):
        lv = hist[left - 1] if left > 0 else -1.0
        rv = hist[right + 1] if right < bins - 1 else -1.0
        if rv >= lv:
            right += 1
            acc += hist[right]
        else:
            left -= 1
            acc += hist[left]
    return VolumeProfile(
        poc=float(centers[poc_idx]),
        value_area_high=float(edges[right + 1]),
        value_area_low=float(edges[left]),
    )


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average True Range (RMA smoothing, alpha = 1/period)."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
