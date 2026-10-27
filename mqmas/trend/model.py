"""Trend ensemble: the rules shared, line for line, by the backtest and the live allocator.

Reverse-engineered from the best documented profitable crypto system we found, Zarattini, Pagani & Barbon
(2025), "Catching Crypto Trends; A Tactical Approach for Bitcoin and Altcoins" (SSRN 5209907): an ensemble
of Donchian channel models on daily closes, sized by volatility, on a rotating universe of the most liquid
coins. It is the same family as our own walk-forward winner (daily breakout), with three differences that
remove the parts our research kept failing on:

* **No parameter is optimised.** Nine fixed lookbacks are averaged instead of picking the "best" one,
  so there is nothing to overfit and nothing to re-select every window.
* **No fixed stop or target.** Each model exits when the close falls below its own ratcheting channel
  midline; the winners run as long as the trend does.
* **Size by risk, not by conviction.** Weight = signal strength / volatility, then the whole basket is
  scaled down to a volatility target. A coin that trends weakly or swings wildly gets a small slice.

Spot only, long or cash, never leverage: the weights always sum to at most 100 %.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOOKBACKS = (5, 10, 20, 30, 60, 90, 150, 250, 360)
VOL_WINDOW = 90
MIN_HISTORY = 365  # a coin trades only after a full year of daily candles (as in the paper)
VOLUME_WINDOW = 30
# Exchange-side disaster stop for when the bot is down; trails the close and never moves down. Wide on purpose:
# the model's own exit is the daily close under the channel midline, and a 3-ATR stop tested in v1 sold normal
# swings (full-period Sharpe 0.89 with it, 1.18 without). At 6 ATR it only fires in a crash.
CRASH_STOP_ATR = 6.0
ATR_WINDOW = 14
DAYS_PER_YEAR = 365


@dataclass(frozen=True)
class TrendParams:
    vol_target: float = 0.40  # annualised volatility of the whole basket
    top_n: int = 10  # universe: the N most liquid eligible coins, re-ranked every month
    max_weight: float = 0.35  # no single coin above this share of equity

    @property
    def key(self) -> str:
        return f"trend|1d|vt={self.vol_target:g},top={self.top_n},maxw={self.max_weight:g}"


def donchian_model(close: np.ndarray, n: int) -> np.ndarray:
    """1 while one Donchian model is long, else 0.

    Enter when the close makes a new ``n``-day closing high; while long, the stop is the channel midline
    (half-way between the ``n``-day closing high and low), ratcheted so it never moves down; exit on the
    first close below the stop of the previous day.
    """
    s = pd.Series(close)
    hi = s.rolling(n, min_periods=n).max().to_numpy()
    lo = s.rolling(n, min_periods=n).min().to_numpy()
    mid = (hi + lo) / 2
    out = np.zeros(len(close))
    long, stop = False, -math.inf
    for t in range(len(close)):
        if not np.isfinite(hi[t]):
            continue
        if long and close[t] < stop:
            long = False
        elif not long and close[t] >= hi[t]:
            long, stop = True, -math.inf
        if long:
            stop = max(stop, mid[t])
            out[t] = 1.0
    return out


def ensemble_signal(close: pd.Series) -> pd.Series:
    """Average of the nine models: 0 (no trend on any horizon) … 1 (long on every horizon)."""
    c = close.to_numpy(dtype=float)
    states = np.vstack([donchian_model(c, n) for n in LOOKBACKS])
    return pd.Series(states.mean(axis=0), index=close.index)


def realised_vol(close: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    r = np.log(close).diff()
    return r.rolling(window, min_periods=window // 2).std() * math.sqrt(DAYS_PER_YEAR)


def atr(df: pd.DataFrame, n: int = ATR_WINDOW) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def eligibility(daily: dict[str, pd.DataFrame], top_n: int) -> pd.DataFrame:
    """Boolean frame (day × coin): in the universe that day.

    Point in time: on the first day of each month the coins with a full year of history are ranked by
    the median dollar volume of the previous 30 days; the top ``top_n`` stay eligible for that month.
    """
    closes = pd.DataFrame({s: d["close"] for s, d in daily.items()}).sort_index()
    dollar = pd.DataFrame({s: d["close"] * d["volume"] for s, d in daily.items()}).reindex(closes.index)
    age = closes.notna().cumsum()
    liq = dollar.rolling(VOLUME_WINDOW, min_periods=VOLUME_WINDOW // 2).median().shift(1)
    out = pd.DataFrame(False, index=closes.index, columns=closes.columns)
    months = closes.index.tz_localize(None).to_period("M")
    for m in months.unique():
        days = closes.index[months == m]
        first = days[0]
        ok = (age.loc[first] >= MIN_HISTORY) & liq.loc[first].notna()
        ranked = liq.loc[first][ok].sort_values(ascending=False)
        out.loc[days, ranked.index[:top_n]] = True
    return out & closes.notna()


def target_weights(signal: pd.Series, vol: pd.Series, eligible: pd.Series, cov: np.ndarray | None,
                   params: TrendParams) -> pd.Series:
    """Portfolio weights for one day from per-coin signal, volatility and the coins' covariance.

    Inverse-volatility risk budget over the *whole* eligible universe, multiplied by each coin's signal:
    with every coin trending on every horizon the basket is 100 % invested; with few trends it is mostly
    in USDT. Then each coin is capped at ``max_weight`` and the basket is scaled down (never up) so its
    expected volatility does not exceed ``vol_target``.
    """
    cols = signal.index
    ok = eligible.reindex(cols).fillna(False).to_numpy(dtype=bool).copy()
    v = vol.reindex(cols).to_numpy(dtype=float)
    ok &= np.isfinite(v) & (v > 0)
    w = np.zeros(len(cols))
    if not ok.any():
        return pd.Series(w, index=cols)
    inv = np.where(ok, 1.0 / np.where(v > 0, v, np.inf), 0.0)
    s = np.nan_to_num(signal.to_numpy(dtype=float)) * ok
    w = s * inv / inv.sum()
    w = np.minimum(w, params.max_weight)
    if cov is not None and w.sum() > 0:
        c = np.nan_to_num(cov)
        port_vol = math.sqrt(max(float(w @ c @ w), 0.0) * DAYS_PER_YEAR)
        if port_vol > params.vol_target:
            w *= params.vol_target / port_vol
    total = w.sum()
    if total > 1.0:
        w /= total
    return pd.Series(w, index=cols)


def daily_cov(closes: pd.DataFrame, window: int = VOL_WINDOW) -> dict[pd.Timestamp, np.ndarray]:
    """Rolling covariance of daily log returns (per day, coins in the closes' column order)."""
    r = np.log(closes).diff()
    out: dict[pd.Timestamp, np.ndarray] = {}
    vals = r.to_numpy()
    for i, day in enumerate(closes.index):
        if i < window // 2:
            continue
        win = vals[max(0, i - window + 1): i + 1]
        frame = pd.DataFrame(win)
        out[day] = frame.cov(min_periods=window // 3).to_numpy()
    return out


def weights_frame(daily: dict[str, pd.DataFrame], params: TrendParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Target weights (day × coin, decided on that day's close) and the per-coin ensemble signal."""
    closes = pd.DataFrame({s: d["close"] for s, d in daily.items()}).sort_index()
    sig = pd.DataFrame({s: ensemble_signal(closes[s].dropna()) for s in closes.columns}).reindex(closes.index)
    vol = pd.DataFrame({s: realised_vol(closes[s].dropna()) for s in closes.columns}).reindex(closes.index)
    elig = eligibility(daily, params.top_n)
    covs = daily_cov(closes)
    rows = [target_weights(sig.loc[d], vol.loc[d], elig.loc[d], covs.get(d), params) for d in closes.index]
    return pd.DataFrame(rows, index=closes.index), sig


# ---------------------------------------------------------------------------------------------- wave hunter

MOM_TOP_N = 30  # universe of the wave hunter
MOM_K = 5  # coins held
MOM_LOOKBACK = 21  # 3 weeks, the middle of Liu-Tsyvinski-Wu's 1–4 week range
MOM_BTC_SMA = 100  # in USDT while BTC closes under its 100-day average


def momentum_frame(daily: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wave hunter: every Monday close, the 5 coins with the best 3-week return, equal weight, while BTC is
    above its 100-day average (else all USDT). Returns the daily target weights and the momentum score.

    Validated in ``mqmas.research.waves`` (Binance 2020-01→2026-10, costs included): CAGR +135 %, max drawdown
    67 %, Sharpe 1.57; +92 % without its three best coins. A high-volatility sleeve, not a whole portfolio.
    """
    closes = pd.DataFrame({s: d["close"] for s, d in daily.items()}).sort_index()
    elig = eligibility(daily, MOM_TOP_N)
    score = closes / closes.shift(MOM_LOOKBACK) - 1
    btc = closes["BTC/USDT"]
    regime = btc > btc.rolling(MOM_BTC_SMA).mean()
    return weekly_top(closes, elig, score, regime, MOM_K), score


def weekly_top(closes: pd.DataFrame, elig: pd.DataFrame, score: pd.DataFrame, regime: pd.Series | None,
               k: int) -> pd.DataFrame:
    """Equal-weight top-``k`` by ``score`` on each Monday close (only positive scores), held until next Monday."""
    w = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
    for day in closes.index[closes.index.dayofweek == 0]:
        row = pd.Series(0.0, index=closes.columns)
        if regime is None or bool(regime.get(day, False)):
            sc = score.loc[day].where(elig.loc[day]).dropna()
            top = sc[sc > 0].sort_values(ascending=False).index[:k]
            if len(top):
                row[top] = 1.0 / len(top)
        w.loc[day] = row
    return w.ffill().fillna(0.0)
