"""Library of long-only spot setups, each an exact and testable entry rule.

Every rule reads only the *closed* candle ``i`` and earlier bars (``shift`` for
anything "previous"), so the same function drives the backtest and the live
signal without look-ahead. Exits are not part of a setup: all setups share the
mandate's bracket (TP 0.6–0.8 %, ATR stop clamped to 0.3–0.6 %, time stop),
and the tournament picks the best bracket per setup on training data only.

Order-book based ideas (absorption, imbalance) cannot be backtested from
OHLCV history, so they stay as live confirmation gates instead of setups.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mqmas.quant.indicators import atr, rolling_zscore, session_vwap

WARMUP_BARS = 250  # EMA200 + rolling windows must be settled before any signal


def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(gain.notna())


def features(df: pd.DataFrame) -> pd.DataFrame:
    """All indicators the setups need, computed once per frame."""
    f = df[["open", "high", "low", "close", "volume"]].copy()
    c = f["close"]
    f["ema20"], f["ema50"], f["ema200"] = ema(c, 20), ema(c, 50), ema(c, 200)
    f["rsi2"], f["rsi14"] = rsi(c, 2), rsi(c, 14)
    f["rsi14_min20"] = f["rsi14"].shift(1).rolling(20).min()
    f["z50"] = rolling_zscore(c, 50)["zscore"]
    f["vwap"] = session_vwap(df)["vwap"]
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    f["bb_up"], f["bb_lo"] = mid + 2 * sd, mid - 2 * sd
    f["bbw"] = (f["bb_up"] - f["bb_lo"]) / mid
    f["bbw_q20"] = f["bbw"].rolling(120).quantile(0.2)
    f["vol20"] = f["volume"].shift(1).rolling(20).mean()
    f["hh20"] = f["high"].shift(1).rolling(20).max()
    f["ll20"] = f["low"].shift(1).rolling(20).min()
    a = atr(df)
    f["atr_pct"] = a / c
    return f


Rule = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class Setup:
    key: str
    name: str
    family: str
    thesis: str
    rule: Rule

    def signals(self, f: pd.DataFrame) -> np.ndarray:
        s: np.ndarray = self.rule(f).fillna(False).to_numpy(dtype=bool, copy=True)
        s[:WARMUP_BARS] = False
        return s


def _zscore_reversion(f: pd.DataFrame) -> pd.Series:
    return (f["z50"] <= -2.0) & (f["z50"].shift(1) > -2.0) & (f["close"] < f["vwap"])


def _rsi2_dip(f: pd.DataFrame) -> pd.Series:
    return (f["rsi2"] < 10) & (f["close"] > f["ema200"])


def _bb_reentry(f: pd.DataFrame) -> pd.Series:
    return (f["close"].shift(1) < f["bb_lo"].shift(1)) & (f["close"] > f["bb_lo"]) & (f["close"] > f["open"])


def _vwap_reclaim(f: pd.DataFrame) -> pd.Series:
    return ((f["close"].shift(1) < f["vwap"].shift(1)) & (f["close"] > f["vwap"])
            & (f["close"] > f["ema50"]) & (f["volume"] > 1.2 * f["vol20"]))


def _trend_pullback(f: pd.DataFrame) -> pd.Series:
    return ((f["ema20"] > f["ema50"]) & (f["ema50"] > f["ema200"]) & (f["low"] <= f["ema20"])
            & (f["close"] > f["ema20"]) & f["rsi14"].between(40, 60))


def _donchian_breakout(f: pd.DataFrame) -> pd.Series:
    return (f["close"] > f["hh20"]) & (f["volume"] > 1.5 * f["vol20"]) & (f["close"] > f["ema200"])


def _squeeze_breakout(f: pd.DataFrame) -> pd.Series:
    return ((f["bbw"].shift(1) <= f["bbw_q20"].shift(1)) & (f["close"] > f["bb_up"])
            & (f["close"] > f["ema50"]))


def _capitulation_wick(f: pd.DataFrame) -> pd.Series:
    rng = (f["high"] - f["low"]).replace(0.0, np.nan)
    return ((f["volume"] > 2.5 * f["vol20"]) & ((f["close"] - f["low"]) / rng >= 0.6)
            & (f["z50"] <= -1.5))


def _rsi_divergence(f: pd.DataFrame) -> pd.Series:
    return ((f["low"] <= f["ll20"]) & (f["rsi14"] > f["rsi14_min20"] + 3) & (f["rsi14"] < 40)
            & (f["close"] > f["open"]))


SETUPS: tuple[Setup, ...] = (
    Setup("zscore_reversion", "Reversão Z-Score", "reversão",
          "Preço 2σ abaixo da média de 50 barras e abaixo da VWAP tende a voltar à média.", _zscore_reversion),
    Setup("rsi2_dip", "RSI-2 em tendência", "reversão",
          "Connors: RSI(2) < 10 acima da EMA200 compra o recuo curto dentro da alta.", _rsi2_dip),
    Setup("bb_reentry", "Reentrada Bollinger", "reversão",
          "Fechamento volta para dentro da banda inferior após rompê-la: exaustão vendedora.", _bb_reentry),
    Setup("vwap_reclaim", "Recuperação da VWAP", "fluxo",
          "Preço reconquista a VWAP do dia com volume, acima da EMA50: compradores no controle.", _vwap_reclaim),
    Setup("trend_pullback", "Pullback na EMA20", "tendência",
          "EMAs 20>50>200 alinhadas e toque na EMA20 com RSI neutro: continuação da alta.", _trend_pullback),
    Setup("donchian_breakout", "Rompimento Donchian", "rompimento",
          "Fechamento acima da máxima de 20 barras com volume 1,5× e acima da EMA200.", _donchian_breakout),
    Setup("squeeze_breakout", "Squeeze de Bollinger", "rompimento",
          "Compressão de volatilidade (largura no menor quintil) seguida de rompimento da banda superior.",
          _squeeze_breakout),
    Setup("capitulation_wick", "Capitulação com pavio", "reversão",
          "Volume 2,5× com pavio inferior longo após queda esticada: absorção da capitulação.", _capitulation_wick),
    Setup("rsi_divergence", "Divergência de RSI", "reversão",
          "Nova mínima de 20 barras com RSI mais alto que na mínima anterior: força vendedora acabando.",
          _rsi_divergence),
)

SETUPS_BY_KEY = {s.key: s for s in SETUPS}
