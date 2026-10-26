"""Assembles a full ``QuantSnapshot`` from candles + order book.

The heavy pandas/numpy work runs in a worker thread (``asyncio.to_thread``)
so the event loop that serves WebSockets never stalls on computation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pandas as pd

from mqmas.config import Settings
from mqmas.models import QuantSnapshot
from mqmas.quant.forecast import polynomial_forecast
from mqmas.quant.indicators import atr, rolling_zscore, session_vwap, validate_ohlcv, volume_profile
from mqmas.quant.orderbook import Level, orderbook_metrics

PROFILE_BARS = 96  # 24h of 15m candles for the volume profile


class QuantPipeline:
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._prev_imbalance: float | None = None

    def compute(
        self, df: pd.DataFrame, bids: Sequence[Level], asks: Sequence[Level]
    ) -> QuantSnapshot:
        s = self._s
        validate_ohlcv(df, max(s.zscore_window, s.regression_lookback, 20) + 1)
        z = rolling_zscore(df["close"], s.zscore_window).iloc[-1]
        vw = session_vwap(df).iloc[-1]
        a = float(atr(df).iloc[-1])
        ob = orderbook_metrics(bids, asks, s.ob_levels, self._prev_imbalance)
        self._prev_imbalance = ob.imbalance
        fc = polynomial_forecast(
            df["close"].to_numpy()[-s.regression_lookback :], horizon=s.forecast_horizon
        )
        # Use the live mid as "price": the last candle close may be seconds stale.
        price = ob.mid
        return QuantSnapshot(
            symbol=s.symbol,
            price=price,
            zscore=float((price - z["mean"]) / z["std"]) if z["std"] > 0 else 0.0,
            rolling_mean=float(z["mean"]),
            rolling_std=float(z["std"]),
            vwap=float(vw["vwap"]),
            vwap_upper=float(vw["upper"]),
            vwap_lower=float(vw["lower"]),
            volume_profile=volume_profile(df.iloc[-PROFILE_BARS:]),
            atr=a,
            atr_pct=a / price,
            orderbook=ob,
            forecast=fc,
        )

    async def compute_async(
        self, df: pd.DataFrame, bids: Sequence[Level], asks: Sequence[Level]
    ) -> QuantSnapshot:
        return await asyncio.to_thread(self.compute, df, bids, asks)
