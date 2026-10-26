"""Real-time market data service.

* Order book: ccxt.pro WebSocket (``watch_order_book``) with automatic
  degradation to REST polling if the stream fails repeatedly.
* Candles: one paginated REST backfill at boot (``history_bars``) and then
  incremental refreshes merged into the cached frame.
* Every network call is wrapped in exponential backoff with jitter; transient
  ccxt errors never crash the loop, they mark the feed ``stale`` instead, and
  the decision loop refuses to trade on stale data.

A synthetic GBM feed is provided for offline demos and tests (paper only).
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Protocol

import numpy as np
import pandas as pd

from mqmas.config import MarketDataSource, Settings

log = logging.getLogger(__name__)

HISTORY_BARS = 4000  # ~41 days of 15m candles for the event study
STALE_AFTER_SEC = 20.0
TIMEFRAME_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}


class MarketFeed(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def candles(self) -> pd.DataFrame: ...
    def book(self) -> tuple[list[list[float]], list[list[float]]]: ...
    def is_stale(self) -> bool: ...
    @property
    def exchange(self) -> Any: ...


def _to_frame(rows: list[list[float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop_duplicates("ts", keep="last").set_index("ts").sort_index().astype(float)


async def with_backoff(coro_factory: Any, what: str, attempts: int = 5, base: float = 1.0) -> Any:
    """Retry an awaitable factory on ccxt network-type errors."""
    import ccxt  # local import keeps module importable without ccxt for tests

    for attempt in range(attempts):
        try:
            return await coro_factory()
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout, ccxt.DDoSProtection) as exc:
            delay = base * 2**attempt + random.uniform(0, base)
            log.warning("%s failed (%s), retry %d/%d in %.1fs", what, type(exc).__name__, attempt + 1, attempts, delay)
            await asyncio.sleep(delay)
    raise RuntimeError(f"{what}: exhausted {attempts} retries")


class ExchangeFeed:
    def __init__(self, settings: Settings) -> None:
        import ccxt.pro as ccxtpro

        self._s = settings
        opts: dict[str, Any] = {"enableRateLimit": True, "options": {"defaultType": "spot"}}
        if settings.exchange_api_key and settings.exchange_api_secret:
            opts["apiKey"] = settings.exchange_api_key.get_secret_value()
            opts["secret"] = settings.exchange_api_secret.get_secret_value()
        klass = getattr(ccxtpro, settings.exchange_id)
        self._ex = klass(opts)
        if settings.exchange_sandbox:
            self._ex.set_sandbox_mode(True)
        self._df = pd.DataFrame()
        self._bids: list[list[float]] = []
        self._asks: list[list[float]] = []
        self._book_ts = 0.0
        self._tasks: list[asyncio.Task[None]] = []
        self._stopping = asyncio.Event()

    @property
    def exchange(self) -> Any:
        return self._ex

    async def start(self) -> None:
        await with_backoff(self._ex.load_markets, "load_markets")
        market = self._ex.market(self._s.symbol)
        if not market.get("spot"):
            raise RuntimeError(f"{self._s.symbol} is not a spot market on {self._s.exchange_id}")
        await self._backfill()
        self._tasks = [
            asyncio.create_task(self._book_loop(), name="book_loop"),
            asyncio.create_task(self._candle_loop(), name="candle_loop"),
        ]

    async def stop(self) -> None:
        self._stopping.set()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._ex.close()

    def candles(self) -> pd.DataFrame:
        return self._df

    def book(self) -> tuple[list[list[float]], list[list[float]]]:
        return self._bids, self._asks

    def is_stale(self) -> bool:
        return self._df.empty or not self._bids or (time.time() - self._book_ts) > STALE_AFTER_SEC

    async def _backfill(self) -> None:
        tf_ms = TIMEFRAME_MS.get(self._s.timeframe, 900_000)
        since = self._ex.milliseconds() - HISTORY_BARS * tf_ms
        rows: list[list[float]] = []
        while since < self._ex.milliseconds() - tf_ms:
            batch = await with_backoff(
                lambda s=since: self._ex.fetch_ohlcv(self._s.symbol, self._s.timeframe, since=s, limit=self._s.ohlcv_limit),
                "fetch_ohlcv(backfill)",
            )
            if not batch:
                break
            rows.extend(batch)
            since = int(batch[-1][0]) + tf_ms
        self._df = _to_frame(rows)
        log.info("backfilled %d candles for %s", len(self._df), self._s.symbol)

    async def _candle_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                recent = await with_backoff(
                    lambda: self._ex.fetch_ohlcv(self._s.symbol, self._s.timeframe, limit=5), "fetch_ohlcv"
                )
                merged = pd.concat([self._df, _to_frame(recent)])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                self._df = merged.iloc[-HISTORY_BARS:]
            except Exception:
                log.exception("candle refresh failed")
            await asyncio.sleep(15)

    async def _book_loop(self) -> None:
        ws_failures = 0
        while not self._stopping.is_set():
            try:
                if ws_failures < 5 and self._ex.has.get("watchOrderBook"):
                    ob = await self._ex.watch_order_book(self._s.symbol, limit=20)
                    ws_failures = 0
                else:
                    ob = await with_backoff(lambda: self._ex.fetch_order_book(self._s.symbol, limit=20), "fetch_order_book")
                    await asyncio.sleep(2)
                    if random.random() < 0.02:
                        ws_failures = 0  # periodically retry the stream
                self._bids, self._asks = ob["bids"][:20], ob["asks"][:20]
                self._book_ts = time.time()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                ws_failures += 1
                log.warning("order book stream error (%s): %s", ws_failures, exc)
                await asyncio.sleep(min(2**ws_failures, 30))


class SyntheticFeed:
    """Geometric Brownian motion with occasional mean-reverting dislocations."""

    def __init__(self, settings: Settings, seed: int = 7) -> None:
        self._s = settings
        self._rng = np.random.default_rng(seed)
        self._df = pd.DataFrame()
        self._price = 100_000.0
        self._task: asyncio.Task[None] | None = None

    @property
    def exchange(self) -> Any:
        return None

    async def start(self) -> None:
        n = 1500
        idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("15min"), periods=n, freq="15min")
        rets = self._rng.normal(0, 0.0025, n)
        close = self._price * np.exp(np.cumsum(rets))
        self._df = self._bars(idx, close)
        self._price = float(close[-1])
        self._task = asyncio.create_task(self._tick())

    def _bars(self, idx: pd.DatetimeIndex, close: np.ndarray) -> pd.DataFrame:
        open_ = np.concatenate([[close[0]], close[:-1]])
        wig = np.abs(self._rng.normal(0, 0.0012, len(close)))
        return pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) * (1 + wig),
                "low": np.minimum(open_, close) * (1 - wig),
                "close": close,
                "volume": self._rng.gamma(2.0, 20.0, len(close)),
            },
            index=idx,
        )

    async def _tick(self) -> None:
        while True:
            await asyncio.sleep(2)
            shock = self._rng.normal(0, 0.0006)
            if self._rng.random() < 0.01:
                shock -= 0.006  # dislocation → oversold candidates for the demo
            self._price *= float(np.exp(shock))
            now = pd.Timestamp.now(tz="UTC").floor("15min")
            last = self._df.index[-1]
            if now > last:
                row = self._bars(pd.DatetimeIndex([now]), np.array([self._price]))
                self._df = pd.concat([self._df, row]).iloc[-HISTORY_BARS:]
            else:
                self._df.loc[last, "close"] = self._price
                self._df.loc[last, "high"] = max(self._df.loc[last, "high"], self._price)
                self._df.loc[last, "low"] = min(self._df.loc[last, "low"], self._price)
                self._df.loc[last, "volume"] += float(self._rng.gamma(1.0, 1.0))

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    def candles(self) -> pd.DataFrame:
        return self._df

    def book(self) -> tuple[list[list[float]], list[list[float]]]:
        p = self._price
        tick = 0.01
        bids = [[p - tick * (i + 1), float(self._rng.gamma(2.0, 0.4))] for i in range(20)]
        asks = [[p + tick * (i + 1), float(self._rng.gamma(2.0, 0.4))] for i in range(20)]
        return bids, asks

    def is_stale(self) -> bool:
        return bool(self._df.empty)


def build_feed(settings: Settings) -> MarketFeed:
    if settings.market_data_source is MarketDataSource.SYNTHETIC:
        return SyntheticFeed(settings)
    return ExchangeFeed(settings)
