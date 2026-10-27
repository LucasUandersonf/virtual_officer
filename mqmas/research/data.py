"""Universe selection and OHLCV history cache for the research lab.

The candidate pool is today's most liquid Binance spot USDT pairs, which
introduces **survivorship bias** (coins that died or were delisted since 2022
are missing). The backtest reduces, but cannot remove, that bias by
re-selecting the tradable universe every month from *trailing* volume and
volatility only.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TF_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
EXCLUDED_BASES = {
    "USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "EUR", "AEUR", "EURI", "USDE", "PYUSD", "USD1",
    "RLUSD", "XUSD", "BFUSD", "PAXG", "XAUT", "WBTC", "WBETH", "BETH", "STETH", "BNSOL", "UST", "USTC",
    # tokenized equities / ETFs that showed up in the volume ranking: not crypto, different market hours
    "SPCXB", "SNDKB", "SOXLB", "CRCLB", "MUB", "TSLAB", "NVDAB", "AAPLB", "MSTRB", "COINB", "SPYB", "QQQB",
    "GOOGLB", "INTCB", "SKHYB", "AMZNB", "METAB", "MSFTB", "AMDB", "HOODB", "PLTRB",
    "RWUSD",
}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")


def is_crypto(base: str) -> bool:
    """False for stablecoins, wrappers, tokenized equities and leveraged tokens."""
    if base in EXCLUDED_BASES or base.endswith(LEVERAGED_SUFFIXES) or len(base) < 2:
        return False
    return not (base.endswith("USD") or base.startswith("USD"))  # dollar stablecoins not yet in the list


def candidate_pool(ex: object, size: int) -> list[str]:
    """Most traded USDT spot pairs right now, excluding stables, wrappers and leveraged tokens."""
    markets = ex.load_markets()  # type: ignore[attr-defined]
    tickers = ex.fetch_tickers()  # type: ignore[attr-defined]
    rows: list[tuple[float, str]] = []
    for sym, m in markets.items():
        if not (m.get("spot") and m.get("active") and m.get("quote") == "USDT"):
            continue
        base = str(m.get("base", ""))
        if not is_crypto(base):
            continue
        qv = float((tickers.get(sym) or {}).get("quoteVolume") or 0.0)
        rows.append((qv, sym))
    rows.sort(reverse=True)
    pool = [s for _, s in rows[:size]]
    if "BTC/USDT" not in pool:
        pool.insert(0, "BTC/USDT")
    return pool


def _cache_file(root: Path, symbol: str, tf: str) -> Path:
    return root / f"{symbol.replace('/', '_')}_{tf}.npz"


def load_history(ex: object, symbol: str, tf: str, since_ms: int, root: Path) -> pd.DataFrame:
    """Closed candles since ``since_ms``; cached on disk, only the missing tail is fetched."""
    root.mkdir(parents=True, exist_ok=True)
    path = _cache_file(root, symbol, tf)
    data = np.load(path)["rows"] if path.exists() else np.empty((0, 6))
    step = TF_MS[tf]
    now = int(time.time() * 1000)
    cursor = int(data[-1, 0]) + step if len(data) else since_ms
    new: list[list[float]] = []
    while cursor < now - step:
        for attempt in range(5):
            try:
                batch = ex.fetch_ohlcv(symbol, tf, since=cursor, limit=1000)  # type: ignore[attr-defined]
                break
            except Exception as exc:  # network hiccups: back off and retry
                log.warning("fetch %s %s failed (%s), retry %d", symbol, tf, type(exc).__name__, attempt + 1)
                time.sleep(2 ** attempt)
        else:
            break
        closed = [r for r in batch if int(r[0]) + step <= now]
        if not closed:
            break
        new.extend(closed)
        cursor = int(closed[-1][0]) + step
    if new:
        data = np.vstack([data, np.asarray(new, dtype=float)]) if len(data) else np.asarray(new, dtype=float)
        np.savez_compressed(path, rows=data)
    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df[df["ts"] >= since_ms]
    df.index = pd.to_datetime(df.pop("ts").astype("int64"), unit="ms", utc=True)
    return df[~df.index.duplicated(keep="last")].sort_index()
