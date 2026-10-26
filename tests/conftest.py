from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mqmas.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, db_path=tmp_path / "t.db")


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 600
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    close = 100_000 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) * 1.001,
        "low": np.minimum(open_, close) * 0.999,
        "close": close,
        "volume": rng.gamma(2.0, 10.0, n),
    }, index=idx)
