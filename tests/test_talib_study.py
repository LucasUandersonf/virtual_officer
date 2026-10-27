from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

talib = pytest.importorskip("talib")

from mqmas.research import talib_study  # noqa: E402


def daily(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.03, n)))
    o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.005, n))
    idx = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * (1 + rng.random(n) * 0.02),
                         "low": np.minimum(o, c) * (1 - rng.random(n) * 0.02), "close": c,
                         "volume": rng.random(n) * 1000 + 100}, index=idx)


def test_all_61_patterns_become_bullish_signals_with_a_stop():
    df = daily()
    sig, st, exits = talib_study.frame(df)
    cdl = [c for c in sig.columns if c.startswith("cdl_")]
    assert len(cdl) == 61 and len(talib_study.patterns()) == 61
    assert sig[cdl].to_numpy().any() and set(cdl) <= set(st.columns)
    engulf = talib.CDLENGULFING(*(df[x].to_numpy() for x in ("open", "high", "low", "close")))
    assert (sig["cdl_engulfing"].to_numpy() == (engulf > 0)).all()  # bearish engulfing never fires a buy
    assert len(exits) == len(df)


def test_candidates_cover_alone_macro_and_breakout_confirmation():
    c = talib_study.candidates("2d")
    assert sum(x.get("referencia", False) for x in c) == 2
    assert len(c) == 2 + 61 * 2 * 3
    assert all(x["tf"] == "2d" for x in c)


def test_our_indicators_match_talib():
    gaps = talib_study.audit(daily())
    assert gaps["EMA20"] < 0.01 and gaps["ATR14"] < 0.01 and gaps["RSI14"] < 0.01
    assert gaps["Bollinger sup"] < 1e-6
