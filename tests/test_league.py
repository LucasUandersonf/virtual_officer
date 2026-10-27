from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mqmas.league.backtest import metrics, run_tournament, simulate
from mqmas.league.setups import SETUPS, features
from mqmas.models import ExitProfile, LeagueReport
from mqmas.persistence.db import Database


def frame(close: np.ndarray, wick: float = 0.0015, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = close.size
    open_ = np.r_[close[0], close[:-1]]
    w = np.abs(rng.normal(0, wick, n))
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * (1 + w),
                         "low": np.minimum(open_, close) * (1 - w), "close": close,
                         "volume": rng.gamma(2.0, 20.0, n)}, index=idx)


def bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=len(rows), freq="15min", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1.0
    df["atr_pct"] = 0.004
    return df


def test_bracket_target_and_stop_first_on_tie(settings):
    prof = ExitProfile(tp_pct=0.008, stop_atr_mult=1.0)  # stop = 0.4%
    f = bars([(100, 100, 100, 100), (100, 100.9, 99.9, 100.5), (100.5, 100.6, 100.4, 100.5)])
    sig = np.array([True, False, False])
    tr = simulate(f, sig, prof, settings, hold=2)
    assert tr.entry_idx.tolist() == [1]
    assert tr.returns[0] == pytest.approx(0.008 - 2 * settings.fee_rate)

    f = bars([(100, 100, 100, 100), (100, 100.9, 99.5, 100.5), (100.5, 100.6, 100.4, 100.5)])
    tr = simulate(f, sig, prof, settings, hold=2)
    assert tr.returns[0] == pytest.approx(-0.004 - 2 * settings.fee_rate - settings.max_slippage_pct)


def test_time_stop_marks_to_market_and_one_position_at_a_time(settings):
    prof = ExitProfile(tp_pct=0.008, stop_atr_mult=1.0)
    f = bars([(100, 100, 100, 100)] + [(100, 100.2, 99.8, 100.1)] * 5)
    sig = np.array([True, True, True, False, False, False])
    tr = simulate(f, sig, prof, settings, hold=3)
    assert tr.entry_idx.tolist() == [1]  # signals at bars 1-2 overlap the open trade
    assert tr.returns[0] == pytest.approx(0.001 - 2 * settings.fee_rate - settings.max_slippage_pct)


def test_signals_have_no_lookahead():
    rng = np.random.default_rng(11)
    close = 50_000 * np.exp(np.cumsum(rng.normal(0, 0.004, 1200)))
    df = frame(close)
    full = features(df)
    cut = features(df.iloc[:900])
    for st in SETUPS:
        assert (st.signals(full)[:900] == st.signals(cut)).all(), st.key


def test_random_walk_approves_nothing(settings):
    rng = np.random.default_rng(5)
    close = 60_000 * np.exp(np.cumsum(rng.normal(0, 0.003, 20_000)))
    report = run_tournament(frame(close), settings)
    assert len(report.entries) == len(SETUPS)
    assert report.approved == []
    assert all(e.reasons for e in report.entries)
    assert LeagueReport.model_validate_json(report.model_dump_json()).entries[0].key == report.entries[0].key


def test_metrics_posterior_and_profit_factor():
    m = metrics(np.array([0.006, 0.006, 0.006, -0.008]))
    assert m.trades == 4 and m.wins == 3
    assert m.win_rate == pytest.approx(4 / 6)
    assert m.profit_factor == pytest.approx(0.018 / 0.008)
    assert m.max_drawdown_pct == pytest.approx(0.008)


@pytest.mark.asyncio
async def test_candle_cache_and_setup_column(settings):
    db = Database(settings.db_path)
    await db.connect()
    await db.save_candles("BTC/USDT", "15m", [[1000, 1, 2, 0.5, 1.5, 10], [2000, 1.5, 2, 1, 1.8, 12]])
    await db.save_candles("BTC/USDT", "15m", [[2000, 1.5, 2, 1, 1.9, 12]])
    assert await db.last_candle_ts("BTC/USDT", "15m") == 2000
    rows = await db.load_candles("BTC/USDT", "15m")
    assert len(rows) == 2 and rows[-1][4] == 1.9
    await db.prune_candles("BTC/USDT", "15m", 1500)
    assert len(await db.load_candles("BTC/USDT", "15m")) == 1
    assert await db.win_loss_counts("paper", setup="rsi2_dip") == (0, 0)
    await db.close()
