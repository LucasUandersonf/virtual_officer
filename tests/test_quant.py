from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mqmas.quant.evidence import compute_setup_stats, event_study
from mqmas.quant.forecast import polynomial_forecast
from mqmas.quant.indicators import atr, rolling_zscore, session_vwap, volume_profile
from mqmas.quant.orderbook import estimate_buy_slippage, orderbook_metrics
from mqmas.quant.pipeline import QuantPipeline
from mqmas.quant.sizing import build_sizing_plan, kelly_fraction, trade_sharpe, unit_economics


def test_zscore_matches_definition(ohlcv):
    z = rolling_zscore(ohlcv["close"], 50)
    win = ohlcv["close"].iloc[-50:]
    expected = (win.iloc[-1] - win.mean()) / win.std(ddof=1)
    assert z["zscore"].iloc[-1] == pytest.approx(expected)
    assert z["zscore"].iloc[:49].isna().all()


def test_vwap_resets_daily_and_is_bounded(ohlcv):
    vw = session_vwap(ohlcv)
    day = pd.Series(ohlcv.index.floor("D"), index=ohlcv.index)
    first_of_day = day != day.shift(1)
    tp = (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3
    assert np.allclose(vw["vwap"][first_of_day], tp[first_of_day])
    assert (vw["lower"] <= vw["vwap"] + 1e-9).all() and (vw["upper"] >= vw["vwap"] - 1e-9).all()


def test_volume_profile_poc_at_heavy_level():
    idx = pd.date_range("2026-01-01", periods=30, freq="15min", tz="UTC")
    price = np.full(30, 100.0)
    vol = np.ones(30)
    price[10:20] = 110.0
    vol[10:20] = 50.0
    df = pd.DataFrame({"open": price, "high": price + 0.5, "low": price - 0.5, "close": price, "volume": vol}, index=idx)
    vp = volume_profile(df, bins=20)
    assert abs(vp.poc - 110.0) < 1.0
    assert vp.value_area_low <= vp.poc <= vp.value_area_high


def test_atr_positive(ohlcv):
    a = atr(ohlcv, 14)
    assert a.iloc[-1] > 0 and a.iloc[:13].isna().all()


def test_orderbook_imbalance_and_microprice():
    bids = [[99.0, 3.0], [98.0, 1.0]]
    asks = [[101.0, 1.0], [102.0, 1.0]]
    m = orderbook_metrics(bids, asks, levels=5, previous_imbalance=0.1)
    assert m.imbalance == pytest.approx((4 - 2) / 6)
    assert m.depth_delta == pytest.approx(m.imbalance - 0.1)
    assert m.microprice == pytest.approx((101 * 3 + 99 * 1) / 4)  # leans toward the ask
    with pytest.raises(ValueError):
        orderbook_metrics([[101, 1]], [[100, 1]])


def test_slippage_walk():
    avg, slip = estimate_buy_slippage([[100.0, 1.0], [101.0, 1.0]], 1.5)
    assert avg == pytest.approx((100 + 50.5) / 1.5)
    assert slip == pytest.approx((avg - 100) / 100)
    with pytest.raises(ValueError):
        estimate_buy_slippage([[100.0, 1.0]], 2.0)


def test_forecast_recovers_linear_trend():
    y = 100 + 0.5 * np.arange(60) + np.random.default_rng(1).normal(0, 0.05, 60)
    fc = polynomial_forecast(y, horizon=4)
    assert fc.expected_price == pytest.approx(100 + 0.5 * 63, abs=0.3)
    assert fc.lower_95 < fc.expected_price < fc.upper_95
    assert fc.slope_per_bar == pytest.approx(0.5, abs=0.05)
    assert fc.r_squared > 0.99


def test_kelly_and_sharpe():
    assert kelly_fraction(0.5, 1.0) == pytest.approx(0.0)
    assert kelly_fraction(0.75, 0.5) == pytest.approx(0.25)
    e, sr = trade_sharpe(0.75, 0.006, 0.0085)
    assert e == pytest.approx(0.75 * 0.006 - 0.25 * 0.0085)
    assert sr > 0


def test_sizing_never_exceeds_risk_ceiling(settings):
    plan = build_sizing_plan(50.0, 100_000.0, 0.006, 0.008, kelly_applied=1.0, s=settings)
    loss_u, _ = unit_economics(0.006, 0.008, settings)
    assert plan.max_loss_usdt <= 50.0 * settings.max_risk_per_trade_pct + 1e-9
    assert plan.max_loss_usdt == pytest.approx(plan.notional * loss_u)
    assert plan.notional <= 50.0  # spot, no leverage
    zero = build_sizing_plan(50.0, 100_000.0, 0.006, 0.008, kelly_applied=0.0, s=settings)
    assert zero.notional == 0.0


def test_event_study_and_stats(ohlcv, settings):
    bt = event_study(ohlcv, settings)
    assert bt.trades == len(bt.net_returns) and 0 <= bt.wins <= bt.trades
    st = compute_setup_stats(bt, live_trades=0, live_wins=0, stop_pct=0.006, tp_pct=0.006, s=settings)
    assert 0 < st.win_rate < 1 and st.win_rate_lower_95 < st.win_rate
    assert st.kelly_applied >= 0


def test_pipeline_snapshot(ohlcv, settings):
    p = float(ohlcv["close"].iloc[-1])
    bids = [[p - i, 1.0] for i in range(1, 6)]
    asks = [[p + i, 1.0] for i in range(1, 6)]
    snap = QuantPipeline(settings).compute(ohlcv, bids, asks)
    assert snap.price == pytest.approx(p)
    assert snap.forecast.lower_95 < snap.forecast.upper_95
