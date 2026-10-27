from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mqmas.book import swing
from mqmas.book.engine import Signal
from mqmas.models import ExitReason
from tests.test_book import cro_router, make


def daily(n: int = 300, end: str = "2026-10-26") -> pd.DataFrame:
    rng = np.random.default_rng(3)
    idx = pd.date_range(end=pd.Timestamp(end, tz="UTC"), periods=n, freq="1D")
    c = 100 + np.cumsum(rng.normal(0, 1, n)) * 0.3
    df = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1000.0}, index=idx)
    df.iloc[-1, df.columns.get_loc("close")] = df["high"].iloc[-21:-1].max() + 3  # breakout of the 20-day high
    df.iloc[-1, df.columns.get_loc("high")] = df["close"].iloc[-1] + 0.5
    df.iloc[-1, df.columns.get_loc("volume")] = 5000.0  # with 5× volume
    return df


def test_breakout_fires_with_the_library_stop_rule():
    hits, mgmt = swing.scan(daily())
    one = [h for h in hits if h.setup["tf"] == "1d" and h.setup["nome"] == "rompimento"]
    assert one, [h.layers for h in hits]
    h = one[0]
    assert 1.5 * h.atr - 1e-9 <= h.risk <= 3.0 * h.atr + 1e-9
    assert h.cfg.family == "swing" and h.cfg.exit.hold == 20
    assert set(mgmt["1d"]) == {"low3", "atr", "exit"}


def test_multi_day_candles_only_act_on_the_day_they_complete():
    d = daily(300)
    done = [swing.frame(d.iloc[: len(d) - k], "3d") is not None for k in range(6)]
    assert sum(done) == 2  # one day in three
    assert swing.frame(d, "1d") is d or len(swing.frame(d, "1d")) == len(d)


def test_stop_moves_to_breakeven_then_trails_and_never_down():
    e, r, atr = 100.0, 5.0, 2.0
    assert swing.new_stop(e, 95.0, r, 104.0, atr, 0.0, 0.003) == 95.0
    assert swing.new_stop(e, 95.0, r, 105.0, atr, 0.0, 0.003) == pytest.approx(100.3)
    assert swing.new_stop(e, 95.0, r, 112.0, atr, 0.0, 0.003) == pytest.approx(107.0)  # peak − 2.5 ATR
    assert swing.new_stop(e, 95.0, r, 112.0, atr, 108.0, 0.003) == pytest.approx(108.0)  # low of last 3
    assert swing.new_stop(e, 109.0, r, 112.0, atr, 0.0, 0.003) == 109.0


def swing_signal(target_r: float = 0.0) -> Signal:
    return Signal("SOL/USDT", "swing|1d|trigger=rompimento", 100.0, 4.0, 0.9, {"setup": "1d rompimento"},
                  cfg=swing.SwingHit(swing.SETUPS[0], 100.0, 4.0, 8.0, {}).cfg, risk=8.0, target_r=target_r,
                  meta="|mgmt=be2r|tf=1d|r=8")


@pytest.mark.asyncio
async def test_swing_position_breakeven_trailing_and_exit_pattern(settings):
    db, ex, book = await make(settings, cro_router("APPROVE"))
    ex.px["SOL/USDT"] = 101.0  # fill moved: stop distance is kept, anchored on the fill
    [pos] = await book.try_entries([swing_signal()])
    assert pos.entry_price - pos.stop_price == pytest.approx(8.0)
    assert pos.take_profit_price > 500  # trailing only
    entry = pos.entry_price
    ex.px["SOL/USDT"] = entry + 8.5  # +1R → breakeven covering costs
    await book.supervise()
    [p] = await book.positions()
    assert p.stop_price == pytest.approx(entry * (1 + 2 * (settings.fee_rate + settings.max_slippage_pct)))
    ex.px["SOL/USDT"] = entry + 20.0  # +2.5R → trail at peak − 2.5 ATR
    await book.supervise()
    [p] = await book.positions()
    assert p.stop_price == pytest.approx(entry + 20.0 - 2.5 * 4.0)
    ex.px["SOL/USDT"] = entry + 15.0  # pullback: stop stays
    await book.supervise()
    [p] = await book.positions()
    assert p.stop_price == pytest.approx(entry + 10.0)
    await book._swing_manage({"SOL/USDT": {"1d": {"low3": entry + 12.0, "atr": 4.0, "exit": True}}})
    [closed] = await db.closed_positions("paper")
    assert closed.exit_reason is ExitReason.SIGNAL_EXIT and closed.pnl_usdt > 0
    await db.close()


@pytest.mark.asyncio
async def test_swing_target_in_r_when_a_setup_asks_for_one(settings):
    db, ex, book = await make(settings, cro_router("APPROVE"))
    [pos] = await book.try_entries([swing_signal(2.0)])
    assert pos.take_profit_price - pos.entry_price == pytest.approx(16.0)
    assert "hold=28800" in (pos.setup or "")
    await db.close()


def test_no_setup_caps_the_gain():
    assert all(s["alvo_r"] == 0 for s in swing.SETUPS)
    h = swing.SwingHit(swing.SETUPS[0], 100.0, 4.0, 8.0, {})
    assert h.cfg.exit.target_atr is None
