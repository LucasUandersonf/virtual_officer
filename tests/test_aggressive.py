from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mqmas.book.engine import Signal
from mqmas.research import aggressive, arena
from mqmas.research.engine import Costs, RiskRules, simulate_portfolio
from tests.test_book import KEY, cro_router, make


def daily(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.03, n)))
    o = np.r_[c[0], c[:-1]]
    idx = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.01, "low": np.minimum(o, c) * 0.99, "close": c,
                         "volume": rng.random(n) * 1000 + 100}, index=idx)


def test_new_signals_fire_and_are_causal():
    df = daily()
    full = aggressive.new_signals(df)
    cut = aggressive.new_signals(df.iloc[:400])
    for k in aggressive.NEW:
        assert full[k].any(), k
        assert (full[k].iloc[:400].to_numpy() == cut[k].to_numpy()).all(), k  # no look-ahead
    sig, st, _ = aggressive.frame(df)
    assert set(aggressive.NEW) <= set(sig.columns) & set(st.columns)


def test_management_variants_change_exits():
    rng = np.random.default_rng(2)
    n = 3000
    c = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n)))
    o = np.r_[c[0], c[:-1]]
    h, lo, atr = np.maximum(o, c) * 1.01, np.minimum(o, c) * 0.99, np.full(n, 2.0)
    ent = rng.random(n) < 0.03
    args = (o, h, lo, c, atr, ent, np.full(n, np.nan), 0.0, np.zeros(n, bool), Costs(), 20)
    base = arena.manage(*args)
    assert np.allclose(base, arena.manage(*args, 1.0, 2.0, 2.5, True))
    loose = arena.manage(*args, 0.0, 2.0, 3.5, False)
    assert loose.shape != base.shape or not np.allclose(base[:, 2], loose[:, 2])
    assert len(aggressive.GESTAO) == 12 and aggressive.BASE in aggressive.GESTAO


def test_heat_cap_limits_simultaneous_risk():
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    tr = pd.DataFrame({"coin": [f"C{i}" for i in range(5)], "t_in": [t0 + pd.Timedelta(hours=i) for i in range(5)],
                       "t_out": [t0 + pd.Timedelta(days=10)] * 5, "ret": [0.01] * 5, "stop_pct": [0.10] * 5})
    free = simulate_portfolio(tr, RiskRules(equity0=1000, risk_per_trade=0.02, max_positions=5), Costs())
    capped = simulate_portfolio(tr, RiskRules(equity0=1000, risk_per_trade=0.02, max_positions=5, heat_max=0.05),
                                Costs())
    assert len(free.trades) == 5
    at_risk = (capped.trades["notional"] * (0.10 + Costs().round_trip)).sum()
    assert at_risk == pytest.approx(50.0) and len(capped.trades) == 3


@pytest.mark.asyncio
async def test_book_halves_risk_in_drawdown_and_respects_heat(settings):
    db, ex, book = await make(settings, cro_router("APPROVE"))
    full = await book._risk_budget(50.0)
    assert full == pytest.approx(50 * settings.book_risk_per_trade)
    assert await book._risk_budget(44.0) == pytest.approx(44 * settings.book_risk_per_trade * 0.5)  # −12 %
    assert await book._risk_budget(39.0) == pytest.approx(39 * settings.book_risk_per_trade * 0.25)  # −22 %
    await book.try_entries([Signal("SOL/USDT", KEY, 100.0, 4.0, 0.9, {})])
    left = await book._risk_budget(50.0)
    assert left <= 50 * settings.book_heat_max - 50 * settings.book_risk_per_trade + 1e-6
    await db.close()
