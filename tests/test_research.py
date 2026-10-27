from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mqmas.research.engine import Costs, Exit, RiskRules, bracket_trades, simulate_portfolio
from mqmas.research.lab import deflated_sharpe

NO_COST = Costs(0.0, 0.0)


def arr(*xs: float) -> np.ndarray:
    return np.array(xs, dtype=float)


def test_gap_through_stop_fills_at_open():
    # entry at open 100 on bar 1, stop 98 (2×ATR=1); bar 2 opens at 90 (flash crash)
    o, h, lo, c = arr(100, 100, 90, 91), arr(100, 101, 92, 92), arr(100, 99, 89, 90), arr(100, 100, 91, 91)
    t = bracket_trades(o, h, lo, c, arr(1, 1, 1, 1), np.array([True, False, False, False]),
                       Exit(2.0, None, None, 2), NO_COST)
    assert t["ret"][0] == pytest.approx(-0.10)  # not the -2% the stop "promised"
    assert t["stop_pct"][0] == pytest.approx(0.02)


def test_trailing_stop_locks_profit_and_target_wins():
    o = arr(100, 100, 104, 108, 109)
    h = arr(100, 104, 108, 110, 109)
    lo = arr(100, 99.5, 103, 106, 104)
    c = arr(100, 104, 108, 109, 105)
    t = bracket_trades(o, h, lo, c, arr(1, 1, 1, 1, 1), np.array([True, False, False, False, False]),
                       Exit(2.0, None, 3.0, 4), NO_COST)
    # highest high before bar 4 is 110 → trail 107; bar 4 opens 109, low 104 → exit at 107
    assert t["ret"][0] == pytest.approx(0.07)
    t = bracket_trades(o, h, lo, c, arr(1, 1, 1, 1, 1), np.array([True, False, False, False, False]),
                       Exit(2.0, 5.0, None, 4), NO_COST)
    assert t["ret"][0] == pytest.approx(0.05)


def _trades(rows):
    return pd.DataFrame(rows, columns=["coin", "t_in", "t_out", "ret", "stop_pct"])


def test_portfolio_sizes_by_risk_and_locks_after_two_losses():
    ts = lambda h: pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(hours=h)  # noqa: E731
    rows = [("A", ts(0), ts(1), -0.02, 0.02), ("B", ts(2), ts(3), -0.02, 0.02),
            ("C", ts(4), ts(5), 0.04, 0.02), ("D", ts(30), ts(31), 0.04, 0.02)]
    res = simulate_portfolio(_trades(rows), RiskRules(equity0=1000), NO_COST)
    assert list(res.trades["coin"]) == ["A", "B", "D"]  # C blocked by the 24h lock
    assert res.trades["pnl"].iloc[0] == pytest.approx(-6.0)  # 0.6% of 1000 at a 2% stop
    assert res.skipped == 1


def test_portfolio_respects_max_positions_and_cash():
    ts = lambda h: pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(hours=h)  # noqa: E731
    rows = [(c, ts(0), ts(10), 0.01, 0.001) for c in "ABCD"]
    res = simulate_portfolio(_trades(rows), RiskRules(equity0=100), NO_COST)
    assert len(res.trades) <= 3
    assert res.trades["notional"].sum() <= 100 + 1e-9


def test_deflated_sharpe_penalises_many_trials():
    few = deflated_sharpe(0.1, 1000, 0.0, 3.0, [0.0, 0.01])
    many = deflated_sharpe(0.1, 1000, 0.0, 3.0, list(np.random.default_rng(1).normal(0, 0.05, 500)))
    assert many < few
