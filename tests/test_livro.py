from __future__ import annotations

import numpy as np
import pandas as pd

from mqmas.research import livro
from mqmas.research.engine import Costs

FREE = Costs(fee=0.0, slippage=0.0)


def _bars(closes: list[float], spread: float = 0.5) -> tuple[np.ndarray, ...]:
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    return o, np.maximum(o, c) + spread, np.minimum(o, c) - spread, c


def test_initial_stop_is_never_glued_nor_absurd():
    o, h, lo, c = _bars([100] * 3 + [90] * 5)
    atr = np.full(len(c), 2.0)
    ent = np.zeros(len(c), dtype=bool)
    ent[1] = True
    # technical stop 0.5 below → widened to 1.5 ATR = 3; technical stop 20 below → capped at 3 ATR = 6
    for tech, risk in ((99.5, 3.0), (80.0, 6.0)):
        t = livro.managed_trades(o, h, lo, c, atr, ent, np.full(len(c), tech), None, np.zeros(len(c), bool), FREE, 10)
        assert abs(t["stop_pct"][0] - risk / 100) < 1e-9 and t["reason"][0] == 1


def test_breakeven_after_1r_then_trailing_locks_gains():
    path = [100, 100, 104, 107, 110, 113, 116, 108, 100, 95]
    o, h, lo, c = _bars(path)
    atr = np.full(len(c), 2.0)
    ent = np.zeros(len(c), dtype=bool)
    ent[0] = True
    t = livro.managed_trades(o, h, lo, c, atr, ent, np.full(len(c), np.nan), None, np.zeros(len(c), bool), FREE, 20)
    # risk 3; the high runs to ~116.5 (> +2R), trailing stop = max(low of last 3, peak − 5) ≈ 111.5 → exits well in profit
    assert t["reason"][0] == 1 and t["ret"][0] > 0.10


def test_breakeven_turns_a_reversal_into_a_scratch_not_a_loss():
    path = [100, 100, 101, 103.5, 102, 99, 96, 93]
    o, h, lo, c = _bars(path)
    atr = np.full(len(c), 2.0)
    ent = np.zeros(len(c), dtype=bool)
    ent[0] = True
    t = livro.managed_trades(o, h, lo, c, atr, ent, np.full(len(c), np.nan), None, np.zeros(len(c), bool), Costs(), 20)
    assert t["reason"][0] == 1 and abs(t["ret"][0]) < 0.002  # stopped at entry + costs instead of −1R


def test_measured_target_and_exit_pattern():
    o, h, lo, c = _bars([100, 100, 102, 104, 106, 108, 110])
    atr = np.full(len(c), 2.0)
    ent = np.zeros(len(c), dtype=bool)
    ent[0] = True
    t = livro.managed_trades(o, h, lo, c, atr, ent, np.full(len(c), np.nan), np.full(len(c), 105.0),
                             np.zeros(len(c), bool), FREE, 20)
    assert t["reason"][0] == 2 and abs(t["ret"][0] - 0.05) < 1e-9
    ex = np.zeros(len(c), dtype=bool)
    ex[2] = True
    t = livro.managed_trades(o, h, lo, c, atr, ent, np.full(len(c), np.nan), None, ex, FREE, 20)
    assert t["reason"][0] == 3 and t["exit"][0] == 3


def _frame(path: np.ndarray) -> pd.DataFrame:
    c = pd.Series(path)
    df = pd.DataFrame({"open": c.shift(1).fillna(c[0]), "high": c + 0.3, "low": c - 0.3, "close": c, "volume": 1.0})
    df.index = pd.date_range("2024-01-01", periods=len(df), freq="1D", tz="UTC")
    return df


def test_rounding_bottom_and_falling_wedge_are_detected():
    x = np.linspace(-1, 1, 50)
    bowl = 100 - 20 * (1 - x ** 2)  # U from 100 down to 80 and back
    path = np.r_[np.full(30, 100.0), np.linspace(100, 100, 12), bowl, np.linspace(100, 104, 4), np.full(5, 104.0)]
    ev, stops, targets = livro.detect(_frame(path))
    assert ev["fundo_arredondado"].any()
    legs = [(130, 110), (110, 122), (122, 106), (106, 113), (113, 103), (103, 118), (118, 118)]
    wedge = np.concatenate([np.linspace(a, b, 9)[:-1] for a, b in legs])
    ev2, _, _ = livro.detect(_frame(np.r_[np.full(30, 130.0), wedge]))
    assert ev2["cunha_descendente"].any()


def test_catalog_has_every_book_rule_and_candidates_are_unique():
    ids = {s["id"] for s in livro.CATALOG}
    assert len(ids) == 16 and len(livro.LONGS) == 11 and len(livro.EXITS) == 5
    keys = [c["key"] for c in livro.candidates()]
    assert len(keys) == len(set(keys)) == 300  # 2 timeframes × ±macro × (18 singles + 55 pairs + 2 k-of-n)
