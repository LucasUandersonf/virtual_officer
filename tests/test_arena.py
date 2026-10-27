from __future__ import annotations

import numpy as np
import pandas as pd

from mqmas.research import arena, livro
from mqmas.research.engine import Costs


def test_fast_core_matches_the_library_engine():
    rng = np.random.default_rng(0)
    for _ in range(10):
        n = 400
        c = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
        o = np.r_[c[0], c[:-1]]
        h, lo, atr = np.maximum(o, c) * 1.01, np.minimum(o, c) * 0.99, np.full(n, 2.0)
        ent, ex = rng.random(n) < 0.05, rng.random(n) < 0.02
        st = np.where(rng.random(n) < 0.5, c * 0.97, np.nan)
        t = livro.managed_trades(o, h, lo, c, atr, ent, st, None, ex, Costs(), 20)
        seq, busy = [], -1
        for e, x, r in zip(t["entry"], t["exit"], t["ret"], strict=True):
            if e > busy:
                seq.append(r)
                busy = x
        fast = arena.manage(o, h, lo, c, atr, ent, st, 0.0, ex, Costs(), 20)
        assert np.allclose(np.array(seq), fast[:, 2])


def test_fixed_target_raises_the_hit_rate():
    rng = np.random.default_rng(1)
    n = 5000
    c = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    o = np.r_[c[0], c[:-1]]
    h, lo, atr = np.maximum(o, c) * 1.004, np.minimum(o, c) * 0.996, np.full(n, 1.0)
    ent = rng.random(n) < 0.05
    none = arena.manage(o, h, lo, c, atr, ent, np.full(n, np.nan), 0.0, np.zeros(n, bool), Costs(), 48)
    one_r = arena.manage(o, h, lo, c, atr, ent, np.full(n, np.nan), 1.0, np.zeros(n, bool), Costs(), 48)
    assert (one_r[:, 2] > 0).mean() > (none[:, 2] > 0).mean()


def test_resample_and_gate_never_look_ahead():
    idx = pd.date_range("2024-01-01", periods=24 * 10, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 1.0}, index=idx)
    two = arena.resample(df, "2h")
    assert len(two) == 120 and two["high"].iloc[0] == 2.0
    gate = pd.Series([False, True] + [False] * 8, index=pd.date_range("2024-01-01", periods=10, freq="1D", tz="UTC"))
    g = arena.gate_for(idx, "1h", gate)
    # the True of Jan 2 is usable only from candles closing on Jan 3
    assert not g[idx < pd.Timestamp("2024-01-02 23:00", tz="UTC")].any()
    assert g[(idx >= pd.Timestamp("2024-01-02 23:00", tz="UTC")) & (idx < pd.Timestamp("2024-01-03 23:00", tz="UTC"))].all()


def test_candidate_lists():
    assert len(arena.stage1("1h")) == len(arena.SIGNALS) * 2 * len(arena.TARGETS) == 120
    assert len(arena.stage2("1d", list(arena.SIGNALS[:6]))) == 2 * len(arena.TARGETS) * (15 + 2)
