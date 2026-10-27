from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from mqmas.research import crosscheck, livro
from mqmas.research.engine import Costs


def test_our_engine_matches_backtesting_py_on_the_same_rules():
    rng = np.random.default_rng(3)
    n = 1500
    c = 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.035, n)))
    o = np.r_[c[0], c[:-1]]
    idx = pd.date_range("2021-10-01", periods=n, freq="1D", tz="UTC")
    d = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.02, "low": np.minimum(o, c) * 0.98, "close": c,
                      "volume": 1.0}, index=idx)
    for setup in crosscheck.SETUPS:
        df = d if setup["tf"] == "1d" else livro.resample_3d(d, idx[-1])
        arr = crosscheck.prepare(df, setup)
        mine = crosscheck.ours(df, arr, livro.HOLD[setup["tf"]], Costs())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lib, _ = crosscheck.library(df, arr, livro.HOLD[setup["tf"]], Costs())
        assert len(mine) == len(lib), setup["nome"]
        if len(mine):
            assert np.sign(mine["ret"].to_numpy()).tolist() == np.sign(lib["ret"].to_numpy()).tolist()
            assert abs(mine["ret"].mean() - lib["ret"].mean()) < 0.003, setup["nome"]
