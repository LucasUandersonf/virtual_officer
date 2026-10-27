from __future__ import annotations

import numpy as np
import pandas as pd

from mqmas.research import tcc
from mqmas.research.engine import Costs


def _daily(n: int = 900, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.03, n)))
    o = np.r_[c[0], c[:-1]]
    idx = pd.date_range("2018-06-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.01, "low": np.minimum(o, c) * 0.99,
                         "close": c, "volume": 1.0}, index=idx)


def test_models_trade_long_only_and_report_the_thesis_tables():
    df = _daily()
    for model, (buy, sell) in tcc.signals(df).items():
        ops, eq = tcc.backtest(df, buy, sell, Costs())
        assert len(ops), model
        assert (ops["tipo"] == "Compra").all() and (ops["saida"] > ops["entrada"]).all()
        assert abs(ops["saldo"].iloc[-1] - (tcc.CAPITAL + ops["resultado"].sum())) < 1e-6
        m = tcc.averages(ops, eq, df)
        assert set(m) >= {"taxa_acerto", "media_operacional", "media_ganho", "media_perda", "rentabilidade_media_pct",
                          "buy_hold_pct", "drawdown_max_pct"}


def test_costs_reduce_every_operation():
    df = _daily()
    buy, sell = tcc.signals(df)["Cruzamento do MACD"]
    free, _ = tcc.backtest(df, buy, sell, Costs(fee=0.0, slippage=0.0))
    paid, _ = tcc.backtest(df, buy, sell, Costs())
    assert len(free) == len(paid) and (paid["retorno_pct"] < free["retorno_pct"]).all()


def test_signal_fills_at_the_next_open():
    df = _daily()
    buy, sell = tcc.signals(df)["FFFD"]
    ops, _ = tcc.backtest(df, buy, sell, Costs(fee=0.0, slippage=0.0))
    first = ops.iloc[0]
    day_before = df.index[df.index.get_loc(first["entrada"]) - 1]
    assert bool(buy.loc[day_before])
