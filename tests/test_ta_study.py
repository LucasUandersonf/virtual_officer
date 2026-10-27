"""The pattern study must find an edge that exists and must not invent one that does not."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mqmas.research import ta_study
from mqmas.ta.indicators import all_indicators


def _write_market(root: Path, closes: np.ndarray, name: str = "TEST_USDT") -> None:
    n = closes.size
    ts = int(pd.Timestamp("2022-01-01", tz="UTC").timestamp() * 1000) + np.arange(n) * 3_600_000
    o = np.r_[closes[0], closes[:-1]]
    rows = np.c_[ts, o, np.maximum(o, closes) * 1.001, np.minimum(o, closes) * 0.999, closes,
                 np.full(n, 100.0), np.full(n, 50.0)]
    (root / "swing").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "swing" / f"{name}_1h_tb.npz", rows=rows)


def _make(tmp_path: Path, planted: bool) -> dict:
    rng = np.random.default_rng(42)
    n = 24 * 365 * 4  # 4 years of 1h bars, both study periods covered
    rets = rng.normal(0, 0.004, n)
    signal_bars = rng.choice(np.arange(24 * 60, n - 48, 4), size=900, replace=False)  # 4h-aligned bars
    signal_bars.sort()
    if planted:  # after each event the next 24h drift up by +2.4 % in total
        for b in signal_bars:
            rets[b + 4:b + 28] += 0.001
    closes = 100 * np.exp(np.cumsum(rets))
    _write_market(tmp_path, closes)
    # mark the event on the 4h bar whose close is the 1h bar right before the drift starts
    times = pd.Timestamp("2022-01-01", tz="UTC") + pd.to_timedelta(signal_bars + 3, unit="h")
    marks = set(times.floor("4h"))

    def events(df: pd.DataFrame):
        f = all_indicators(df)
        ev = pd.DataFrame({"controle_alta": [t in marks for t in df.index]}, index=df.index)
        return ev, f, None

    return ta_study.run(tmp_path, events_fn=events)


def test_positive_control_is_detected(tmp_path):
    rep = _make(tmp_path, planted=True)
    row = next(r for r in rep["rows"] if r["padrao"] == "controle_alta" and r["tf"] == "4h")
    assert row["veredito"] == "funciona", row


def test_negative_control_is_rejected(tmp_path):
    rep = _make(tmp_path, planted=False)
    row = next(r for r in rep["rows"] if r["padrao"] == "controle_alta" and r["tf"] == "4h")
    assert row["veredito"] == "sem efeito", row


def _confluence(tmp_path: Path, planted: bool) -> dict:
    from mqmas.research import confluence_study

    study = _make(tmp_path, planted)
    rng = np.random.default_rng(42)  # same draws as _make, to rebuild the event marks
    n = 24 * 365 * 4
    rng.normal(0, 0.004, n)
    bars = np.sort(rng.choice(np.arange(24 * 60, n - 48, 4), size=900, replace=False))
    marks = set((pd.Timestamp("2022-01-01", tz="UTC") + pd.to_timedelta(bars + 3, unit="h")).floor("4h"))

    def events(df: pd.DataFrame):
        f = all_indicators(df)
        return pd.DataFrame({"controle_alta": [t in marks for t in df.index]}, index=df.index), f, None

    return confluence_study.run(tmp_path, events_fn=events, study=study)


def test_confluence_score_validates_a_real_edge(tmp_path):
    rep = _confluence(tmp_path, planted=True)
    assert rep["tf"]["4h"]["validado"], rep["tf"]["4h"]


def test_confluence_score_rejects_noise(tmp_path):
    rep = _confluence(tmp_path, planted=False)
    assert not rep["tf"]["4h"]["validado"], rep["tf"]["4h"]


def test_pattern_portfolio_trades_the_planted_edge(tmp_path):
    from mqmas.research import pattern_portfolio

    study = _make(tmp_path, planted=True)
    rng = np.random.default_rng(42)
    n = 24 * 365 * 4
    rng.normal(0, 0.004, n)
    bars = np.sort(rng.choice(np.arange(24 * 60, n - 48, 4), size=900, replace=False))
    marks = set((pd.Timestamp("2022-01-01", tz="UTC") + pd.to_timedelta(bars + 3, unit="h")).floor("4h"))

    def events(df: pd.DataFrame):
        f = all_indicators(df)
        return pd.DataFrame({"controle_alta": [t in marks for t in df.index]}, index=df.index), f, None

    rep = pattern_portfolio.run(tmp_path, events_fn=events, study=study)
    row = next(r for r in rep["linhas"] if r["padrao"] == "controle_alta")
    acc = row["contas"]["A: US$120 a 0,6%"]
    assert acc["trades"] > 30 and acc["total_return"] > 0 and acc["profit_factor"] > 1.2, row
    assert pattern_portfolio.summarize(rep)[1].startswith(("APROVADO", "reprovado"))
