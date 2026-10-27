from __future__ import annotations

import numpy as np
import pandas as pd

from mqmas.ta.candles import candle_patterns
from mqmas.ta.signals import all_events, analyze
from mqmas.ta.structure import pivots, structure_events


def frame(rows: list[tuple[float, float, float, float]], vol: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=len(rows), freq="4h", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = vol
    return df


def downtrend(n: int = 8, start: float = 120.0) -> list[tuple[float, float, float, float]]:
    out, p = [], start
    for _ in range(n):
        out.append((p, p + 0.5, p - 2.5, p - 2.0))
        p -= 2.0
    return out


def test_hammer_and_engulfing_need_a_prior_downtrend():
    rows = downtrend()
    p = rows[-1][3]
    rows.append((p, p + 0.2, p - 3.0, p + 0.1))  # small body at the top, long lower wick
    pat = candle_patterns(frame(rows))
    assert pat["martelo"].iloc[-1]
    rows2 = downtrend()
    p = rows2[-1][3]
    rows2.append((p + 0.5, p + 0.8, p - 1.5, p - 1.0))  # red
    rows2.append((p - 1.2, p + 1.5, p - 1.3, p + 1.2))  # green body engulfs it
    assert candle_patterns(frame(rows2))["engolfo_alta"].iloc[-1]
    flat = [(100, 100.5, 99.5, 100.0)] * 8 + [(100, 100.2, 97.0, 100.1)]
    assert not candle_patterns(frame(flat))["martelo"].iloc[-1]  # no downtrend → not a hammer


def test_pivots_are_stamped_when_confirmed():
    rows = [(100 + k, 101 + k, 99 + k, 100.5 + k) for k in range(10)] + \
           [(110 - k, 111 - k, 109 - k, 109.5 - k) for k in range(10)]
    piv = [p for p in pivots(frame(rows), n=3) if p.kind == "H"]
    top = max(piv, key=lambda p: p.price)
    assert top.conf == top.idx + 3  # never known before 3 lower highs printed


def test_fair_value_gap_retest_and_break_of_structure():
    rng = np.random.default_rng(1)
    base = 100.0
    rows = []
    for _k in range(40):  # gentle range to settle ATR
        o = base + rng.normal(0, 0.3)
        c = base + rng.normal(0, 0.3)
        rows.append((o, max(o, c) + 0.3, min(o, c) - 0.3, c))
    rows += [(100, 100.4, 99.7, 100.2), (100.2, 104.0, 100.1, 103.8), (103.8, 106.0, 101.5, 105.5)]  # gap 100.4→101.5
    rows += [(105.5, 105.8, 101.2, 102.0)]  # back into the gap, closes above its bottom
    ev, _ = structure_events(frame(rows))
    assert ev["fvg_alta_reteste"].iloc[-1]
    assert ev["bos_alta"].any() or ev["choch_alta"].any()


def test_double_bottom_breaks_the_neckline():
    rows = []
    path = [110, 106, 102, 98, 94, 90, 94, 98, 102, 104, 102, 98, 94, 90.3, 94, 98, 102, 106]
    for a, b in zip(path[:-1], path[1:], strict=True):
        rows.append((a, max(a, b) + 0.3, min(a, b) - 0.3, b))
    rows = [(110, 110.3, 109.7, 110)] * 20 + rows
    ev, _ = structure_events(frame(rows), n=2)
    assert ev["fundo_duplo"].any()


def test_report_is_complete_and_json_friendly():
    rng = np.random.default_rng(5)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.004, "low": np.minimum(o, c) * 0.996, "close": c,
                       "volume": rng.gamma(2, 50, 400)}, index=pd.date_range("2025-01-01", periods=400, freq="4h", tz="UTC"))
    rep = analyze(df)
    assert set(rep) >= {"tendencia", "momento", "volatilidade", "volume", "estrutura", "padroes_recentes", "confluencia"}
    ev, _, _ = all_events(df)
    assert ev.dtypes.eq(bool).all() and len(ev.columns) > 60
