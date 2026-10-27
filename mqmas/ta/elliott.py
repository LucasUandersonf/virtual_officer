"""Elliott waves reduced to their objective rules, on confirmed 5-bar pivots (no look-ahead).

* ``elliott_onda5_alta`` — waves 1-2-3-4 of an up impulse are complete: wave 2 holds above the
  start of 1, wave 3 makes a new high and is longer than wave 1, wave 4 retraces 23.6–61.8 % of
  wave 3 and does not overlap wave 1. Fires when the wave-4 low is confirmed (setup for wave 5).
* ``elliott_abc_alta`` — a zigzag A-B-C down ends: B retraces 38.2–88.6 % of A, C measures
  0.8–1.272 × A. Fires when the C low is confirmed.
* ``*_baixa`` — the mirror images.

Wave counting is subjective; only these testable rules are used, and the TA study measures them
like any other signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mqmas.ta.structure import Pivot, pivots

NAMES = ("elliott_onda5_alta", "elliott_onda5_baixa", "elliott_abc_alta", "elliott_abc_baixa")
BULLISH_EW = ("elliott_onda5_alta", "elliott_abc_alta")
BEARISH_EW = ("elliott_onda5_baixa", "elliott_abc_baixa")


def _alternating(piv: list[Pivot]) -> list[Pivot]:
    seq: list[Pivot] = []
    for p in sorted(piv, key=lambda q: q.idx):
        if seq and seq[-1].kind == p.kind:
            if (p.kind == "H" and p.price > seq[-1].price) or (p.kind == "L" and p.price < seq[-1].price):
                seq[-1] = p
            continue
        seq.append(p)
    return seq


def elliott_events(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    size = len(df)
    ev = {k: np.zeros(size, dtype=bool) for k in NAMES}
    seq = _alternating(pivots(df, n))
    for k in range(len(seq)):
        last = seq[k]
        conf = max(p.conf for p in seq[max(k - 4, 0):k + 1])
        if conf >= size:
            continue
        if k >= 4:
            p0, p1, p2, p3, p4 = (q.price for q in seq[k - 4:k + 1])
            s = 1.0 if last.kind == "L" else -1.0  # up impulse ends its wave 4 on a low
            w1, w3, w4 = s * (p1 - p0), s * (p3 - p2), s * (p3 - p4)
            if (w1 > 0 and s * (p2 - p0) > 0 and s * (p3 - p1) > 0 and w3 > w1 and s * (p4 - p1) > 0
                    and 0.236 <= w4 / w3 <= 0.618):
                ev["elliott_onda5_alta" if s > 0 else "elliott_onda5_baixa"][conf] = True
        if k >= 3:
            o, a, b, c = (q.price for q in seq[k - 3:k + 1])
            s = 1.0 if last.kind == "L" else -1.0  # a correction down ends on a low
            wa, wb, wc = s * (o - a), s * (b - a), s * (b - c)
            if wa > 0 and 0.382 <= wb / wa <= 0.886 and 0.8 <= wc / wa <= 1.272:
                ev["elliott_abc_alta" if s > 0 else "elliott_abc_baixa"][conf] = True
    return pd.DataFrame(ev, index=df.index)
