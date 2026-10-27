"""Market structure (SMC vocabulary) and classic chart patterns, built on *confirmed* swing pivots.

A swing high at bar j needs ``n`` lower highs on each side, so it is only known at bar j + n; every
event below is stamped at the bar where it could actually have been seen live.

Events (boolean columns, True on the bar the event completes):

* ``bos_alta`` / ``bos_baixa`` — break of structure in the direction of the current trend;
* ``choch_alta`` / ``choch_baixa`` — change of character (first break against the trend);
* ``ob_alta_reteste`` / ``ob_baixa_reteste`` — first return into the order block (last opposite
  candle before the impulse that broke structure);
* ``fvg_alta_reteste`` / ``fvg_baixa_reteste`` — first return into a fair value gap;
* ``varredura_fundo`` / ``varredura_topo`` — liquidity sweep: wick through the last swing low/high and
  close back inside (stop hunt);
* chart patterns: ``fundo_duplo``, ``topo_duplo``, ``fundo_triplo``, ``topo_triplo``, ``oco_invertido``,
  ``oco``, ``triangulo_ascendente``, ``triangulo_descendente``, ``triangulo_simetrico_alta``,
  ``triangulo_simetrico_baixa``, ``alargamento``, ``bandeira_alta``, ``bandeira_baixa``,
  ``xicara_alca`` — stamped on the breakout/confirmation close.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from mqmas.ta.indicators import atr

OB_LIFETIME = 60
FVG_LIFETIME = 50
PATTERN_WINDOW = 40  # a breakout must happen within this many bars of the last pivot of the pattern


@dataclass
class Pivot:
    idx: int  # bar of the extreme
    conf: int  # bar where it became known
    price: float
    kind: str  # "H" or "L"


@dataclass
class StructureState:
    trend: int = 0
    last_high: Pivot | None = None
    last_low: Pivot | None = None
    zones: list[dict[str, Any]] = field(default_factory=list)


def pivots(df: pd.DataFrame, n: int = 3) -> list[Pivot]:
    h, lo = df["high"], df["low"]
    # extreme of the 2n+1 window, and strictly beyond the previous n bars so a flat top/bottom counts once
    hi_mask = (h == h.rolling(2 * n + 1, center=True).max()) & (h > h.shift(1).rolling(n, min_periods=1).max())
    lo_mask = (lo == lo.rolling(2 * n + 1, center=True).min()) & (lo < lo.shift(1).rolling(n, min_periods=1).min())
    out: list[Pivot] = []
    hv, lv = h.to_numpy(), lo.to_numpy()
    for j in np.flatnonzero(hi_mask.to_numpy()):
        out.append(Pivot(int(j), int(j + n), float(hv[j]), "H"))
    for j in np.flatnonzero(lo_mask.to_numpy()):
        out.append(Pivot(int(j), int(j + n), float(lv[j]), "L"))
    out.sort(key=lambda p: (p.conf, p.idx))
    return out


def structure_events(df: pd.DataFrame, n: int = 3) -> tuple[pd.DataFrame, StructureState]:
    o, h, lo, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    a = atr(df).to_numpy()
    size = len(df)
    names = ["bos_alta", "bos_baixa", "choch_alta", "choch_baixa", "ob_alta_reteste", "ob_baixa_reteste",
             "fvg_alta_reteste", "fvg_baixa_reteste", "varredura_fundo", "varredura_topo"] + PATTERNS
    ev = {k: np.zeros(size, dtype=bool) for k in names}
    piv = pivots(df, n)
    st = StructureState()
    highs: list[Pivot] = []
    lows: list[Pivot] = []
    p_ptr = 0
    high_broken = low_broken = True
    obs: list[list[float]] = []  # [bottom, top, dir, born]
    fvgs: list[list[float]] = []
    for i in range(size):
        while p_ptr < len(piv) and piv[p_ptr].conf <= i:
            p = piv[p_ptr]
            p_ptr += 1
            if p.kind == "H":
                highs.append(p)
                st.last_high, high_broken = p, False
            else:
                lows.append(p)
                st.last_low, low_broken = p, False
        if np.isnan(a[i]):
            continue
        # liquidity sweeps (stop hunts) of the last confirmed swing
        if st.last_low is not None and lo[i] < st.last_low.price < c[i]:
            ev["varredura_fundo"][i] = True
        if st.last_high is not None and h[i] > st.last_high.price > c[i]:
            ev["varredura_topo"][i] = True
        # breaks of structure / change of character, with the order block of the impulse
        if st.last_high is not None and not high_broken and c[i] > st.last_high.price:
            ev["bos_alta" if st.trend >= 0 else "choch_alta"][i] = True
            st.trend, high_broken = 1, True
            start = max(st.last_low.idx if st.last_low is not None else 0, i - 50)
            reds = [k for k in range(start, i) if c[k] < o[k]]
            if reds:
                k = reds[-1]
                obs.append([lo[k], h[k], 1, i])
        if st.last_low is not None and not low_broken and c[i] < st.last_low.price:
            ev["bos_baixa" if st.trend <= 0 else "choch_baixa"][i] = True
            st.trend, low_broken = -1, True
            start = max(st.last_high.idx if st.last_high is not None else 0, i - 50)
            greens = [k for k in range(start, i) if c[k] > o[k]]
            if greens:
                k = greens[-1]
                obs.append([lo[k], h[k], -1, i])
        # fair value gaps (3-candle imbalance of at least 0.25 ATR)
        if i >= 2:
            if lo[i] - h[i - 2] > 0.25 * a[i]:
                fvgs.append([h[i - 2], lo[i], 1, i])
            if lo[i - 2] - h[i] > 0.25 * a[i]:
                fvgs.append([h[i], lo[i - 2], -1, i])
        # first return into a zone: bullish zone touched from above and held; bearish from below and rejected
        for zones, life, up_name, dn_name in ((obs, OB_LIFETIME, "ob_alta_reteste", "ob_baixa_reteste"),
                                              (fvgs, FVG_LIFETIME, "fvg_alta_reteste", "fvg_baixa_reteste")):
            keep = []
            for z in zones:
                bottom, top, d, born = z
                if i - born > life:
                    continue
                if i > born and d == 1 and lo[i] <= top and c[i] > bottom:
                    ev[up_name][i] = True
                    continue
                if i > born and d == -1 and h[i] >= bottom and c[i] < top:
                    ev[dn_name][i] = True
                    continue
                if (d == 1 and c[i] < bottom) or (d == -1 and c[i] > top):
                    continue  # invalidated
                keep.append(z)
            zones[:] = keep
        _chart_patterns(ev, i, c, h, lo, a, highs[-12:], lows[-12:])
    st.zones = [{"tipo": "order block" if z in obs else "fvg", "direcao": "alta" if z[2] == 1 else "baixa",
                 "de": float(z[0]), "ate": float(z[1])} for z in obs + fvgs]
    return pd.DataFrame(ev, index=df.index), st


PATTERNS = ["fundo_duplo", "topo_duplo", "fundo_triplo", "topo_triplo", "oco_invertido", "oco",
            "triangulo_ascendente", "triangulo_descendente", "triangulo_simetrico_alta", "triangulo_simetrico_baixa",
            "alargamento", "bandeira_alta", "bandeira_baixa", "xicara_alca"]


def _near(a: float, b: float, tol: float) -> bool:
    return bool(abs(a - b) <= tol)


def _chart_patterns(ev: dict[str, np.ndarray], i: int, c: np.ndarray, h: np.ndarray, lo: np.ndarray, a: np.ndarray,
                    highs: list[Pivot], lows: list[Pivot]) -> None:
    tol = 0.6 * a[i]
    prev_c = c[i - 1] if i else c[i]

    def crossed_up(level: float) -> bool:
        return bool(prev_c <= level < c[i])

    def crossed_dn(level: float) -> bool:
        return bool(prev_c >= level > c[i])

    def fresh(p: Pivot) -> bool:
        return i - p.conf <= PATTERN_WINDOW

    if len(lows) >= 2 and highs:
        l1, l2 = lows[-2], lows[-1]
        mids = [hp for hp in highs if l1.idx < hp.idx < l2.idx]
        if mids and fresh(l2) and _near(l1.price, l2.price, tol) and l2.idx - l1.idx >= 5:
            neck = max(hp.price for hp in mids)
            if crossed_up(neck):
                ev["fundo_duplo"][i] = True
        if len(lows) >= 3:
            l0 = lows[-3]
            mids3 = [hp for hp in highs if l0.idx < hp.idx < l2.idx]
            if mids3 and fresh(l2) and _near(l0.price, l1.price, tol) and _near(l1.price, l2.price, tol):
                if crossed_up(max(hp.price for hp in mids3)):
                    ev["fundo_triplo"][i] = True
            # inverse head & shoulders: middle low clearly lowest, shoulders level
            if (mids3 and fresh(l2) and l1.price < l0.price - tol and l1.price < l2.price - tol
                    and _near(l0.price, l2.price, 1.5 * tol)):
                if crossed_up(max(hp.price for hp in mids3)):
                    ev["oco_invertido"][i] = True
    if len(highs) >= 2 and lows:
        h1, h2 = highs[-2], highs[-1]
        mids = [lp for lp in lows if h1.idx < lp.idx < h2.idx]
        if mids and fresh(h2) and _near(h1.price, h2.price, tol) and h2.idx - h1.idx >= 5:
            if crossed_dn(min(lp.price for lp in mids)):
                ev["topo_duplo"][i] = True
        if len(highs) >= 3:
            h0 = highs[-3]
            mids3 = [lp for lp in lows if h0.idx < lp.idx < h2.idx]
            if mids3 and fresh(h2) and _near(h0.price, h1.price, tol) and _near(h1.price, h2.price, tol):
                if crossed_dn(min(lp.price for lp in mids3)):
                    ev["topo_triplo"][i] = True
            if (mids3 and fresh(h2) and h1.price > h0.price + tol and h1.price > h2.price + tol
                    and _near(h0.price, h2.price, 1.5 * tol)):
                if crossed_dn(min(lp.price for lp in mids3)):
                    ev["oco"][i] = True
    # triangles and broadening formations from the last two highs and lows
    if len(highs) >= 2 and len(lows) >= 2 and fresh(highs[-1]) and fresh(lows[-1]):
        h1, h2, l1, l2 = highs[-2], highs[-1], lows[-2], lows[-1]
        flat_top, flat_bot = _near(h1.price, h2.price, tol), _near(l1.price, l2.price, tol)
        rising_lows, falling_highs = l2.price > l1.price + tol, h2.price < h1.price - tol
        if flat_top and rising_lows and crossed_up(max(h1.price, h2.price)):
            ev["triangulo_ascendente"][i] = True
        if flat_bot and falling_highs and crossed_dn(min(l1.price, l2.price)):
            ev["triangulo_descendente"][i] = True
        if falling_highs and rising_lows:
            if crossed_up(h2.price):
                ev["triangulo_simetrico_alta"][i] = True
            if crossed_dn(l2.price):
                ev["triangulo_simetrico_baixa"][i] = True
        if h2.price > h1.price + tol and l2.price < l1.price - tol and (crossed_up(h2.price) or crossed_dn(l2.price)):
            ev["alargamento"][i] = True
    # flags / pennants: strong pole, tight counter-trend pause, breakout in the pole's direction
    if i >= 30 and not np.isnan(a[i]):
        for pause in (5, 8, 12):
            p0 = i - pause
            pole = c[p0] - c[p0 - 10]
            box_h, box_l = h[p0:i].max(), lo[p0:i].min()
            if box_h - box_l > 0.5 * abs(pole) or abs(pole) < 4 * a[p0]:
                continue
            if pole > 0 and c[i] > box_h and prev_c <= box_h:
                ev["bandeira_alta"][i] = True
            if pole < 0 and c[i] < box_l and prev_c >= box_l:
                ev["bandeira_baixa"][i] = True
    # cup & handle: rim, rounded bottom (12–50 % deep), second rim at the same level, shallow handle, breakout
    if len(highs) >= 2 and lows:
        rim1, rim2 = highs[-2], highs[-1]
        cup = [lp for lp in lows if rim1.idx < lp.idx < rim2.idx]
        if cup and rim2.idx - rim1.idx >= 15 and abs(rim2.price / rim1.price - 1) <= 0.03 and fresh(rim2):
            bottom = min(lp.price for lp in cup)
            depth = (rim1.price - bottom) / rim1.price
            handle = lo[rim2.idx:i].min() if i > rim2.idx else rim2.price
            if 0.12 <= depth <= 0.5 and (rim2.price - handle) <= (rim2.price - bottom) / 3 \
                    and crossed_up(max(rim1.price, rim2.price)):
                ev["xicara_alca"][i] = True


BULLISH_STRUCT = ("bos_alta", "choch_alta", "ob_alta_reteste", "fvg_alta_reteste", "varredura_fundo", "fundo_duplo",
                  "fundo_triplo", "oco_invertido", "triangulo_ascendente", "triangulo_simetrico_alta", "bandeira_alta",
                  "xicara_alca")
BEARISH_STRUCT = ("bos_baixa", "choch_baixa", "ob_baixa_reteste", "fvg_baixa_reteste", "varredura_topo", "topo_duplo",
                  "topo_triplo", "oco", "triangulo_descendente", "triangulo_simetrico_baixa", "bandeira_baixa")
