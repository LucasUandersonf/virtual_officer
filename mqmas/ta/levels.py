"""Price map: every level a technical analyst would mark on the chart, merged into confluence zones,
plus a long trade plan (entry, stop, targets, risk:reward) built from those zones.

Sources of levels (all from closed candles):

* swing highs/lows (5-bar confirmed pivots) and liquidity pools (equal highs/lows);
* Fibonacci retracements (38.2/50/61.8/65/78.6 %) and extensions (127.2/161.8 %) of the last leg;
* Bollinger bands and middle line, EMA 20/50/200, session VWAP;
* volume profile of the last 120 bars (POC, value-area high/low);
* Ichimoku cloud edges; open order blocks and fair value gaps.

Levels closer than 0.3 ATR are merged into one zone; a zone's *strength* is how many independent
tools put a level there. The plan only proposes a long when the first target pays at least 1.5R.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mqmas.ta.indicators import all_indicators, volume_profile
from mqmas.ta.structure import pivots, structure_events

MERGE_ATR = 0.3
MIN_RR = 1.5


def _levels(df: pd.DataFrame, f: pd.DataFrame) -> list[dict[str, Any]]:
    last = f.iloc[-1]
    c = float(df["close"].iloc[-1])
    out: list[dict[str, Any]] = []

    def add(price: float, source: str) -> None:
        if np.isfinite(price) and price > 0:
            out.append({"price": float(price), "source": source})

    piv = [p for p in pivots(df, 5) if p.conf < len(df)]
    highs = [p for p in piv if p.kind == "H"][-12:]
    lows = [p for p in piv if p.kind == "L"][-12:]
    for p in highs:
        add(p.price, "topo anterior")
    for p in lows:
        add(p.price, "fundo anterior")
    a = float(last["atr"])
    for group, name in ((highs, "liquidez acima (topos iguais)"), (lows, "liquidez abaixo (fundos iguais)")):
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if abs(group[i].price - group[j].price) <= 0.2 * a:
                    add((group[i].price + group[j].price) / 2, name)
    if highs and lows:
        h, lo = highs[-1], lows[-1]
        if h.idx > lo.idx:  # last leg was up: retracements below, extensions above
            leg = h.price - lo.price
            for r in (0.382, 0.5, 0.618, 0.65, 0.786):
                add(h.price - r * leg, f"Fibo {r * 100:.1f}%")
            for e in (1.272, 1.618):
                add(lo.price + e * leg, f"extensão Fibo {e * 100:.1f}%")
        else:  # last leg was down: retracements above, extensions below
            leg = h.price - lo.price
            for r in (0.382, 0.5, 0.618, 0.65, 0.786):
                add(lo.price + r * leg, f"Fibo {r * 100:.1f}%")
            for e in (1.272, 1.618):
                add(h.price - e * leg, f"extensão Fibo {e * 100:.1f}%")
    for col, name in (("bb_up", "Bollinger superior"), ("bb_mid", "média de 20 (Bollinger)"), ("bb_lo", "Bollinger inferior"),
                      ("ema20", "EMA20"), ("ema50", "EMA50"), ("ema200", "EMA200"), ("vwap", "VWAP")):
        add(float(last[col]), name)
    if np.isfinite(last["span_a"]) and np.isfinite(last["span_b"]):
        add(max(last["span_a"], last["span_b"]), "topo da nuvem Ichimoku")
        add(min(last["span_a"], last["span_b"]), "base da nuvem Ichimoku")
    vp = volume_profile(df.iloc[-120:])
    add(vp["poc"], "POC (maior volume)")
    add(vp["vah"], "topo da value area")
    add(vp["val"], "base da value area")
    _, state = structure_events(df)
    for z in state.zones[-8:]:
        add((z["de"] + z["ate"]) / 2, f"{z['tipo']} de {z['direcao']}")
    return [lv for lv in out if 0.5 * c <= lv["price"] <= 2.0 * c]


def _zones(levels: list[dict[str, Any]], a: float) -> list[dict[str, Any]]:
    zones: list[dict[str, Any]] = []
    for lv in sorted(levels, key=lambda x: x["price"]):
        if zones and lv["price"] - zones[-1]["ate"] <= MERGE_ATR * a:
            z = zones[-1]
            z["ate"] = lv["price"]
            z["fontes"].append(lv["source"])
        else:
            zones.append({"de": lv["price"], "ate": lv["price"], "fontes": [lv["source"]]})
    for z in zones:
        z["preco"] = (z["de"] + z["ate"]) / 2
        z["forca"] = len(set(z["fontes"]))
        z["fontes"] = sorted(set(z["fontes"]))
    return zones


def price_map(df: pd.DataFrame) -> dict[str, Any]:
    """Map of the market at the last closed bar of ``df``: support/resistance zones and a long plan."""
    f = all_indicators(df)
    c = float(df["close"].iloc[-1])
    a = float(f["atr"].iloc[-1])
    zones = _zones(_levels(df, f), a)
    supports = sorted([z for z in zones if z["ate"] < c - 0.1 * a], key=lambda z: -z["preco"])[:6]
    resistances = sorted([z for z in zones if z["de"] > c + 0.1 * a], key=lambda z: z["preco"])[:6]
    plan: dict[str, Any] = {"valido": False}
    strong_sup = [z for z in supports if z["forca"] >= 2 and c - z["preco"] <= 3 * a]
    if strong_sup and resistances:
        entry_zone = strong_sup[0]
        entry = entry_zone["ate"]
        stop = entry_zone["de"] - 0.5 * a
        risk = entry - stop
        targets = [z["preco"] for z in resistances[:3]]
        rr = [(t - entry) / risk for t in targets] if risk > 0 else []
        plan = {
            "valido": bool(rr and rr[0] >= MIN_RR),
            "entrada": round(entry, 8), "zona_entrada": [round(entry_zone["de"], 8), round(entry_zone["ate"], 8)],
            "fontes_entrada": entry_zone["fontes"], "stop": round(stop, 8),
            "risco_pct": round(risk / entry * 100, 2) if entry else None,
            "alvos": [round(t, 8) for t in targets], "risco_retorno": [round(x, 2) for x in rr],
            "invalidacao": f"fechamento abaixo de {stop:.6g} anula a zona de compra",
            "motivo": (f"compra limitada na zona de confluência ({', '.join(entry_zone['fontes'])}); "
                       f"alvo 1 na resistência mais próxima") if rr else "sem alvo acima",
        }
        if rr and rr[0] < MIN_RR:
            plan["motivo"] = f"risco:retorno do 1º alvo {rr[0]:.2f} < {MIN_RR}: esperar preço melhor"

    def fmt(z: dict[str, Any]) -> dict[str, Any]:
        return {"preco": round(z["preco"], 8), "de": round(z["de"], 8), "ate": round(z["ate"], 8), "forca": z["forca"],
                "fontes": z["fontes"], "distancia_pct": round((z["preco"] / c - 1) * 100, 2)}

    return {"preco": c, "atr": a, "atr_pct": round(a / c * 100, 2),
            "suportes": [fmt(z) for z in supports], "resistencias": [fmt(z) for z in resistances], "plano_compra": plan}


def compact_map(m: dict[str, Any]) -> dict[str, Any]:
    def z(items: list[dict[str, Any]]) -> list[str]:
        return [f"{x['preco']:.6g} ({x['distancia_pct']:+.1f}%, força {x['forca']}: {', '.join(x['fontes'][:3])})"
                for x in items[:4]]

    p = m["plano_compra"]
    plan = (f"entrada {p['entrada']:.6g}, stop {p['stop']:.6g} ({p['risco_pct']}%), alvos "
            f"{', '.join(f'{t:.6g}' for t in p['alvos'])} (R:R {p['risco_retorno']})") if p.get("entrada") else "sem plano"
    return {"suportes": z(m["suportes"]), "resistencias": z(m["resistencias"]), "plano_compra": plan,
            "plano_valido": p.get("valido", False)}
