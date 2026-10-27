"""Live technical analysis: report for one market, annotated with the historical study, compacted for prompts."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from mqmas.ta.signals import analyze

log = logging.getLogger(__name__)

_study: dict[str, Any] = {"loaded_at": 0.0, "data": None}


def load_study(root: Path) -> dict[str, Any] | None:
    """The TA study report (pattern → historical verdict), cached for an hour."""
    if time.time() - _study["loaded_at"] > 3600:
        path = root / "ta_report.json"
        try:
            _study["data"] = json.loads(path.read_text()) if path.exists() else None
        except (OSError, ValueError):
            log.warning("ta study unreadable")
            _study["data"] = None
        _study["loaded_at"] = time.time()
    data = _study["data"]
    return data if isinstance(data, dict) else None


def report(df: pd.DataFrame, tf: str, root: Path) -> dict[str, Any]:
    study = load_study(root)
    events = {k: v.get(tf) for k, v in (study or {}).get("events", {}).items() if v.get(tf)}
    return analyze(df, {"events": events} if events else None)


def compact(rep: dict[str, Any]) -> dict[str, Any]:
    """A few hundred tokens: what an analyst would say, plus how each recent pattern fared historically."""

    def pats(items: list[dict[str, Any]]) -> list[str]:
        out = []
        for p in items:
            h = p.get("historico")
            out.append(f"{p['padrao']} ({h['veredito']}, acerto {h['acerto']:.0%})" if h else p["padrao"])
        return out

    t, m, v, s = rep["tendencia"], rep["momento"], rep["volatilidade"], rep["estrutura"]
    return {"vies": t["vies"], "adx": t["adx"], "forca_tendencia": t["forca"], "estrutura_smc": s["tendencia_smc"],
            "rsi": m["rsi"], "macd_hist": m["macd_hist"], "estocastico": m["estocastico_k"],
            "atr_pct": v["atr_pct"], "squeeze_ttm": v["squeeze_ttm"], "bollinger": v["posicao_bollinger"],
            "acima_vwap": rep["volume"]["acima_vwap"], "obv_acima_media": rep["volume"]["obv_acima_media"],
            "padroes_alta": pats(rep["padroes_recentes"]["alta"]),
            "padroes_baixa": pats(rep["padroes_recentes"]["baixa"]),
            "zonas_proximas": [f"{z['tipo']} de {z['direcao']} {z['de']:.6g}–{z['ate']:.6g}" for z in s["zonas_proximas"]],
            "saldo_confluencia": rep["confluencia"]["saldo"]}


def agent_view(df: pd.DataFrame, tf: str, root: Path) -> dict[str, Any]:
    """What the agents receive: the compact analysis plus the marked price levels and a long plan."""
    from mqmas.ta.confluence import live_score
    from mqmas.ta.levels import compact_map, price_map

    sc = live_score(df, load_study(root), tf)
    return {"leitura": compact(report(df, tf, root)), "mapa_de_precos": compact_map(price_map(df)),
            "score_confluencia": {"score": sc["score"], "componentes": sc["componentes"],
                                  "penalidade": sc["penalidade"], "padroes_com_historico": sc["padroes_com_historico"]}}
