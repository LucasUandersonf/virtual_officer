"""Pattern library: every pattern the system detects, its mathematical definition, and its measured record.

Three SQLite tables (created by ``mqmas.persistence.db``):

``ta_patterns``        one row per pattern: category, direction, rule, geometric parameters (JSON)
``ta_pattern_stats``   one row per pattern × timeframe: occurrences, hit rate (raw and net of the
                       coin's drift), excess return, trade PF, verdict — refreshed from the TA study
``ta_signals``         every live scan: symbol, timeframe, bar, confluence score and components,
                       active patterns, price map, proposed order and decision — the audit trail
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from mqmas.ta.signals import all_events, category, is_bear, is_bull

FIB_RETRACEMENTS = (0.382, 0.5, 0.618, 0.65, 0.786)
FIB_EXTENSIONS = (1.272, 1.618)

PARAMS: dict[str, dict[str, Any]] = {
    "gartley": {"AB/XA": [0.58, 0.66], "BC/AB": [0.382, 0.886], "AD/XA": [0.74, 0.82], "filtro": "D fora da Bollinger"},
    "bat": {"AB/XA": [0.382, 0.5], "AD/XA": [0.86, 0.91]},
    "butterfly": {"AB/XA": [0.74, 0.82], "AD/XA": [1.2, 1.35], "filtro": "D abaixo da Bollinger inferior"},
    "crab": {"AB/XA": [0.382, 0.618], "AD/XA": [1.55, 1.7]},
    "abcd": {"BC/AB": [0.618, 0.786], "CD/AB": [0.9, 1.1]},
    "elliott_onda5": {"onda2": "não passa o início da onda 1", "onda3": "> onda 1 e novo extremo",
                      "onda4/onda3": [0.236, 0.618], "onda4": "não sobrepõe a onda 1", "pivôs": 5},
    "elliott_abc": {"B/A": [0.382, 0.886], "C/A": [0.8, 1.272], "pivôs": 5},
    "fvg": {"regra": "mínima do candle i > máxima do candle i−2 (alta)", "gatilho": "reteste da lacuna"},
    "ob": {"regra": "último candle contrário antes do rompimento de estrutura", "gatilho": "reteste da zona"},
    "bos": {"regra": "fechamento além do último pivô confirmado a favor da tendência", "pivôs": 3},
    "choch": {"regra": "fechamento além do último pivô contra a tendência (mudança de caráter)", "pivôs": 3},
    "varredura": {"regra": "pavio ultrapassa o pivô e o candle fecha de volta (caça à liquidez)"},
    "duplo": {"tolerancia_atr": 0.6, "gatilho": "fechamento além da linha de pescoço"},
    "triplo": {"tolerancia_atr": 0.6, "gatilho": "fechamento além da linha de pescoço"},
    "oco": {"tolerancia_ombros_atr": 0.6, "gatilho": "rompimento da linha de pescoço"},
    "triangulo": {"pivôs": 4, "gatilho": "fechamento fora do triângulo"},
    "bandeira": {"mastro_atr": 3, "gatilho": "rompimento do canal de consolidação"},
    "fibo": {"retracoes": list(FIB_RETRACEMENTS), "extensoes": list(FIB_EXTENSIONS)},
}


def _family(name: str) -> str | None:
    n = name.lower()
    for key in ("gartley", "bat", "butterfly", "crab", "abcd", "elliott_onda5", "elliott_abc", "fvg", "ob_", "bos",
                "choch", "varredura", "duplo", "triplo", "oco", "triangulo", "bandeira", "fibo"):
        if key in n:
            return key.rstrip("_")
    return None


@lru_cache(maxsize=1)
def pattern_names() -> tuple[str, ...]:
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.003, "low": np.minimum(o, c) * 0.997, "close": c,
                       "volume": rng.gamma(2, 50, 300)},
                      index=pd.date_range("2025-01-01", periods=300, freq="4h", tz="UTC"))
    ev, _, _ = all_events(df)
    return tuple(ev.columns)


def catalog() -> list[dict[str, Any]]:
    out = []
    for name in pattern_names():
        fam = _family(name)
        out.append({"id": name, "categoria": category(name),
                    "direcao": "alta" if is_bull(name) else ("baixa" if is_bear(name) else "neutro"),
                    "familia": fam, "parametros": PARAMS.get(fam, {}) if fam else {}})
    return out


def stats_rows(study: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows = []
    for name, per_tf in (study or {}).get("events", {}).items():
        for tf, e in per_tf.items():
            rows.append({"pattern_id": name, "timeframe": tf, "veredito": e.get("veredito"),
                         "ocorrencias": e.get("ocorrencias"), "acerto": e.get("acerto"),
                         "acerto_bruto": e.get("acerto_bruto"), "excesso_bps": e.get("excesso_bps"),
                         "pf_operacao": e.get("pf_operacao")})
    return rows
