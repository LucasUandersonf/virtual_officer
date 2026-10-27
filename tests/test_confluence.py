from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pandas as pd

from mqmas.persistence.db import Database
from mqmas.ta.catalog import catalog, stats_rows
from mqmas.ta.confluence import RULE, entries, live_score, order_proposal, score_frame
from mqmas.ta.elliott import elliott_events
from mqmas.ta.signals import all_events


def market(seed: int = 3, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.004, "low": np.minimum(o, c) * 0.996, "close": c,
                         "volume": rng.gamma(2, 50, n)},
                        index=pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC"))


def test_score_is_bounded_and_proven_bearish_patterns_subtract():
    df = market()
    ev, f, _ = all_events(df)
    neutral = score_frame(df, ev, f, {})
    assert neutral["score"].between(0, 100).all()
    bear = score_frame(df, ev, f, {"engolfo_baixa": -1.0})
    hit = ev["engolfo_baixa"].rolling(3, min_periods=1).max().astype(bool)
    assert (bear["score"][hit] <= neutral["score"][hit]).all() and (bear["score"][hit] < neutral["score"][hit]).any()


def test_entries_fire_once_per_crossing():
    s = pd.Series([10, 80, 85, 60, 75, 90])
    assert entries(s, 70).tolist() == [1, 4]


def test_order_rule_is_1_to_2_and_needs_a_validated_threshold():
    sc = {"score": 82.0, "score_anterior": 60.0, "preco": 100.0, "atr": 2.0}
    p = order_proposal(sc, "4h", None, 1000.0, 0.006)
    assert p["decisao"] == "OBSERVAR"
    o = p["ordem"]
    assert o["stop"] == 100 - RULE.stop_atr * 2 and o["alvo"] == 100 + 2 * (100 - o["stop"])
    assert abs(o["risco_usdt"] - 6.0) < 1e-6
    val = {"tf": {"4h": {"limiar": 70}}}
    assert order_proposal(sc, "4h", val, 1000.0, 0.006)["decisao"] == "EXECUTAR"
    assert order_proposal({**sc, "score_anterior": 75.0}, "4h", val, 1000.0, 0.006)["decisao"] == "AGUARDAR"
    assert order_proposal({**sc, "score": 50.0}, "4h", val, 1000.0, 0.006)["decisao"] == "NÃO OPERAR"


def test_live_score_and_catalog():
    sc = live_score(market(), None, "4h")
    assert 0 <= sc["score"] <= 100 and set(sc["componentes"]) == {"localizacao", "padroes", "tendencia", "momento",
                                                                   "volume"}
    cat = catalog()
    ids = {p["id"] for p in cat}
    assert len(ids) == len(cat) > 150 and "elliott_onda5_alta" in ids and "01_BB_Bullish_Trap_Classic" in ids
    assert next(p for p in cat if "gartley" in p["id"].lower())["parametros"]["AD/XA"] == [0.74, 0.82]


def test_elliott_wave4_is_detected_on_a_textbook_impulse():
    # 0 →(1) 110 →(2) 104 →(3) 130 →(4) 120, each leg 12 bars; the wave-4 low confirms 5 bars later
    legs = [(106, 100), (100, 110), (110, 104), (104, 130), (130, 120), (120, 126)]
    path = np.concatenate([np.linspace(a, b, 13)[:-1] for a, b in legs] + [np.full(8, 126.0)])
    c = pd.Series(path)
    df = pd.DataFrame({"open": c.shift(1).fillna(c[0]), "high": c + 0.2, "low": c - 0.2, "close": c, "volume": 1.0})
    df.index = pd.date_range("2025-01-01", periods=len(df), freq="4h", tz="UTC")
    ev = elliott_events(df)
    assert ev["elliott_onda5_alta"].any() and not ev["elliott_onda5_baixa"].any()


def test_pattern_library_roundtrip(tmp_path: Path):
    async def go() -> list[dict]:
        db = Database(tmp_path / "t.db")
        await db.connect()
        study = {"events": {"engolfo_alta": {"4h": {"veredito": "sem efeito", "ocorrencias": 10, "acerto": 0.5,
                                                    "acerto_bruto": 0.55, "excesso_bps": 1.0, "pf_operacao": 0.9}}}}
        await db.sync_pattern_library(catalog(), stats_rows(study))
        await db.save_ta_signal("2025-01-01T00:00:00+00:00", "BTC/USDT", "4h", 72.5, {"a": 1}, [], {}, "OBSERVAR", {})
        rows = await db.pattern_library("4h")
        sigs = await db.recent_ta_signals()
        await db.close()
        assert sigs[0]["score"] == 72.5
        return rows

    rows = asyncio.run(go())
    row = next(r for r in rows if r["id"] == "engolfo_alta")
    assert row["veredito"] == "sem efeito" and row["acerto_bruto"] == 0.55
