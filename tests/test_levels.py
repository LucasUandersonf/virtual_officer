from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mqmas.ta.levels import compact_map, price_map


def market(seed: int = 5, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.004, "low": np.minimum(o, c) * 0.996, "close": c,
                         "volume": rng.gamma(2, 50, n)},
                        index=pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC"))


def test_price_map_marks_supports_below_and_resistances_above():
    for seed in range(5):
        m = price_map(market(seed))
        json.dumps(m)
        assert m["suportes"] and m["resistencias"]
        assert all(z["preco"] < m["preco"] for z in m["suportes"])
        assert all(z["preco"] > m["preco"] for z in m["resistencias"])
        assert all(z["forca"] == len(z["fontes"]) for z in m["suportes"] + m["resistencias"])
        p = m["plano_compra"]
        if p.get("entrada"):
            assert p["stop"] < p["entrada"] < p["alvos"][0]
            assert p["valido"] == (p["risco_retorno"][0] >= 1.5)
        assert set(compact_map(m)) == {"suportes", "resistencias", "plano_compra", "plano_valido"}
