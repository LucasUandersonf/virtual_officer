from __future__ import annotations

import numpy as np
import pandas as pd

from mqmas.research.classic_study import btc_strength, evaluate, trade_stats
from mqmas.research.engine import Costs
from mqmas.research.strategies import FAMILIES, classic_configs, config_by_key, features
from tests.test_levels import market


def _trades(rets: list[float], start: str = "2022-08-01") -> pd.DataFrame:
    t_in = pd.date_range(start, periods=len(rets), freq="5D", tz="UTC")
    return pd.DataFrame({"coin": "SOL/USDT", "t_in": t_in, "t_out": t_in + pd.Timedelta(days=3),
                         "ret": rets, "stop_pct": 0.05, "regime": "alta"})


def test_classic_family_is_tradable_by_the_book():
    f = features(market(4, 1500))
    reg = np.ones(len(f), dtype=bool)
    counts = {c.p["setup"] + str(c.p["macro"]): int(FAMILIES[c.family](f, c.p, reg).sum()) for c in classic_configs()}
    assert counts["todosFalse"] > counts["todosTrue"] > 0  # the macro gate filters
    assert all(c.key in config_by_key() for c in classic_configs())


def test_elephant_loss_is_measured():
    ants = [0.01] * 9 + [-0.12]  # nine small wins, one big loss: the classic trap
    st = trade_stats(_trades(ants), Costs().round_trip)
    assert st["acerto"] == 0.9 and st["maior_perda_R"] == 2.4 and st["maior_perda_sobre_ganho_medio"] == 12.0
    assert st["pf"] < 1 and st["pior_devolucao_do_mes"] > 1


def test_three_to_one_with_forty_percent_is_approved_when_it_holds_everywhere():
    rets = ([0.15, -0.05, -0.05, 0.15, -0.05] * 40)
    tr = _trades(rets)
    btc = market(9, 1600).set_axis(pd.date_range("2021-06-01", periods=1600, freq="1D", tz="UTC"))
    out = evaluate(tr, Costs(), pd.Timestamp("2025-06-01", tz="UTC"), btc_strength(btc))
    assert out["estatisticas"]["acerto"] == 0.4 and out["estatisticas"]["payoff"] == 3.0
    assert out["aprovado"], out["falhas"]
