"""The approved strategy on the *real* account: which size, risk and fee setting actually make money?

The research lab measures edges on a large notional (so Binance's US$5 minimum never censors a trade).
This lab replays the book's strategy, out of sample, on small accounts with every rule of the live book —
risk per trade, max positions, daily loss limit, 2-loss lock and the US$5 minimum — under standard fees
and the BNB discount (0.075 % per side). Sizing choices do not create edge; they decide whether the edge
can be traded at all and how deep the drawdowns get.
"""

from __future__ import annotations

import logging
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from mqmas.research.engine import Costs, RiskRules, simulate_portfolio
from mqmas.research.lab import FIRST_TEST, Lab, monte_carlo
from mqmas.research.pattern_portfolio import equity_monthly
from mqmas.research.strategies import config_by_key

log = logging.getLogger(__name__)

VERSION = 1
EQUITY = (50.0, 80.0, 120.0)
RISK = (0.006, 0.01, 0.015)
MAX_POS = (1, 2, 3)
FEES = {"padrão 0,10%": 0.001, "BNB 0,075%": 0.00075}


def run(root: Path, strategy_key: str, ex: Any = None) -> dict[str, Any]:
    lab = Lab(root)
    lab.load(ex)
    cfg = config_by_key()[strategy_key]
    rows = []
    for fee_label, fee in FEES.items():
        lab.costs = Costs(fee=fee)
        trades = lab.bracket_trade_table([cfg])[cfg.key]
        for eq0, risk, mp in product(EQUITY, RISK, MAX_POS):
            rules = RiskRules(equity0=eq0, risk_per_trade=risk, max_positions=mp)
            res = simulate_portfolio(trades, rules, lab.costs, FIRST_TEST, lab.end)
            m = res.metrics
            monthly = equity_monthly(res.equity, eq0, lab.end)
            mc = monte_carlo(monthly)
            rows.append({"taxa": fee_label, "capital": eq0, "risco": risk, "max_posicoes": mp,
                         "trades": int(m["trades"]), "pulados": res.skipped, "acerto": round(m["win_rate"], 3),
                         "pf": round(m["profit_factor"], 2), "retorno_total": round(m["total_return"], 4),
                         "lucro_usd": round(m["total_return"] * eq0, 2), "cagr": round(m["cagr"], 4),
                         "max_dd": round(m["max_drawdown"], 4),
                         "prob_ano_negativo": round(mc.get("prob_losing_year", float("nan")), 3),
                         "dd_p95_mc": round(mc.get("dd_p95", float("nan")), 3)})
        log.info("account lab: %s done", fee_label)
    years = max((lab.end - FIRST_TEST).days / 365.25, 1e-9)
    return {"version": VERSION, "generated_at": pd.Timestamp.now(tz="UTC").isoformat(), "estrategia": strategy_key,
            "periodo": f"{FIRST_TEST.date()} → {lab.end.date()}", "anos": round(years, 2), "linhas": rows}


def summarize(rep: dict[str, Any]) -> list[str]:
    lines = [f"ACCOUNT v{rep['version']} | {rep['estrategia']} | {rep['periodo']} ({rep['anos']} anos)"]
    for r in sorted(rep["linhas"], key=lambda r: (r["taxa"], r["capital"], r["risco"], r["max_posicoes"])):
        lines.append(f"ACCOUNT {r['taxa']:<12} US${r['capital']:<5.0f} risco {r['risco']:.1%} maxpos {r['max_posicoes']} | "
                     f"n={r['trades']} pulados={r['pulados']} acerto {r['acerto']:.0%} PF {r['pf']} "
                     f"lucro US${r['lucro_usd']:+.2f} ({r['retorno_total']:+.1%}, {r['cagr']:+.1%}/ano) "
                     f"DD {r['max_dd']:.1%} | MC ano negativo {r['prob_ano_negativo']:.0%} DD95 {r['dd_p95_mc']:.1%}")
    return lines
