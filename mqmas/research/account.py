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


# ---------------------------------------------------------------------- swing book (several setups, one account)

SWING_VERSION = 1
# The three families with PF ≥ 1.7 out of sample and ≥ 1.6 with doubled costs in the lab report. They were
# picked after seeing that report, so this test checks how they combine, not whether they are real.
SWING_MEMBERS = ("confluence|1d", "breakout|1d", "tpl:golden_cross|4h")
SWING_ACCOUNTS = {"US$50 risco 1,5% até 3 posições": RiskRules(equity0=50.0, risk_per_trade=0.015, max_positions=3),
                  "US$50 risco 1,5% até 5 posições": RiskRules(equity0=50.0, risk_per_trade=0.015, max_positions=5),
                  "US$50 risco 1,0% até 5 posições": RiskRules(equity0=50.0, risk_per_trade=0.01, max_positions=5),
                  "US$120 risco 1,0% até 5 posições": RiskRules(equity0=120.0, risk_per_trade=0.01, max_positions=5)}


def _account_row(trades: pd.DataFrame, rules: RiskRules, costs: Costs, end: pd.Timestamp) -> dict[str, Any]:
    res = simulate_portfolio(trades, rules, costs, FIRST_TEST, end)
    m = res.metrics
    weeks = max((end - FIRST_TEST).days / 7, 1e-9)
    mc = monte_carlo(equity_monthly(res.equity, rules.equity0, end))
    return {"trades": int(m["trades"]), "por_semana": round(m["trades"] / weeks, 2), "pulados": res.skipped,
            "acerto": round(m["win_rate"], 3), "pf": round(m["profit_factor"], 2),
            "lucro_usd": round(m["total_return"] * rules.equity0, 2), "cagr": round(m["cagr"], 4),
            "max_dd": round(m["max_drawdown"], 4), "prob_ano_negativo": round(mc.get("prob_losing_year", 0.0), 3),
            "dd_p95_mc": round(mc.get("dd_p95", 0.0), 3)}


def run_swing(root: Path, ex: Any = None) -> dict[str, Any]:
    from mqmas.research.strategies import build_configs

    out: dict[str, Any] = {"version": SWING_VERSION, "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
                           "membros": list(SWING_MEMBERS), "universos": {}}
    for top_n in (15, 30):
        lab = Lab(root, top_n=top_n)
        lab.load(ex)
        configs = [c for c in build_configs() if f"{c.family}|{c.tf}" in SWING_MEMBERS]
        table = lab.bracket_trade_table(configs)
        groups: dict[str, list[str]] = {}
        for c in configs:
            groups.setdefault(f"{c.family}|{c.tf}", []).append(c.key)
        tpl_table, tpl_groups = lab.template_trade_table()
        table.update(tpl_table)
        groups.update({k: v for k, v in tpl_groups.items() if k in SWING_MEMBERS})
        wf = lab.walk_forward(groups, table)
        parts = {n: wf[n]["oos_trades"].assign(setup=n) for n in SWING_MEMBERS
                 if n in wf and wf[n]["oos_trades"] is not None}
        ens = pd.concat(parts.values(), ignore_index=True)
        stressed = ens.assign(ret=ens["ret"] - lab.costs.round_trip)
        big = RiskRules(equity0=10_000.0, risk_per_trade=0.01, max_positions=5)
        out["universos"][f"top{top_n}"] = {
            "periodo": f"{FIRST_TEST.date()} → {lab.end.date()}",
            "membros": {n: _account_row(t, big, lab.costs, lab.end) for n, t in parts.items()},
            "carteira_custo_dobrado": _account_row(stressed, big, lab.costs, lab.end),
            "contas": {label: _account_row(ens, rules, lab.costs, lab.end) for label, rules in SWING_ACCOUNTS.items()}}
        log.info("swing book: top%d done", top_n)
    return out


def summarize_swing(rep: dict[str, Any]) -> list[str]:
    lines = [f"SWINGBOOK v{rep['version']} | membros {', '.join(rep['membros'])}"]

    def fmt(r: dict[str, Any]) -> str:
        return (f"n={r['trades']} ({r['por_semana']}/sem) pulados={r['pulados']} acerto {r['acerto']:.0%} PF {r['pf']} "
                f"lucro US${r['lucro_usd']:+.2f} ({r['cagr']:+.1%}/ano) DD {r['max_dd']:.1%} | MC ano negativo "
                f"{r['prob_ano_negativo']:.0%} DD95 {r['dd_p95_mc']:.1%}")

    for uni, u in rep["universos"].items():
        lines.append(f"SWINGBOOK {uni} {u['periodo']} | custo dobrado: {fmt(u['carteira_custo_dobrado'])}")
        for n, r in u["membros"].items():
            lines.append(f"SWINGBOOK {uni} membro {n}: {fmt(r)}")
        for label, r in u["contas"].items():
            lines.append(f"SWINGBOOK {uni} {label}: {fmt(r)}")
    return lines
