"""Classic swing system: validation of a strategy with *no optimised parameters*.

Rules (mqmas.research.strategies, family ``classic``): textbook values only — EMA 20/50/200, ADX 20,
Donchian 20 with 1.5× volume, Bollinger 20/2, Fibonacci 50–61.8 % — behind a macro gate (BTC above
its SMA100, coin above EMA200 with EMA50 > EMA200, ADX ≥ 20, daily ATR between 1.5 % and 8 %).
Exit for every setup: stop 2 ATR, target 3R (6 ATR), time stop 20 days; entry at the next open.

Because nothing is fitted, every 6-month window is out of sample; the walk-forward table is simply the
result of each window in sequence. Checks, all fixed before running:

* costs — 0.10 % fee + 0.05 % slippage per side (0.30 % round trip); doubled; and the BNB rate;
* "ant gains, elephant losses" — largest loss in R, largest loss ÷ average win, longest losing streak,
  the share of a month's gains a single loss gave back, breakeven win rate vs the real one;
* regimes — results split by the coin's phase (alta/lateral/volatil/queda) and BTC's trend strength;
* value of the macro filter — the same setups with the gate switched off;
* the real account — US$50 at 1.5 % and 1.0 % risk, US$120 at 0.6 %, with the US$5 minimum.

Approval (pre-registered): net PF ≥ 1.3, doubled-cost PF ≥ 1.1, ≥ 60 trades, ≥ 60 % of windows
positive, largest loss ≤ 1.5R, no regime with ≥ 20 trades below PF 1.0.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mqmas.research.engine import Costs, RiskRules, simulate_portfolio
from mqmas.research.lab import Lab
from mqmas.research.pattern_portfolio import equity_monthly
from mqmas.research.strategies import adx, classic_configs, ema

log = logging.getLogger(__name__)

VERSION = 1
START = pd.Timestamp("2022-07-01", tz="UTC")  # first day with a valid EMA200 on the daily chart
CRITERIA: dict[str, float] = {"pf": 1.3, "pf_custo_dobrado": 1.1, "trades": 60, "janelas_positivas": 0.6, "maior_perda_R": 1.5,
            "pf_min_por_regime": 1.0}
ACCOUNTS = {"US$50 risco 1,5%": RiskRules(equity0=50.0, risk_per_trade=0.015),
            "US$50 risco 1,0%": RiskRules(equity0=50.0, risk_per_trade=0.01),
            "US$120 risco 0,6%": RiskRules(equity0=120.0, risk_per_trade=0.006)}


def _pf(r: np.ndarray) -> float:
    loss, gain = float(-r[r < 0].sum()), float(r[r > 0].sum())
    return round(min(gain / loss, 99.0), 2) if loss > 0 else (99.0 if gain > 0 else 0.0)


def windows(end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    out, s = [], START
    while s < end:
        e = min(s + pd.DateOffset(months=6), end)
        out.append((s, e))
        s = e
    return out


def btc_strength(btc: pd.DataFrame) -> pd.Series:
    """BTC phase known at each day's open: forte (above EMA200, ADX ≥ 25), lateral (ADX < 20), fraca otherwise,
    queda (below EMA200)."""
    c = btc["close"]
    e200, strength = ema(c, 200), adx(btc)
    lab = pd.Series("tendência fraca", index=btc.index)
    lab[(c > e200) & (strength >= 25)] = "tendência forte"
    lab[strength < 20] = "lateral"
    lab[c < e200] = "queda"
    return lab.shift(1).fillna("lateral")


def trade_stats(tr: pd.DataFrame, round_trip: float) -> dict[str, Any]:
    r = tr["ret"].to_numpy(dtype=float)
    if r.size == 0:
        return {"trades": 0}
    rr = r / tr["stop_pct"].to_numpy(dtype=float)  # result in R (multiples of the initial risk)
    wins, losses = r[r > 0], r[r <= 0]
    w = wins.size / r.size
    payoff = float(wins.mean() / -losses.mean()) if losses.size and wins.size else 0.0
    streak = best = 0
    for x in r:
        streak = streak + 1 if x <= 0 else 0
        best = max(best, streak)
    month = tr["t_out"].dt.tz_convert(None).dt.to_period("M")
    by_m = pd.DataFrame({"m": month.to_numpy(), "r": r})
    month_gain = by_m[by_m["r"] > 0].groupby("m")["r"].sum()
    month_worst = by_m.groupby("m")["r"].min()
    giveback = (-month_worst / month_gain.reindex(month_worst.index)).replace([np.inf, -np.inf], np.nan)
    gross = r + round_trip
    return {"trades": int(r.size), "acerto": round(w, 3), "payoff": round(payoff, 2),
            "acerto_de_equilibrio": round(1 / (1 + payoff), 3) if payoff else None,
            "pf": _pf(r), "pf_bruto": _pf(gross), "expectativa_pct": round(float(r.mean()) * 100, 3),
            "expectativa_R": round(float(rr.mean()), 3),
            "custos_sobre_lucro_bruto": round(round_trip * r.size / float(gross[gross > 0].sum()), 3)
            if (gross > 0).any() else None,
            "maior_perda_R": round(float(-rr.min()), 2), "maior_ganho_R": round(float(rr.max()), 2),
            "maior_perda_sobre_ganho_medio": round(float(-r.min() / wins.mean()), 2) if wins.size else None,
            "maior_sequencia_perdas": int(best),
            "pior_devolucao_do_mes": round(float(giveback.max()), 2) if giveback.notna().any() else None}


def evaluate(tr: pd.DataFrame, costs: Costs, end: pd.Timestamp, btc_phase: pd.Series) -> dict[str, Any]:
    tr = tr[tr["t_in"] >= START].sort_values("t_in")
    stats = trade_stats(tr, costs.round_trip)
    wins: list[dict[str, Any]] = []
    for s, e in windows(end):
        part = tr[(tr["t_in"] >= s) & (tr["t_in"] < e)]
        r = part["ret"].to_numpy(dtype=float)
        wins.append({"janela": f"{s.date()}→{e.date()}", "trades": int(r.size), "pf": _pf(r) if r.size else None,
                     "soma_pct": round(float(r.sum()) * 100, 1)})
    with_trades = [w for w in wins if w["trades"]]
    regimes: dict[str, Any] = {}
    for name, col in (("fase da moeda", tr["regime"]),
                      ("BTC", pd.Series(btc_phase.reindex(tr["t_in"].dt.normalize()).to_numpy(), index=tr.index))):
        regimes[name] = {}
        for k, part in tr.groupby(col.fillna("?")):
            r = part["ret"].to_numpy(dtype=float)
            regimes[name][str(k)] = {"trades": int(r.size), "pf": _pf(r), "acerto": round(float((r > 0).mean()), 3),
                                     "expectativa_pct": round(float(r.mean()) * 100, 2)}
    stressed = tr["ret"].to_numpy(dtype=float) - costs.round_trip
    accounts = {}
    for label, rules in ACCOUNTS.items():
        res = simulate_portfolio(tr, rules, costs, START, end)
        m = res.metrics
        monthly = equity_monthly(res.equity, rules.equity0, end)
        accounts[label] = {"trades": int(m["trades"]), "pulados": res.skipped, "pf": round(m["profit_factor"], 2),
                           "lucro_usd": round(m["total_return"] * rules.equity0, 2), "cagr": round(m["cagr"], 4),
                           "max_dd": round(m["max_drawdown"], 4),
                           "meses_positivos": round(float((monthly > 0).mean()), 2) if len(monthly) else None}
    out: dict[str, Any] = {"estatisticas": stats, "pf_custo_dobrado": _pf(stressed), "janelas": wins,
           "janelas_positivas": round(sum(w["soma_pct"] > 0 for w in with_trades) / max(len(with_trades), 1), 2),
           "regimes": regimes, "contas": accounts}
    fails = []
    if stats.get("trades", 0) < CRITERIA["trades"]:
        fails.append(f"trades {stats.get('trades', 0)} < {CRITERIA['trades']}")
    if stats.get("pf", 0) < CRITERIA["pf"]:
        fails.append(f"PF {stats.get('pf')} < {CRITERIA['pf']}")
    if out["pf_custo_dobrado"] < CRITERIA["pf_custo_dobrado"]:
        fails.append(f"PF com custo dobrado {out['pf_custo_dobrado']} < {CRITERIA['pf_custo_dobrado']}")
    if out["janelas_positivas"] < CRITERIA["janelas_positivas"]:
        fails.append(f"janelas positivas {out['janelas_positivas']:.0%} < 60%")
    if stats.get("maior_perda_R", 0) > CRITERIA["maior_perda_R"]:
        fails.append(f"maior perda {stats['maior_perda_R']}R > {CRITERIA['maior_perda_R']}R")
    for name, regs in regimes.items():
        for k, v in regs.items():
            if v["trades"] >= 20 and v["pf"] < CRITERIA["pf_min_por_regime"]:
                fails.append(f"perde em {name}={k} (PF {v['pf']}, {v['trades']} trades)")
    out["falhas"] = fails
    out["aprovado"] = not fails
    return out


def run(root: Path, ex: Any = None) -> dict[str, Any]:
    lab = Lab(root)
    lab.load(ex)
    phase = btc_strength(lab.btc_daily)
    out: dict[str, Any] = {"version": VERSION, "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
                           "periodo": f"{START.date()} → {lab.end.date()}", "criterios": CRITERIA, "custos": {}}
    for label, costs in (("padrão 0,10%", Costs()), ("BNB 0,075%", Costs(fee=0.00075))):
        lab.costs = costs
        table = lab.bracket_trade_table(classic_configs())
        out["custos"][label] = {c.key: {"setup": c.p["setup"], "filtro_macro": c.p["macro"],
                                        **evaluate(table[c.key], costs, lab.end, phase)} for c in classic_configs()}
        log.info("classic study: %s done", label)
    return out


def summarize(rep: dict[str, Any]) -> list[str]:
    lines = [f"CLASSIC v{rep['version']} | {rep['periodo']} | parâmetros otimizados: 0 | critérios {rep['criterios']}"]
    for cost_label, rows in rep["custos"].items():
        for r in rows.values():
            st = r["estatisticas"]
            tag = f"{r['setup']}{'' if r['filtro_macro'] else ' SEM filtro macro'}"
            if not st.get("trades"):
                lines.append(f"CLASSIC {cost_label} {tag}: sem trades")
                continue
            lines.append(
                f"CLASSIC {cost_label} {'APROVADO' if r['aprovado'] else 'reprovado'} {tag} | n={st['trades']} "
                f"acerto {st['acerto']:.0%} (equilíbrio {st['acerto_de_equilibrio']:.0%}) payoff {st['payoff']} "
                f"PF {st['pf']} (bruto {st['pf_bruto']}, custo 2x {r['pf_custo_dobrado']}) exp {st['expectativa_R']}R | "
                f"custos comem {st['custos_sobre_lucro_bruto']:.0%} do lucro bruto | maior perda {st['maior_perda_R']}R "
                f"= {st['maior_perda_sobre_ganho_medio']}x ganho médio, seq. perdas {st['maior_sequencia_perdas']} | "
                f"janelas+ {r['janelas_positivas']:.0%} | falhas: {'; '.join(r['falhas']) or '-'}")
            lines.append("    janelas: " + " | ".join(f"{w['janela']} n={w['trades']} {w['soma_pct']:+.1f}%"
                                                   for w in r["janelas"]))
            for name, regs in r["regimes"].items():
                lines.append(f"    regime {name}: " + " | ".join(
                    f"{k} n={v['trades']} PF {v['pf']} exp {v['expectativa_pct']:+.2f}%" for k, v in regs.items()))
            lines.append("    contas: " + " | ".join(
                f"{k}: n={v['trades']} pulados {v['pulados']} PF {v['pf']} US${v['lucro_usd']:+.2f} "
                f"({v['cagr']:+.1%}/ano) DD {v['max_dd']:.1%} meses+ {v['meses_positivos']}" for k, v in r["contas"].items()))
    return lines
