"""Portfolio test of the TA study's candidates: is a pattern that "works" also tradable with our risk rules?

Each candidate (every pattern with verdict ``funciona``, plus the team's combinations) is traded on all
coins with the study's bracket (next open, stop 1.5 ATR, target 2R, time stop), then replayed on one
shared account with the book's rules: 0.6 % risk, max 3 positions, daily loss limit, 2-loss lock,
US$5 minimum notional. Period: the out-of-sample window (from FIRST_TEST). Stress: costs doubled.
Caveat, stated in the report: candidates were picked by a study that saw both periods, so the multiple
testing across ~340 signals is only partly corrected (it required |t| ≥ 2 in each period separately).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mqmas.research.engine import Costs, RiskRules, daily_curve, simulate_portfolio
from mqmas.research.lab import FIRST_TEST, monte_carlo
from mqmas.research.seasonality import load_cached
from mqmas.research.swing import resample
from mqmas.research.ta_study import HOLD
from mqmas.research.templates import Plan, structured_trades
from mqmas.ta.signals import COMBOS, all_events

log = logging.getLogger(__name__)

VERSION = 1
ACCOUNTS = {"A: US$120 a 0,6%": RiskRules(equity0=120.0, risk_per_trade=0.006),
            "B: US$50 a 1,5%": RiskRules(equity0=50.0, risk_per_trade=0.015),
            "atual: US$50 a 0,6%": RiskRules(equity0=50.0, risk_per_trade=0.006)}


def candidates(study: dict[str, Any]) -> list[tuple[str, str]]:
    out = [(r["padrao"], r["tf"]) for r in study.get("rows", []) if r["veredito"] == "funciona" and r["vies"] != "baixa"]
    return out + [(c, tf) for c in COMBOS for tf in ("4h", "1d") if (c, tf) not in out]


def run(root: Path, events_fn: Any = None, study: dict[str, Any] | None = None) -> dict[str, Any]:
    events_fn = events_fn or all_events
    if study is None:
        path = root / "ta_report.json"
        study = json.loads(path.read_text()) if path.exists() else {}
    cands = candidates(study)
    h1 = load_cached(root)
    tables: dict[tuple[str, str, float], list[pd.DataFrame]] = {}
    for coin, d in h1.items():
        for tf in ("4h", "1d"):
            names = [n for n, t in cands if t == tf]
            if not names:
                continue
            df = resample(d, "4h", 4) if tf == "4h" else resample(d, "1D", 24)
            df = df[["open", "high", "low", "close", "volume"]]
            ev, f, _ = events_fn(df)
            o, hi, lo, cl = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
            a = f["atr"].to_numpy()
            for name in names:
                if name not in ev.columns:
                    continue
                idx = np.flatnonzero(ev[name].to_numpy() & (np.arange(len(df)) >= 200))
                for mult in (1.0, 2.0):
                    t = structured_trades(o, hi, lo, cl, a, Plan(idx, np.full(idx.size, np.nan), target_r=2.0,
                                                                 stop_atr=1.5, hold=HOLD[tf]),
                                          Costs(fee=0.001 * mult, slippage=0.0005 * mult))
                    if t["entry"].size:
                        tables.setdefault((name, tf, mult), []).append(pd.DataFrame({
                            "coin": coin, "t_in": df.index[t["entry"].astype(int)],
                            "t_out": df.index[np.minimum(t["exit"].astype(int), len(df) - 1)],
                            "ret": t["ret"], "stop_pct": t["stop_pct"]}))
        log.info("pattern portfolio: %s done", coin)
    end = max((d.index[-1] for d in h1.values()), default=pd.Timestamp.now(tz="UTC"))
    rows = []
    for name, tf in cands:
        if (name, tf, 1.0) not in tables:
            continue
        tr = pd.concat(tables[(name, tf, 1.0)])
        tr2 = pd.concat(tables[(name, tf, 2.0)])
        accounts: dict[str, dict[str, Any]] = {}
        for label, rules in ACCOUNTS.items():
            res = simulate_portfolio(tr, rules, Costs(), start=FIRST_TEST, end=end)
            m = res.metrics
            monthly = equity_monthly(res.equity, rules.equity0, end)
            accounts[label] = {k: round(float(m[k]), 4) for k in ("trades", "win_rate", "profit_factor",
                                                                    "max_drawdown", "total_return", "cagr")}
            accounts[label]["pulados_por_minimo_ou_risco"] = res.skipped
            accounts[label]["monte_carlo"] = {k: round(v, 3) for k, v in monte_carlo(monthly).items()}
        stress = simulate_portfolio(tr2, ACCOUNTS["A: US$120 a 0,6%"], Costs(), start=FIRST_TEST, end=end).metrics
        a = accounts["A: US$120 a 0,6%"]
        ok = bool(a["trades"] >= 30 and a["profit_factor"] >= 1.2 and a["total_return"] > 0
                  and a["max_drawdown"] <= 0.15 and stress["profit_factor"] > 1.0)
        rows.append({"padrao": name, "tf": tf, "contas": accounts,
                     "custo_dobrado": {"profit_factor": round(float(stress["profit_factor"]), 2),
                                       "total_return": round(float(stress["total_return"]), 4)},
                     "aprovado": ok})
    return {"version": VERSION, "generated_at": pd.Timestamp.now(tz="UTC").isoformat(), "coins": list(h1),
            "periodo": f"{FIRST_TEST.date()} → {end.date()}", "linhas": rows,
            "ressalva": "candidatos escolhidos por um estudo que viu os dois períodos; trate como evidência, "
                        "não como prova"}


def equity_monthly(eq: pd.Series, equity0: float, end: pd.Timestamp) -> pd.Series:
    months = daily_curve(eq, end).resample("ME").last()
    monthly = months.pct_change()
    if len(months):
        monthly.iloc[0] = months.iloc[0] / equity0 - 1
    return monthly.dropna()


def summarize(rep: dict[str, Any]) -> list[str]:
    lines = [f"PATTERN PORTFOLIO v{rep['version']} | {len(rep['coins'])} moedas | {rep['periodo']} | {rep['ressalva']}"]
    for r in rep["linhas"]:
        parts = []
        for label, a in r["contas"].items():
            parts.append(f"{label}: n={a['trades']:.0f} acerto {a['win_rate']:.0%} PF {a['profit_factor']:.2f} "
                         f"ret {a['total_return']:+.1%} DD {a['max_drawdown']:.1%} pulados {a['pulados_por_minimo_ou_risco']}")
        lines.append(f"{'APROVADO' if r['aprovado'] else 'reprovado'} {r['padrao']} {r['tf']} | " + " | ".join(parts)
                     + f" | custo 2x PF {r['custo_dobrado']['profit_factor']}")
    return lines
