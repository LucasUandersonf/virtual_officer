"""Aggressive-but-controlled study: can the swing book reach 2–3 %/month without a ruinous drawdown?

Three questions, all answered out of sample (after 2023-07) with choices made on the earlier period only:

1. Management — the book's five swing setups (1d/2d/3d breakout, breakout + MA cross, 3d descending
   triangle; no fixed target) under 12 exit variants: breakeven at 1R, 1.5R or never; trailing from 1.5R or
   2R; trailing at the higher of the last 3 lows and peak − 2.5 ATR (current, tight) or at peak − 3.5 ATR only
   (loose: lets winners run). The variant with the best expectancy *before* 2023-07 is the pick.
2. New signals — four "elite" indicators, written here with fixed textbook values (nothing tuned):
   Kalman trend (local linear trend on log price, slope turning up with price above the level), LazyBear's
   Squeeze Momentum (Bollinger 20/2 inside Keltner 20/1.5 released with rising positive momentum),
   WaveTrend (10/21; crosses up below −53) and the 20-candle rolling VWAP reclaimed. Alone, behind the macro
   gate, and confirming the breakout; same survivor bar as the arena.
3. Sizing — the book's trades replayed on one account for risk 1.5–3 %, capital US$50/200/500, a cap on the
   summed open risk (heat 6 % / 9 % / none), drawdown protection (half risk below −10 %, quarter below −20 %)
   and 5 or 8 positions, plus a wider universe (top 25 instead of 15). Pre-registered approval: max drawdown
   ≤ 25 %, mean month ≥ 2 %, PF ≥ 1.2 with doubled costs.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mqmas.book import swing
from mqmas.research import arena
from mqmas.research.engine import Costs, RiskRules
from mqmas.research.lab import FIRST_TEST, Lab
from mqmas.research.strategies import btc_regime, ema, monthly_eligibility

log = logging.getLogger(__name__)

VERSION = 1
BASE: dict[str, Any] = {"be_r": 1.0, "trail_r": 2.0, "trail_atr": 2.5, "low3": True}
TRAILS: tuple[dict[str, Any], ...] = ({"trail_atr": 2.5, "low3": True}, {"trail_atr": 3.5, "low3": False})
GESTAO = [{"be_r": b, "trail_r": t, **m} for b in (1.0, 1.5, 0.0) for t in (1.5, 2.0) for m in TRAILS]
NEW = ("kalman", "squeeze", "wavetrend", "vwap")
RISKS = (0.015, 0.02, 0.025, 0.03)
CAPITALS = (50.0, 200.0, 500.0)
HEATS = (None, 0.06, 0.09)
MAX_POS = (5, 8)
CRITERIA = {"max_dd": 0.25, "media_mensal_pct": 2.0, "pf_custo_dobrado": 1.2}


# ---------------------------------------------------------------------- new signals (causal, fixed values)

def kalman_trend(close: pd.Series, ratio: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Local linear trend on log price; measurement noise = variance of the last 20 returns (known at t−1)."""
    y = np.log(close.to_numpy(dtype=float))
    rv = pd.Series(np.diff(y, prepend=y[0])).rolling(20, min_periods=5).var().shift(1).bfill().to_numpy()
    n = y.size
    level, slope = np.empty(n), np.empty(n)
    lv, sl = y[0], 0.0
    p00, p01, p11 = 1.0, 0.0, 1.0
    for i in range(n):
        r = max(float(rv[i]) if np.isfinite(rv[i]) else 1e-4, 1e-8)
        # predict
        lv, sl = lv + sl, sl
        p00, p01, p11 = p00 + 2 * p01 + p11 + r * ratio, p01 + p11, p11 + r * ratio * 0.1
        # update
        s = p00 + r
        k0, k1 = p00 / s, p01 / s
        innov = y[i] - lv
        lv, sl = lv + k0 * innov, sl + k1 * innov
        p00, p01, p11 = (1 - k0) * p00, (1 - k0) * p01, p11 - k1 * p01
        level[i], slope[i] = lv, sl
    return np.exp(level), slope


def _linreg_end(s: pd.Series, n: int) -> pd.Series:
    x = np.arange(n, dtype=float)
    w = (x - x.mean()) / ((x - x.mean()) ** 2).sum()
    slope = s.rolling(n).apply(lambda v: float(np.dot(w, v)), raw=True)
    return s.rolling(n).mean() + slope * (n - 1 - x.mean())


def new_signals(df: pd.DataFrame) -> dict[str, pd.Series]:
    c, h, lo, v = df["close"], df["high"], df["low"], df["volume"]
    level, slope = kalman_trend(c)
    sl = pd.Series(slope, index=df.index)
    kalman = (sl > 0) & (sl.shift(1) <= 0) & (c > pd.Series(level, index=df.index))
    mid, sd = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    rng = tr.rolling(20).mean()
    sqz_on = (mid - 2 * sd > mid - 1.5 * rng) & (mid + 2 * sd < mid + 1.5 * rng)
    mom = _linreg_end(c - ((h.rolling(20).max() + lo.rolling(20).min()) / 2 + mid) / 2, 20)
    squeeze = sqz_on.shift(1, fill_value=False) & ~sqz_on & (mom > 0) & (mom > mom.shift(1))
    ap = (h + lo + c) / 3
    esa = ema(ap, 10)
    d = ema((ap - esa).abs(), 10)
    wt1 = ema((ap - esa) / (0.015 * d.replace(0, np.nan)), 21)
    wt2 = wt1.rolling(4).mean()
    wavetrend = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1)) & (wt1 < -53)
    vwap = (ap * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    vw = (c > vwap) & (c.shift(1) <= vwap.shift(1))
    return {k: s.fillna(False).astype(bool) for k, s in
            zip(NEW, (kalman, squeeze, wavetrend, vw), strict=True)}


def frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    sig, st, exits = arena.signal_frame(df)
    low5 = df["low"].rolling(5).min()
    for k, s in new_signals(df).items():
        sig[k] = s
        st[k] = low5
    return sig, st, exits


# ---------------------------------------------------------------------- candidates

def gkey(g: dict[str, Any]) -> str:
    be = f"BE {g['be_r']:g}R" if g["be_r"] else "sem BE"
    how = "3 mínimas/2,5 ATR" if g["low3"] else f"{g['trail_atr']:g} ATR"
    return f"{be}, trailing de {g['trail_r']:g}R ({how})"


def setup_cands(tf: str, gestoes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for s in (x for x in swing.SETUPS if x["tf"] == tf):
        for g in gestoes:
            out.append({**s, "key": f"{tf}|{s['nome']}|{gkey(g)}", "macro": False, "alvo_r": 0.0, "gestao": gkey(g),
                        **g})
    return out


def signal_cands(tf: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for n in NEW:
        for m in (False, True):
            out.append({"key": arena._key(tf, "individual", (n,), m, 0.0), "tf": tf, "tipo": "individual",
                        "membros": (n,), "macro": m, "alvo_r": 0.0})
        out.append({"key": arena._key(tf, "par", ("cl_rompimento", n), False, 0.0), "tf": tf, "tipo": "par",
                    "membros": ("cl_rompimento", n), "macro": False, "alvo_r": 0.0})
    return out


# ---------------------------------------------------------------------- accounts

def account(tr: pd.DataFrame, rules: RiskRules, costs: Costs, end: pd.Timestamp) -> dict[str, Any]:
    a = arena.account(tr, rules, costs, FIRST_TEST, end)
    stressed = tr.assign(ret=tr["ret"] - costs.round_trip)
    a["pf_custo_dobrado"] = arena.account(stressed, rules, costs, FIRST_TEST, end)["pf"]
    fails = []
    if a["max_dd"] > CRITERIA["max_dd"]:
        fails.append(f"DD {a['max_dd']:.0%}")
    if (a["media_mensal_pct"] or 0) < CRITERIA["media_mensal_pct"]:
        fails.append(f"mês médio {a['media_mensal_pct']}%")
    if a["pf_custo_dobrado"] < CRITERIA["pf_custo_dobrado"]:
        fails.append(f"PF custo 2x {a['pf_custo_dobrado']}")
    a["aprovado"], a["falhas"] = not fails, fails
    return a


def grid(tr: pd.DataFrame, costs: Costs, end: pd.Timestamp) -> list[dict[str, Any]]:
    rows = []
    for risk, cap, heat, prot, mp in itertools.product(RISKS, CAPITALS, HEATS, (False, True), MAX_POS):
        rules = RiskRules(equity0=cap, risk_per_trade=risk, max_positions=mp, daily_loss_limit=max(0.015, risk),
                          heat_max=heat, dd_throttle=0.10 if prot else None, dd_severe=0.20 if prot else None)
        rows.append({"risco": risk, "capital": cap, "teto_exposicao": heat, "protecao_dd": prot, "posicoes": mp,
                     **account(tr, rules, costs, end)})
    return rows


def _split(v: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (v[v["t_in"] < FIRST_TEST], v[v["t_in"] >= FIRST_TEST]) if len(v) else (v, v)


def run(root: Path, ex: Any = None) -> dict[str, Any]:
    lab = Lab(root)
    lab.load(ex)
    costs = Costs()
    regime = btc_regime(lab.btc_daily)
    out: dict[str, Any] = {"version": VERSION, "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
                           "periodo_oos": f"{FIRST_TEST.date()} → {lab.end.date()}", "criterios": CRITERIA}

    # 1. management variants on the book's setups
    tables: dict[str, pd.DataFrame] = {}
    for tf in swing.TFS:
        tables.update(arena.run_tf(lab, tf, costs, regime, frame_fn=arena.signal_frame,
                                   cands=setup_cands(tf, GESTAO)))
    by_g: dict[str, pd.DataFrame] = {}
    for g in GESTAO:
        parts = [v.assign(tf=k.split("|")[0]) for k, v in tables.items() if k.endswith(gkey(g)) and len(v)]
        by_g[gkey(g)] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for name, v in by_g.items():
        p1, p2 = _split(v)
        rows.append({"gestao": name, "p1": arena._stats(p1), "p2": arena._stats(p2)})
    pick: str = max(rows, key=lambda r: r["p1"].get("expectativa_R", -9))["gestao"]
    out["gestao"] = {"variantes": rows, "escolhida_no_p1": pick, "atual": gkey(BASE)}
    log.info("aggressive: management pick (period 1) = %s", pick)

    # 2. new signals
    sig_rows = []
    for tf in swing.TFS:
        cands = signal_cands(tf)
        t = arena.run_tf(lab, tf, costs, regime, frame_fn=frame, cands=cands)
        for c in cands:
            p1, p2 = _split(t[c["key"]])
            sig_rows.append({"key": c["key"], "p1": arena._stats(p1), "p2": arena._stats(p2)})
    survivors = [r for r in sig_rows if r["p2"].get("trades", 0) >= 40 and r["p1"].get("trades", 0) >= 20
                 and arena.HIT_BAND[0] <= r["p2"]["acerto"] <= arena.HIT_BAND[1] and r["p2"]["pf"] >= 1.3
                 and r["p1"]["pf"] >= 1.2]
    out["sinais_novos"] = {"linhas": sig_rows, "sobreviventes": sorted(survivors, key=lambda r: -r["p2"]["pf"])}

    # 3. sizing grid on the picked management (and the current one, for comparison)
    out["contas"] = {}
    for label, name in (("gestão escolhida", pick), ("gestão atual", gkey(BASE))):
        out["contas"][label] = grid(by_g[name], costs, lab.end)

    # universe: top 25 instead of 15, picked management
    lab.eligible = monthly_eligibility(lab.data["1d"], 25, lab.min_range, lab.max_range)
    wide: list[pd.DataFrame] = []
    chosen = next(x for x in GESTAO if gkey(x) == pick)
    for tf in swing.TFS:
        t = arena.run_tf(lab, tf, costs, regime, frame_fn=arena.signal_frame, cands=setup_cands(tf, [chosen]))
        wide += [v for v in t.values() if len(v)]
    if wide:
        w = pd.concat(wide, ignore_index=True)
        p1, p2 = _split(w)
        out["universo_25"] = {"p1": arena._stats(p1), "p2": arena._stats(p2), "contas": grid(w, costs, lab.end)}
    return out


def summarize(rep: dict[str, Any]) -> list[str]:
    lines = [f"AGRESSIVO v{rep['version']} | OOS {rep['periodo_oos']} | critérios {rep['criterios']}"]
    g = rep["gestao"]
    for r in g["variantes"]:
        lines.append(f"AGRESSIVO gestão {r['gestao']} | p1 {r['p1']} → p2 {r['p2']}")
    lines.append(f"AGRESSIVO gestão escolhida no período 1: {g['escolhida_no_p1']} (atual: {g['atual']})")
    for s in rep["sinais_novos"]["sobreviventes"][:10]:
        lines.append(f"AGRESSIVO sinal novo sobrevivente {s['key']} | p1 {s['p1']} → p2 {s['p2']}")
    if not rep["sinais_novos"]["sobreviventes"]:
        lines.append("AGRESSIVO sinais novos (Kalman, Squeeze, WaveTrend, VWAP): nenhum sobrevivente")

    def acc(a: dict[str, Any]) -> str:
        return (f"risco {a['risco']:.1%} US${a['capital']:.0f} teto {a['teto_exposicao']} proteção {a['protecao_dd']} "
                f"{a['posicoes']}pos | n={a['trades']} pulados {a['pulados']} acerto {a['acerto']:.0%} PF {a['pf']} "
                f"(2x {a['pf_custo_dobrado']}) mês médio {a['media_mensal_pct']}% meses+ {a['meses_positivos']} "
                f"DD {a['max_dd']:.1%} lucro US${a['lucro_usd']:+.2f} {'APROVADO' if a['aprovado'] else ''}")

    for label, rows in rep["contas"].items():
        ok = [a for a in rows if a["aprovado"]]
        lines.append(f"AGRESSIVO {label}: {len(ok)} de {len(rows)} contas aprovadas")
        for a in sorted(rows, key=lambda a: -(a["media_mensal_pct"] or -99))[:8]:
            lines.append(f"AGRESSIVO {label} top: {acc(a)}")
        for a in sorted(ok, key=lambda a: a["max_dd"])[:5]:
            lines.append(f"AGRESSIVO {label} aprovada: {acc(a)}")
        for a in rows:
            if a["capital"] in (50.0, 200.0) and a["teto_exposicao"] == 0.06 and a["protecao_dd"] and a["posicoes"] == 5:
                lines.append(f"AGRESSIVO {label} linha-base: {acc(a)}")
    u = rep.get("universo_25")
    if u:
        lines.append(f"AGRESSIVO universo top 25 | p1 {u['p1']} → p2 {u['p2']} | aprovadas "
                     f"{sum(a['aprovado'] for a in u['contas'])} de {len(u['contas'])}")
        for a in sorted(u["contas"], key=lambda a: -(a["media_mensal_pct"] or -99))[:4]:
            lines.append(f"AGRESSIVO universo 25 top: {acc(a)}")
    return lines
