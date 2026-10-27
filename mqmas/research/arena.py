"""Strategy arena: every rule we have, on six timeframes, alone and in confluence, chosen blind.

Timeframes — day trade 1h, 2h, 4h; swing 1d, 2d, 3d (2h/2d/3d resampled from Binance 1h/1d, epoch-anchored).
Signals — the 11 long rules of the book library (Prates 2021 + Stanzione) and the 4 classic triggers
(EMA20 pullback, 20-candle breakout with volume, Bollinger re-entry, Fibonacci 50–61.8 %).
Management — the library's rule for every trade: initial stop = farther of the technical stop and 1.5 ATR
(cap 3 ATR), breakeven at +1R covering costs, trailing from +2R; a fixed target at 1R, 1.5R or 2R (higher
hit rate) or none (trailing only); time stop. Costs 0.10 % + 0.05 % per side. One trade per coin at a time.
Gate — optional macro filter as it stood at the previous daily close (BTC > SMA100, coin > EMA200 with
EMA50 > EMA200, ADX ≥ 20, ATR 1.5–8 %).

Stage 1: each signal alone (15 × macro × 4 exits per timeframe). Stage 2: pairs among the six best singles
of the timeframe *ranked on period 1 only*, plus "≥ 2 / ≥ 3 distinct signals in 3 candles". Then, per
timeframe, walk-forward: every 6 months the best candidate on all earlier data trades the next window blind.
The combined book takes every timeframe's blind picks at once on one small account (US$5 minimum enforced).
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mqmas.research import livro
from mqmas.research.engine import Costs, RiskRules, simulate_portfolio
from mqmas.research.lab import FIRST_TEST, Lab, deflated_sharpe
from mqmas.research.strategies import align_daily, btc_regime, classic_macro, classic_triggers, features

log = logging.getLogger(__name__)

VERSION = 1
TFS = {"1h": ("1h", None), "2h": ("1h", "2h"), "4h": ("4h", None), "1d": ("1d", None), "2d": ("1d", "2D"),
       "3d": ("1d", "3D")}
TF_DELTA = {"1h": pd.Timedelta(hours=1), "2h": pd.Timedelta(hours=2), "4h": pd.Timedelta(hours=4),
            "1d": pd.Timedelta(days=1), "2d": pd.Timedelta(days=2), "3d": pd.Timedelta(days=3)}
HOLD = {"1h": 48, "2h": 36, "4h": 42, "1d": 20, "2d": 15, "3d": 12}
WARMUP = {"1h": 250, "2h": 250, "4h": 250, "1d": 220, "2d": 120, "3d": 70}
CLASSIC = ("cl_tendencia", "cl_rompimento", "cl_bollinger", "cl_fibonacci")
SIGNALS = tuple(livro.LONGS) + CLASSIC
TARGETS = (0.0, 1.0, 1.5, 2.0)  # R multiples; 0 = trailing only
HIT_BAND = (0.40, 0.60)
ACCOUNTS = {"US$50 risco 1,5% 3 posições": RiskRules(equity0=50.0, risk_per_trade=0.015),
            "US$50 risco 1,5% 5 posições": RiskRules(equity0=50.0, risk_per_trade=0.015, max_positions=5),
            "US$50 risco 2,0% 5 posições": RiskRules(equity0=50.0, risk_per_trade=0.02, max_positions=5),
            "US$120 risco 1,0% 5 posições": RiskRules(equity0=120.0, risk_per_trade=0.01, max_positions=5)}


# ---------------------------------------------------------------------- fast trade management

def _manage_py(o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray, atr: np.ndarray, entries: np.ndarray,
               stop_tech: np.ndarray, tgt_r: float, exit_sig: np.ndarray, rt: float, side: float,
               hold: int) -> np.ndarray:
    n = o.size
    out = np.empty((n, 5))
    m = 0
    busy = -1
    for i in range(n - 2):
        if not entries[i] or i < busy or not (atr[i] > 0):
            continue
        e = i + 1
        px = o[e]
        d_tech = px - stop_tech[i] if stop_tech[i] == stop_tech[i] else 0.0
        risk = min(max(d_tech, 1.5 * atr[i]), 3.0 * atr[i])
        stop = px - risk
        tgt = px + tgt_r * risk if tgt_r > 0 else np.inf
        peak = px
        exit_px = np.nan
        reason = 4
        k = e
        last = min(e + hold, n) - 1
        while True:
            if k > e and o[k] <= stop:
                exit_px, reason = o[k], 1
                break
            if lo[k] <= stop:
                exit_px, reason = stop, 1
                break
            if h[k] >= tgt:
                exit_px, reason = (max(o[k], tgt) if k > e else tgt), 2
                break
            if exit_sig[k] and k + 1 < n:
                k += 1
                exit_px, reason = o[k], 3
                break
            if k >= last:
                exit_px, reason = c[k], 4
                break
            if h[k] > peak:
                peak = h[k]
            if peak >= px + risk and stop < px * (1 + rt):
                stop = px * (1 + rt)
            if peak >= px + 2 * risk and atr[k] > 0:
                lmin = lo[k]
                for j in range(max(k - 2, e), k):
                    lmin = min(lmin, lo[j])
                trail = max(lmin, peak - 2.5 * atr[k])
                if trail > stop:
                    stop = trail
            k += 1
        out[m, 0] = e
        out[m, 1] = k
        out[m, 2] = (exit_px * (1 - side)) / (px * (1 + side)) - 1
        out[m, 3] = risk / px
        out[m, 4] = reason
        m += 1
        busy = k
    return out[:m]


try:  # numba makes the arena ~100× faster; the pure-Python version gives identical results
    from numba import njit

    _manage: Any = njit(cache=False)(_manage_py)
except ImportError:  # pragma: no cover
    _manage = _manage_py


def manage(o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray, atr: np.ndarray, entries: np.ndarray,
           stop_tech: np.ndarray, tgt_r: float, exit_sig: np.ndarray, costs: Costs, hold: int) -> np.ndarray:
    f8 = np.ascontiguousarray
    return np.asarray(_manage(f8(o, float), f8(h, float), f8(lo, float), f8(c, float), f8(np.nan_to_num(atr), float),
                              f8(entries, bool), f8(stop_tech, float), float(tgt_r), f8(exit_sig, bool),
                              float(costs.round_trip), float(costs.fee + costs.slippage), int(hold)))


# ---------------------------------------------------------------------- data and signals

def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    naive = df[list(agg)].tz_localize(None)
    out = naive.resample(rule, origin="epoch", label="left", closed="left").agg(agg).dropna().tz_localize("UTC")
    step = pd.Timedelta(rule.lower().replace("d", "D")) if rule[-1] in "dD" else pd.Timedelta(rule)
    last_close = df.index[-1] + (df.index[1] - df.index[0])
    return out[out.index + step <= last_close]


def signal_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Boolean entry column per signal, technical stop per signal, and the library's exit-pattern union."""
    ev, stops, _ = livro.detect(df)
    f = features(df)
    trig = classic_triggers(f)
    low5 = df["low"].rolling(5).min()
    sig = pd.DataFrame({s: ev[s] for s in livro.LONGS}, index=df.index)
    st = pd.DataFrame({s: stops[s] for s in livro.LONGS}, index=df.index)
    for name, col in zip(CLASSIC, ("tendencia", "rompimento", "bollinger", "fibonacci"), strict=True):
        sig[name] = trig[col].fillna(False).astype(bool)
        st[name] = low5
    exits = ev[livro.EXITS].any(axis=1).to_numpy()
    return sig.fillna(False).astype(bool), st, exits


def daily_gate(daily: pd.DataFrame, regime: pd.Series) -> pd.Series:
    d = daily[["open", "high", "low", "close", "volume"]]
    return pd.Series(classic_macro(features(d), align_daily(regime, d.index)), index=d.index)


def gate_for(index: pd.DatetimeIndex, tf: str, gate: pd.Series) -> np.ndarray:
    """The daily gate as known when the candle closes: the last *completed* day at or before the close."""
    close = index + TF_DELTA[tf]
    known = (close - pd.Timedelta(days=1)).normalize()
    out: np.ndarray = gate.reindex(known, method="ffill").fillna(False).to_numpy(dtype=bool)
    return out


# ---------------------------------------------------------------------- candidates

def _key(tf: str, kind: str, members: tuple[str, ...], macro: bool, tgt: float) -> str:
    return f"{tf}|{kind}|{'+'.join(members)}|macro={macro}|alvo={tgt:g}R"


def stage1(tf: str) -> list[dict[str, Any]]:
    return [{"key": _key(tf, "individual", (s,), m, t), "tf": tf, "tipo": "individual", "membros": (s,), "macro": m,
             "alvo_r": t} for s in SIGNALS for m in (False, True) for t in TARGETS]


def stage2(tf: str, best: list[str]) -> list[dict[str, Any]]:
    out = []
    for m in (False, True):
        for t in TARGETS:
            for a, b in itertools.combinations(best, 2):
                out.append({"key": _key(tf, "par", (a, b), m, t), "tf": tf, "tipo": "par", "membros": (a, b),
                            "macro": m, "alvo_r": t})
            for k in (2, 3):
                out.append({"key": _key(tf, f"≥{k}", ("todas",), m, t), "tf": tf, "tipo": f"≥{k}",
                            "membros": SIGNALS, "kmin": k, "macro": m, "alvo_r": t})
    return out


def entries_for(sig: pd.DataFrame, recent: np.ndarray, cand: dict[str, Any]) -> np.ndarray:
    cols = list(sig.columns)
    if cand["tipo"] == "individual":
        single: np.ndarray = sig[cand["membros"][0]].to_numpy()
        return single
    if cand["tipo"] == "par":
        a, b = cand["membros"]
        ia, ib = cols.index(a), cols.index(b)
        pair: np.ndarray = (sig[a].to_numpy() & (recent[:, ib] > 0)) | (sig[b].to_numpy() & (recent[:, ia] > 0))
        return pair
    out: np.ndarray = sig.to_numpy().any(axis=1) & (recent.sum(axis=1) >= cand["kmin"])
    return out


def stop_for(st: pd.DataFrame, recent: np.ndarray, members: tuple[str, ...], cols: list[str]) -> np.ndarray:
    stop = np.full(len(st), np.nan)
    for mbr in members:
        has = np.isnan(stop) & (recent[:, cols.index(mbr)] > 0)
        stop[has] = st[mbr].to_numpy()[has]
    return stop


# ---------------------------------------------------------------------- study

def _stats(tr: pd.DataFrame) -> dict[str, Any]:
    if tr is None or not len(tr):
        return {"trades": 0}
    r = tr["ret"].to_numpy(dtype=float)
    wins, losses = r[r > 0], r[r <= 0]
    loss = float(-losses.sum())
    return {"trades": int(r.size), "acerto": round(float((r > 0).mean()), 3),
            "payoff": round(float(wins.mean() / -losses.mean()), 2) if wins.size and losses.size else None,
            "pf": round(min(float(wins.sum()) / loss, 99.0), 2) if loss > 0 else 99.0,
            "expectativa_pct": round(float(r.mean()) * 100, 3),
            "expectativa_R": round(float((r / tr["stop_pct"].to_numpy(dtype=float)).mean()), 3)}


def account(tr: pd.DataFrame, rules: RiskRules, costs: Costs, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    from mqmas.research.pattern_portfolio import equity_monthly

    res = simulate_portfolio(tr, rules, costs, start, end)
    m = res.metrics
    monthly = equity_monthly(res.equity, rules.equity0, end)
    weeks = max((end - start).days / 7, 1e-9)
    return {"trades": int(m["trades"]), "por_semana": round(m["trades"] / weeks, 2), "pulados": res.skipped,
            "acerto": round(m["win_rate"], 3), "pf": round(m["profit_factor"], 2),
            "lucro_usd": round(m["total_return"] * rules.equity0, 2),
            "media_mensal_pct": round(float(monthly.mean()) * 100, 2) if len(monthly) else None,
            "meses_positivos": round(float((monthly > 0).mean()), 2) if len(monthly) else None,
            "max_dd": round(m["max_drawdown"], 4)}


def run_tf(lab: Lab, tf: str, costs: Costs, regime: pd.Series, frame_fn: Callable[..., Any] | None = None,
           cands: list[dict[str, Any]] | None = None) -> dict[str, pd.DataFrame]:
    """Trade tables per candidate. Default: stage 1 + stage 2 on ``signal_frame``; or a given frame and list."""
    frame_fn = frame_fn or signal_frame
    base, rule = TFS[tf]
    frames = []
    for coin, raw in lab.data[base].items():
        df = raw[["open", "high", "low", "close", "volume"]]
        df = df[df.index >= pd.Timestamp("2021-10-01", tz="UTC")]
        if rule:
            df = resample(df, rule)
        if len(df) < WARMUP[tf] + 50:
            continue
        sig, st, exits = frame_fn(df)
        months = df.index.tz_localize(None).to_period("M")
        elig = np.array([coin in lab.eligible.get(mm, ()) for mm in months], dtype=bool)
        elig[: WARMUP[tf]] = False
        gate = gate_for(df.index, tf, daily_gate(lab.data["1d"][coin], regime)) if coin in lab.data["1d"] else \
            np.zeros(len(df), dtype=bool)
        recent = sig.astype(float).rolling(livro.CONFIRM, min_periods=1).max().to_numpy()
        frames.append((coin, df, sig, st, exits, elig, gate, recent, features(df)["atr"].to_numpy()))
    log.info("arena %s: %d coins prepared", tf, len(frames))

    def table(cands: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
        rows: dict[str, list[pd.DataFrame]] = {c["key"]: [] for c in cands}
        for coin, df, sig, st, exits, elig, gate, recent, atr in frames:
            o, h, lo, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
            cols = list(sig.columns)
            for cand in cands:
                ent = entries_for(sig, recent, cand) & elig
                if cand["macro"]:
                    ent = ent & gate
                if not ent.any():
                    continue
                stop = stop_for(st, recent, cand["membros"], cols)
                t = manage(o, h, lo, c, atr, ent, stop, cand["alvo_r"], exits, costs, HOLD[tf])
                if len(t):
                    rows[cand["key"]].append(pd.DataFrame({
                        "coin": coin, "t_in": df.index[t[:, 0].astype(int)],
                        "t_out": df.index[t[:, 1].astype(int)] + TF_DELTA[tf], "ret": t[:, 2], "stop_pct": t[:, 3],
                        "reason": t[:, 4]}))
        empty = pd.DataFrame(columns=["coin", "t_in", "t_out", "ret", "stop_pct", "reason"])
        return {k: (pd.concat(v, ignore_index=True) if v else empty) for k, v in rows.items()}

    if cands is not None:
        return table(cands)
    c1 = stage1(tf)
    t1 = table(c1)
    ranked: dict[str, float] = {}
    for c in c1:
        p1 = t1[c["key"]]
        p1 = p1[p1["t_in"] < FIRST_TEST] if len(p1) else p1
        s = _stats(p1)
        if s.get("trades", 0) >= 30:
            sym = c["membros"][0]
            ranked[sym] = max(ranked.get(sym, 0.0), float(s["pf"]))
    best = [s for s, _ in sorted(ranked.items(), key=lambda x: -x[1])[:6]]
    c2 = stage2(tf, best)
    t2 = table(c2)
    log.info("arena %s: %d + %d candidates", tf, len(c1), len(c2))
    return {**t1, **t2}


def run(root: Path, ex: Any = None) -> dict[str, Any]:
    lab = Lab(root)
    lab.load(ex)
    lab.data["1d"] = {k: v for k, v in lab.data["1d"].items()}
    costs = Costs()
    regime = btc_regime(lab.btc_daily)
    out: dict[str, Any] = {"version": VERSION, "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
                           "periodo_oos": f"{FIRST_TEST.date()} → {lab.end.date()}", "faixa_acerto": HIT_BAND,
                           "timeframes": {}}
    picks_all = []
    for tf in TFS:
        tables = run_tf(lab, tf, costs, regime)
        usable = {k: v for k, v in tables.items() if len(v) >= 30}
        rows: list[dict[str, Any]] = []
        for k, v in tables.items():
            p1, p2 = (v[v["t_in"] < FIRST_TEST], v[v["t_in"] >= FIRST_TEST]) if len(v) else (v, v)
            rows.append({"key": k, "total": _stats(v), "p1": _stats(p1), "p2": _stats(p2)})
        survivors = [r for r in rows if r["p2"].get("trades", 0) >= 40 and r["p1"].get("trades", 0) >= 20
                     and HIT_BAND[0] <= r["p2"]["acerto"] <= HIT_BAND[1] and r["p2"]["pf"] >= 1.3
                     and r["p1"]["pf"] >= 1.2]
        survivors.sort(key=lambda r: -r["p2"]["pf"])
        wf: dict[str, Any] = lab.walk_forward({"arena": list(usable)}, usable)["arena"] if usable else {}
        oos = wf.get("oos_trades")
        res: dict[str, Any] = {"candidatos": len(tables), "sobreviventes": survivors[:12],
                               "n_sobreviventes": len(survivors)}
        if oos is not None and len(oos):
            picks_all.append(oos.assign(tf=tf))
            sr = simulate_portfolio(oos, lab.rules, costs, FIRST_TEST, lab.end).metrics
            res["walk_forward"] = {
                "escolhas": [{"janela": p["window"], "setup": p["config"]} for p in wf["picks"]],
                "oos": _stats(oos),
                "dsr": round(float(deflated_sharpe(sr["daily_sr"], sr["days"], sr["daily_skew"], sr["daily_kurt"],
                                                   wf["trial_srs"])) if sr["days"] > 2 else 0.0, 3),
                "contas": {lbl: account(oos, rules, costs, FIRST_TEST, lab.end) for lbl, rules in ACCOUNTS.items()}}
        out["timeframes"][tf] = res
        log.info("arena %s done: %d survivors", tf, len(survivors))
        del tables, usable
    if picks_all:
        book = pd.concat(picks_all, ignore_index=True)
        good = [tf for tf, r in out["timeframes"].items() if r.get("walk_forward") and
                HIT_BAND[0] <= r["walk_forward"]["oos"]["acerto"] <= HIT_BAND[1] and r["walk_forward"]["oos"]["pf"] > 1.1]
        out["carteira_todos"] = {lbl: account(book, rules, costs, FIRST_TEST, lab.end) for lbl, rules in ACCOUNTS.items()}
        sel = book[book["tf"].isin(good)]
        out["carteira_filtrada"] = {"timeframes": good, "nota": "timeframes escolhidos após ver o walk-forward",
                                    "contas": {lbl: account(sel, rules, costs, FIRST_TEST, lab.end)
                                               for lbl, rules in ACCOUNTS.items()} if len(sel) else {}}
    return out


def summarize(rep: dict[str, Any]) -> list[str]:
    lines = [f"ARENA v{rep['version']} | OOS {rep['periodo_oos']} | faixa de acerto {rep['faixa_acerto']}"]

    def acc(a: dict[str, Any]) -> str:
        return (f"n={a['trades']} ({a['por_semana']}/sem, pulados {a['pulados']}) acerto {a['acerto']:.0%} PF {a['pf']} "
                f"lucro US${a['lucro_usd']:+.2f} mês médio {a['media_mensal_pct']}% meses+ {a['meses_positivos']} "
                f"DD {a['max_dd']:.1%}")

    for tf, r in rep["timeframes"].items():
        wf = r.get("walk_forward")
        lines.append(f"ARENA {tf}: {r['candidatos']} candidatos, {r['n_sobreviventes']} com acerto 40–60% e PF≥1,3 "
                     f"nos dois períodos" + (f" | WF OOS {wf['oos']} DSR {wf['dsr']}" if wf else ""))
        for s in r["sobreviventes"][:6]:
            lines.append(f"ARENA {tf} sobrevivente {s['key']} | p1 n={s['p1']['trades']} acerto {s['p1']['acerto']:.0%} "
                         f"PF {s['p1']['pf']} → p2 n={s['p2']['trades']} acerto {s['p2']['acerto']:.0%} PF {s['p2']['pf']} "
                         f"exp {s['p2']['expectativa_R']}R")
        if wf:
            for p in wf["escolhas"]:
                lines.append(f"ARENA {tf} escolha {p['janela']}: {p['setup']}")
            for lbl, a in wf["contas"].items():
                lines.append(f"ARENA {tf} conta {lbl}: {acc(a)}")
    for lbl, a in rep.get("carteira_todos", {}).items():
        lines.append(f"ARENA CARTEIRA todos os timeframes | {lbl}: {acc(a)}")
    cf = rep.get("carteira_filtrada", {})
    for lbl, a in cf.get("contas", {}).items():
        lines.append(f"ARENA CARTEIRA {cf['timeframes']} | {lbl}: {acc(a)}")
    return lines
