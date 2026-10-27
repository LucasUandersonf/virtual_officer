"""Does each technical-analysis signal actually work in crypto? An event study over years of data.

For every candle pattern, chart pattern, market-structure event and indicator signal, on 4h and
daily bars of the liquid universe:

* **direction** — the forward return after the signal, *minus the coin's average forward return*
  (so a bullish signal does not look good just because crypto went up), signed by the signal's
  bias; measured in two separate periods and called stable only with the same sign and |t| ≥ 2 in both;
* **tradability** — bullish signals traded on spot with the book's bracket (entry next open, stop
  1.5 ATR, target 2R, time stop), net of 0.30 % costs; bearish signals are scored as *exit*
  signals: the loss they would have avoided.

Verdicts: ``funciona`` (stable edge that pays costs), ``direção certa, não paga custos``,
``sem efeito``, ``funciona ao contrário`` (stable edge in the opposite direction).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mqmas.research.engine import Costs
from mqmas.research.lab import FIRST_TEST
from mqmas.research.seasonality import load_cached, split_stats
from mqmas.research.swing import resample
from mqmas.research.templates import Plan, structured_trades
from mqmas.ta.signals import all_events, category, is_bear, is_bull

log = logging.getLogger(__name__)

VERSION = 5
HORIZONS = {"4h": (1, 3, 6, 12), "1d": (1, 3, 5, 10)}
HOLD = {"4h": 12, "1d": 10}
HEADLINE = {"4h": 6, "1d": 5}  # horizon used for the verdict: 1 day on 4h bars, 1 week on daily bars


def _verdict(dir_stats: dict[str, Any], trade: dict[str, Any] | None, bull: bool) -> str:
    if dir_stats["stable"] and dir_stats["mean_bps_p1"] < 0:
        return "funciona ao contrário"
    if not dir_stats["stable"]:
        return "sem efeito"
    if bull and trade and trade["pf_p1"] > 1.1 and trade["pf_p2"] > 1.1:
        return "funciona"
    if not bull and dir_stats["pays_costs"]:
        return "funciona"
    return "direção certa, não paga custos"


def _pf(r: np.ndarray) -> float:
    loss = float(-r[r <= 0].sum())
    gain = float(r[r > 0].sum())
    return min(gain / loss, 99.0) if loss > 0 else (99.0 if r.size else 0.0)


def run(root: Path, events_fn: Any = None) -> dict[str, Any]:
    events_fn = events_fn or all_events
    h1 = load_cached(root)
    costs = Costs()
    store: dict[tuple[str, str], list[pd.Series]] = {}
    trades: dict[tuple[str, str], list[pd.DataFrame]] = {}
    counts: dict[tuple[str, str], int] = {}
    raw: dict[tuple[str, str], list[pd.Series]] = {}
    for coin, d in h1.items():
        for tf in ("4h", "1d"):
            df = resample(d, "4h", 4) if tf == "4h" else resample(d, "1D", 24)
            df = df[["open", "high", "low", "close", "volume"]]
            ev, f, _ = events_fn(df)
            c = df["close"]
            o, hi, lo, cl = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
            a = f["atr"].to_numpy()
            fwd = {k: np.log(c.shift(-k) / c) for k in HORIZONS[tf]}
            base = {k: v.mean() for k, v in fwd.items()}
            warm = np.arange(len(df)) >= 200
            for name in ev.columns:
                mask = ev[name].to_numpy() & warm
                if not mask.any():
                    continue
                sign = -1.0 if is_bear(name) else 1.0
                for k in HORIZONS[tf]:
                    store.setdefault((name, f"{tf}|{k}"), []).append(sign * (fwd[k][mask] - base[k]))
                    # what most websites report: the raw move, no drift removed, no costs
                    raw.setdefault((name, f"{tf}|{k}"), []).append(sign * fwd[k][mask])
                counts[(name, tf)] = counts.get((name, tf), 0) + int(mask.sum())
                if is_bull(name):
                    idx = np.flatnonzero(mask)
                    t = structured_trades(o, hi, lo, cl, a, Plan(idx, np.full(idx.size, np.nan), target_r=2.0,
                                                                 stop_atr=1.5, hold=HOLD[tf]), costs)
                    if t["entry"].size:
                        trades.setdefault((name, tf), []).append(
                            pd.DataFrame({"t": df.index[t["entry"].astype(int)], "ret": t["ret"]}))
        log.info("ta study: %s done", coin)
    events: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    names = sorted({n for n, _ in counts})
    for name in names:
        for tf in ("4h", "1d"):
            if (name, tf) not in counts:
                continue
            per_h = {}
            for k in HORIZONS[tf]:
                s = pd.concat(store[(name, f"{tf}|{k}")]).dropna().sort_index()
                st = split_stats(s)
                st["hit"] = round(float((s > -0.0).mean()), 3)
                per_h[k] = st
            head = per_h[HEADLINE[tf]]
            r_raw = pd.concat(raw[(name, f"{tf}|{HEADLINE[tf]}")]).dropna()
            raw_view = {"acerto_bruto": round(float((r_raw > 0).mean()), 3),
                        "movimento_medio_bruto_bps": round(float(r_raw.mean()) * 1e4, 1)}
            trade = None
            if (name, tf) in trades:
                tr = pd.concat(trades[(name, tf)])
                r1 = tr[tr["t"] < FIRST_TEST]["ret"].to_numpy()
                r2 = tr[tr["t"] >= FIRST_TEST]["ret"].to_numpy()
                allr = tr["ret"].to_numpy()
                gross = allr + costs.round_trip
                trade = {"trades": int(allr.size), "win_rate": round(float((allr > 0).mean()), 3),
                         "pf_antes_taxas": round(_pf(gross), 2), "acerto_antes_taxas": round(float((gross > 0).mean()), 3),
                         "expectancy_pct": round(float(allr.mean()) * 100, 3), "pf": round(_pf(allr), 2),
                         "pf_p1": round(_pf(r1), 2), "pf_p2": round(_pf(r2), 2)}
            bull = is_bull(name)
            verdict = _verdict(head, trade, bull)
            row = {"padrao": name, "tf": tf, "categoria": category(name),
                   "vies": "alta" if bull else ("baixa" if is_bear(name) else "neutro"), "ocorrencias": counts[(name, tf)],
                   "horizontes": per_h, "operacao": trade, "veredito": verdict, "visao_bruta": raw_view}
            rows.append(row)
            events.setdefault(name, {})[tf] = {
                "veredito": verdict, "ocorrencias": counts[(name, tf)],
                "excesso_bps": round((head["mean_bps_p1"] + head["mean_bps_p2"]) / 2, 1), "acerto": head["hit"],
                "acerto_bruto": raw_view["acerto_bruto"],
                "pf_operacao": trade["pf"] if trade else None}
    order = {"funciona": 0, "funciona ao contrário": 1, "direção certa, não paga custos": 2, "sem efeito": 3}
    rows.sort(key=lambda r: (order[r["veredito"]], -abs(r["horizontes"][HEADLINE[r["tf"]]]["mean_bps_p2"])))
    return {"version": VERSION, "generated_at": pd.Timestamp.now(tz="UTC").isoformat(), "coins": list(h1),
            "first_test": str(FIRST_TEST.date()), "rows": rows, "events": events}


def summarize(rep: dict[str, Any]) -> list[str]:
    tally: dict[str, int] = {}
    for r in rep["rows"]:
        tally[r["veredito"]] = tally.get(r["veredito"], 0) + 1
    lines = [f"TA STUDY v{rep['version']} | {len(rep['coins'])} moedas | {len(rep['rows'])} sinais×tempo | {tally}"]
    for r in rep["rows"]:
        h = r["horizontes"][HEADLINE[r["tf"]]]
        op = r["operacao"]
        lines.append(
            f"{r['veredito']:<31} {r['padrao']:<30} {r['tf']} {r['vies']:<6} n={r['ocorrencias']:<6} "
            f"excesso p1 {h['mean_bps_p1']:+.0f}bps (t {h['t_p1']:+.1f}) p2 {h['mean_bps_p2']:+.0f}bps (t {h['t_p2']:+.1f}) "
            f"acerto {h['hit']:.0%} bruto {r['visao_bruta']['acerto_bruto']:.0%}"
            + (f" | trade n={op['trades']} win {op['win_rate']:.0%} PF {op['pf']} antes-taxa PF {op['pf_antes_taxas']} "
                                       f"(p1 {op['pf_p1']} p2 {op['pf_p2']}) exp {op['expectancy_pct']:+.2f}%"
                                       if op else ""))
    return lines

