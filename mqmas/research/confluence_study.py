"""Does the confluence score (mqmas.ta.confluence) make money? Out-of-sample test.

Pattern weights are taken from the TA study's *first period* only; the threshold is chosen on the
first period too; the second period (from FIRST_TEST) is then the verdict, never seen before.
Every entry follows the live execution rule: next open, stop 1.5 ATR, target 2R, time stop, 0.30 %
round-trip costs. To be validated the score must also beat a baseline (entering every 10 bars
regardless of score), i.e. the part of the result that is just the market drifting up.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mqmas.research.engine import Costs
from mqmas.research.lab import FIRST_TEST
from mqmas.research.seasonality import load_cached
from mqmas.research.swing import resample
from mqmas.research.ta_study import HEADLINE, _pf
from mqmas.research.templates import Plan, structured_trades
from mqmas.ta.confluence import RULE, entries, score_frame, weights_from_first_period
from mqmas.ta.signals import all_events

log = logging.getLogger(__name__)

VERSION = 1
THRESHOLDS = (40, 50, 60, 70, 80)
MIN_TRADES = 40


def _stats(r: np.ndarray) -> dict[str, Any]:
    return {"trades": int(r.size), "acerto": round(float((r > 0).mean()), 3) if r.size else None,
            "pf": round(_pf(r), 2) if r.size else None,
            "expectativa_pct": round(float(r.mean()) * 100, 3) if r.size else None}


def _trade(df: pd.DataFrame, f: pd.DataFrame, idx: np.ndarray, tf: str, costs: Costs) -> pd.DataFrame:
    o, hi, lo, cl = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    t = structured_trades(o, hi, lo, cl, f["atr"].to_numpy(), Plan(idx, np.full(idx.size, np.nan),
                          target_r=RULE.target_r, stop_atr=RULE.stop_atr, hold=RULE.hold_for(tf)), costs)
    return pd.DataFrame({"t": df.index[t["entry"].astype(int)], "ret": t["ret"]})


def run(root: Path, events_fn: Any = None, study: dict[str, Any] | None = None) -> dict[str, Any]:
    events_fn = events_fn or all_events
    if study is None:
        path = root / "ta_report.json"
        study = json.loads(path.read_text()) if path.exists() else {}
    h1 = load_cached(root)
    costs = Costs()
    acc: dict[tuple[str, Any], list[pd.DataFrame]] = {}
    for coin, d in h1.items():
        for tf in ("4h", "1d"):
            df = resample(d, "4h", 4) if tf == "4h" else resample(d, "1D", 24)
            df = df[["open", "high", "low", "close", "volume"]]
            ev, f, _ = events_fn(df)
            sf = score_frame(df, ev, f, weights_from_first_period(study, tf, HEADLINE[tf]))
            sf.iloc[:200] = 0.0  # indicator warm-up
            for th in THRESHOLDS:
                idx = entries(sf["score"], th)
                if idx.size:
                    acc.setdefault((tf, th), []).append(_trade(df, f, idx, tf, costs))
            acc.setdefault((tf, "base"), []).append(_trade(df, f, np.arange(200, len(df), 10), tf, costs))
        log.info("confluence study: %s done", coin)
    out: dict[str, Any] = {"version": VERSION, "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
                           "coins": list(h1), "first_test": str(FIRST_TEST.date()), "tf": {}}
    for tf in ("4h", "1d"):
        rows: list[dict[str, Any]] = []
        keys: tuple[int | str, ...] = (*THRESHOLDS, "base")
        for key in keys:
            tr = pd.concat(acc[(tf, key)]) if acc.get((tf, key)) else pd.DataFrame(
                {"t": pd.DatetimeIndex([], tz="UTC"), "ret": np.array([], dtype=float)})
            r1 = tr[tr["t"] < FIRST_TEST]["ret"].to_numpy(dtype=float)
            r2 = tr[tr["t"] >= FIRST_TEST]["ret"].to_numpy(dtype=float)
            rows.append({"limiar": key, "periodo1": _stats(r1), "periodo2": _stats(r2)})
        base = rows[-1]  # entering every 10 bars: what the market's drift alone gives
        cands = [r for r in rows if r["limiar"] != "base" and r["periodo1"]["trades"] >= MIN_TRADES]
        best = max(cands, key=lambda r: (r["periodo1"]["pf"] or 0)) if cands else None
        p2 = best["periodo2"] if best else None
        ok = bool(best and best["periodo1"]["pf"] >= 1.1 and p2 and p2["trades"] >= 30 and p2["pf"] >= 1.2
                  and p2["expectativa_pct"] > max(base["periodo2"]["expectativa_pct"] or 0.0, 0.0))
        out["tf"][tf] = {"linhas": rows, "escolhido_no_periodo1": best["limiar"] if best else None,
                         "validado": ok, "limiar": best["limiar"] if ok and best else None}
    return out


def summarize(rep: dict[str, Any]) -> list[str]:
    lines = [f"CONFLUENCE v{rep['version']} | {len(rep['coins'])} moedas | teste fora da amostra desde {rep['first_test']}"]
    for tf, v in rep["tf"].items():
        lines.append(f"CONFLUENCE {tf}: limiar escolhido no período 1 = {v['escolhido_no_periodo1']} → "
                     f"{'VALIDADO' if v['validado'] else 'NÃO validado'} no período 2")
        for r in v["linhas"]:
            p1, p2 = r["periodo1"], r["periodo2"]
            lines.append(f"  score≥{r['limiar']!s:<4} p1 n={p1['trades']} acerto {p1['acerto']} PF {p1['pf']} "
                         f"exp {p1['expectativa_pct']}% | p2 n={p2['trades']} acerto {p2['acerto']} PF {p2['pf']} "
                         f"exp {p2['expectativa_pct']}%")
    return lines
