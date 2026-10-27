"""TA-Lib in the arena: its 61 candlestick patterns tested like every other rule, and our indicators audited.

TA-Lib (C library, BSD) recognises 61 candle patterns (CDL*): +100/+200 = bullish, −100/−200 = bearish.
A pattern "fires" when TA-Lib returns a positive value on the closed candle; the order goes in at the next
open with the library's management (initial stop = farther of the low of the last 5 candles and 1.5 ATR,
cap 3 ATR; breakeven at +1R; trailing from +2R; our bearish exit patterns; time stop; 0.30 % round trip).

Swing timeframes only (1d, 2d, 3d): day trade already failed on costs. Per timeframe and pattern:
alone (with and without the macro gate) and confirming the classic breakout within 3 candles, with no
target (trailing only, the book's policy) and with a 2R target. Same bar as the arena: survivors need
40–60 % hit rate and PF ≥ 1.3 after 2023-07 (≥ 40 trades) and PF ≥ 1.2 before; the walk-forward picks
blind among the TA-Lib candidates every 6 months.

Also an audit: our EMA, RSI, ATR and Bollinger against TA-Lib's on the same candles.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mqmas.research import arena
from mqmas.research.engine import Costs, simulate_portfolio
from mqmas.research.lab import FIRST_TEST, Lab, deflated_sharpe
from mqmas.research.strategies import atr, btc_regime, ema, rsi

log = logging.getLogger(__name__)

VERSION = 1
TFS = ("1d", "2d", "3d")
TARGETS = (0.0, 2.0)
PARTNER = "cl_rompimento"


def patterns() -> list[str]:
    import talib

    groups: dict[str, list[str]] = talib.get_function_groups()  # type: ignore[no-untyped-call]
    return sorted(groups["Pattern Recognition"])


def candles(df: pd.DataFrame) -> pd.DataFrame:
    """Bullish fire of every TA-Lib pattern on each closed candle."""
    import talib

    o, h, lo, c = (df[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    return pd.DataFrame({f"cdl_{n[3:].lower()}": getattr(talib, n)(o, h, lo, c) > 0 for n in patterns()},
                        index=df.index)


def frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    sig, st, exits = arena.signal_frame(df)
    cdl = candles(df)
    low5 = df["low"].rolling(5).min()
    return (pd.concat([sig, cdl], axis=1),
            pd.concat([st, pd.DataFrame({k: low5 for k in cdl.columns}, index=df.index)], axis=1), exits)


def candidates(tf: str) -> list[dict[str, Any]]:
    names = [f"cdl_{n[3:].lower()}" for n in patterns()]
    out = [{"key": arena._key(tf, "individual", (PARTNER,), False, t), "tf": tf, "tipo": "individual",
            "membros": (PARTNER,), "macro": False, "alvo_r": t, "referencia": True} for t in TARGETS]
    for n in names:
        for t in TARGETS:
            for m in (False, True):
                out.append({"key": arena._key(tf, "individual", (n,), m, t), "tf": tf, "tipo": "individual",
                            "membros": (n,), "macro": m, "alvo_r": t})
            out.append({"key": arena._key(tf, "par", (PARTNER, n), False, t), "tf": tf, "tipo": "par",
                        "membros": (PARTNER, n), "macro": False, "alvo_r": t})
    return out


def audit(df: pd.DataFrame) -> dict[str, float]:
    """Largest relative gap between our indicator and TA-Lib's after a 250-candle warm-up."""
    import talib

    c, h, lo = (df[x].to_numpy(dtype=float) for x in ("close", "high", "low"))
    up, _, dn = talib.BBANDS(c, 20, 2.0, 2.0, talib.MA_Type.SMA)  # type: ignore[attr-defined]
    mid, sd = df["close"].rolling(20).mean(), df["close"].rolling(20).std(ddof=0)
    pairs = {"EMA20": (ema(df["close"], 20).to_numpy(), talib.EMA(c, 20)),
             "EMA200": (ema(df["close"], 200).to_numpy(), talib.EMA(c, 200)),
             "RSI14": (rsi(df["close"], 14).to_numpy(), talib.RSI(c, 14)),
             "ATR14": (atr(df).to_numpy(), talib.ATR(h, lo, c, 14)),
             "Bollinger sup": ((mid + 2 * sd).to_numpy(), up), "Bollinger inf": ((mid - 2 * sd).to_numpy(), dn)}
    out = {}
    for k, (a, b) in pairs.items():
        a, b = a[250:], b[250:]
        ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 0)
        out[k] = round(float(np.max(np.abs(a[ok] / b[ok] - 1))) * 100, 4) if ok.any() else float("nan")
    return out


def run(root: Path, ex: Any = None) -> dict[str, Any]:
    lab = Lab(root)
    lab.load(ex)
    costs = Costs()
    regime = btc_regime(lab.btc_daily)
    out: dict[str, Any] = {"version": VERSION, "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
                           "periodo_oos": f"{FIRST_TEST.date()} → {lab.end.date()}", "padroes": len(patterns()),
                           "auditoria_indicadores_pct": audit(lab.btc_daily), "timeframes": {}}
    for tf in TFS:
        cands = candidates(tf)
        tables = arena.run_tf(lab, tf, costs, regime, frame_fn=frame, cands=cands)
        rows = []
        for c in cands:
            v = tables[c["key"]]
            p1, p2 = (v[v["t_in"] < FIRST_TEST], v[v["t_in"] >= FIRST_TEST]) if len(v) else (v, v)
            rows.append({"key": c["key"], "referencia": c.get("referencia", False), "total": arena._stats(v),
                         "p1": arena._stats(p1), "p2": arena._stats(p2)})
        ref = [r for r in rows if r["referencia"]]
        survivors = [r for r in rows if not r["referencia"] and r["p2"].get("trades", 0) >= 40
                     and r["p1"].get("trades", 0) >= 20 and arena.HIT_BAND[0] <= r["p2"]["acerto"] <= arena.HIT_BAND[1]
                     and r["p2"]["pf"] >= 1.3 and r["p1"]["pf"] >= 1.2]
        survivors.sort(key=lambda r: -r["p2"]["pf"])
        usable = {c["key"]: tables[c["key"]] for c in cands if not c.get("referencia") and len(tables[c["key"]]) >= 30}
        res: dict[str, Any] = {"candidatos": len(cands) - len(ref), "com_trades": sum(r["total"].get("trades", 0) > 0
                                                                                     for r in rows),
                               "referencia_rompimento": ref, "n_sobreviventes": len(survivors),
                               "sobreviventes": survivors[:12]}
        if usable:
            wf = lab.walk_forward({"talib": list(usable)}, usable)["talib"]
            oos = wf.get("oos_trades")
            if oos is not None and len(oos):
                sr = simulate_portfolio(oos, lab.rules, costs, FIRST_TEST, lab.end).metrics
                res["walk_forward"] = {
                    "escolhas": [{"janela": p["window"], "setup": p["config"]} for p in wf["picks"]],
                    "oos": arena._stats(oos),
                    "dsr": round(float(deflated_sharpe(sr["daily_sr"], sr["days"], sr["daily_skew"], sr["daily_kurt"],
                                                       wf["trial_srs"])) if sr["days"] > 2 else 0.0, 3)}
        out["timeframes"][tf] = res
        log.info("talib %s done: %d survivors", tf, len(survivors))
        del tables, usable
    return out


def summarize(rep: dict[str, Any]) -> list[str]:
    lines = [f"TALIB v{rep['version']} | {rep['padroes']} padrões de vela | OOS {rep['periodo_oos']} | "
             f"diferença máx. dos nossos indicadores vs TA-Lib (%): {rep['auditoria_indicadores_pct']}"]
    for tf, r in rep["timeframes"].items():
        wf = r.get("walk_forward")
        lines.append(f"TALIB {tf}: {r['candidatos']} candidatos ({r['com_trades']} com trades), "
                     f"{r['n_sobreviventes']} sobreviventes" + (f" | WF cego OOS {wf['oos']} DSR {wf['dsr']}" if wf else ""))
        for s in r["referencia_rompimento"]:
            lines.append(f"TALIB {tf} referência {s['key']} | p2 {s['p2']}")
        for s in r["sobreviventes"][:8]:
            lines.append(f"TALIB {tf} sobrevivente {s['key']} | p1 n={s['p1']['trades']} acerto {s['p1']['acerto']:.0%} "
                         f"PF {s['p1']['pf']} → p2 n={s['p2']['trades']} acerto {s['p2']['acerto']:.0%} "
                         f"PF {s['p2']['pf']} exp {s['p2']['expectativa_R']}R")
        if wf:
            for p in wf["escolhas"]:
                lines.append(f"TALIB {tf} escolha {p['janela']}: {p['setup']}")
    return lines
