"""Confluence engine: one 0–100 score per bar, and the rule that turns it into an order.

The score adds five components, each in [0, 1], with weights fixed *before* any test
(they are never optimised on data, so the validation below is honest):

==============  ====  ==========================================================================
localizacao      25   how many support levels sit just under the price (≤ 1 ATR): prior 20-bar
                      low, EMA20/50/200, lower Bollinger, VWAP, Kijun, top of the Ichimoku cloud
padroes          35   sum of the active patterns (last 3 bars), each weighted by its *measured*
                      historical edge; patterns proven bearish subtract (up to −35)
tendencia        20   share of the six trend votes that are bullish
momento          10   MACD histogram rising; RSI in 50–70 or turning up from below 40
volume           10   OBV above its average; volume above 1.2× its 20-bar mean
==============  ====  ==========================================================================

Pattern weights come from the TA study. Live they use the final verdict (edge stable in both
periods); in the out-of-sample test they use the first period only, so the second period is unseen.

Execution rule (identical in the test and live — what is traded is what was tested):
entry at the next open when the score crosses the validated threshold, stop 1.5 ATR below,
take profit 2R (risk:reward 1:2), time stop; size = risk per trade / stop distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mqmas.ta.signals import is_bear

WEIGHTS = {"localizacao": 25.0, "padroes": 35.0, "tendencia": 20.0, "momento": 10.0, "volume": 10.0}
RECENT = 3
NEAR_ATR = 1.0


@dataclass(frozen=True)
class OrderRule:
    stop_atr: float = 1.5
    target_r: float = 2.0
    hold: dict[str, int] | None = None

    def hold_for(self, tf: str) -> int:
        return (self.hold or {"4h": 18, "1d": 10}).get(tf, 12)


RULE = OrderRule()

_VERDICT_W = {"funciona": 1.0, "funciona ao contrário": -1.0, "direção certa, não paga custos": 0.3}


def weights_from_verdicts(study: dict[str, Any] | None, tf: str) -> dict[str, float]:
    """Bullish weight in [−1, 1] of each pattern on ``tf``, from the study's final verdicts."""
    out: dict[str, float] = {}
    for name, per_tf in (study or {}).get("events", {}).items():
        v = (per_tf.get(tf) or {}).get("veredito")
        w = _VERDICT_W.get(v or "", 0.0)
        if w:
            out[name] = -w if is_bear(name) else w
    return out


def weights_from_first_period(study: dict[str, Any] | None, tf: str, horizon: int) -> dict[str, float]:
    """Same, from the first period only (|t| ≥ 2), so the second period stays out of sample."""
    out: dict[str, float] = {}
    for r in (study or {}).get("rows", []):
        if r["tf"] != tf:
            continue
        h = r["horizontes"].get(horizon) or r["horizontes"].get(str(horizon))
        if not h or abs(h["t_p1"]) < 2:
            continue
        w = float(np.sign(h["mean_bps_p1"])) * min(abs(h["t_p1"]) / 4, 1.0)
        out[r["padrao"]] = -w if is_bear(r["padrao"]) else w
    return out


def score_frame(df: pd.DataFrame, ev: pd.DataFrame, f: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Components and score (0–100) of every bar; uses only data up to that bar."""
    c, lo, a = df["close"], df["low"], f["atr"]
    supports = {
        "fundo de 20 barras": lo.rolling(20).min().shift(1), "EMA20": f["ema20"], "EMA50": f["ema50"],
        "EMA200": f["ema200"], "Bollinger inferior": f["bb_lo"], "VWAP": f["vwap"], "Kijun": f["kijun"],
        "topo da nuvem": f[["span_a", "span_b"]].max(axis=1)}
    near = sum((((c - lv) >= 0) & ((c - lv) <= NEAR_ATR * a)) | ((lo <= lv) & (c >= lv))
               for lv in supports.values())
    loc = np.minimum(near / 3.0, 1.0)

    active = ev.astype(float).rolling(RECENT, min_periods=1).max()
    pat_sum = pd.Series(0.0, index=df.index)
    for name, w in weights.items():
        if name in active.columns:
            pat_sum = pat_sum + w * active[name]
    pat_pos = np.clip(pat_sum, 0, 2) / 2
    pat_neg = np.clip(-pat_sum, 0, 2) / 2

    cloud_top = f[["span_a", "span_b"]].max(axis=1)
    votes = ((f["ema50"] > f["ema200"]).astype(float) + (c > f["ema200"]) + (c > cloud_top) +
             (f["plus_di"] > f["minus_di"]) + (f["macd"] > 0) + (c > f["sar"]))
    trend = votes / 6.0
    rsi = f["rsi"]
    mom = 0.5 * (f["macd_hist"] > f["macd_hist"].shift(1)) + \
        0.5 * (((rsi >= 50) & (rsi <= 70)) | ((rsi.shift(1) < 40) & (rsi > rsi.shift(1))))
    vol = 0.5 * (f["obv"] > f["obv_ema"]) + 0.5 * (df["volume"] > 1.2 * df["volume"].rolling(20).mean())

    out = pd.DataFrame({"localizacao": loc, "padroes": pat_pos, "tendencia": trend, "momento": mom.astype(float),
                        "volume": vol.astype(float)}, index=df.index).fillna(0.0)
    score = sum(WEIGHTS[k] * out[k] for k in WEIGHTS) - WEIGHTS["padroes"] * pat_neg.fillna(0.0)
    out["penalidade"] = pat_neg.fillna(0.0)
    out["score"] = np.clip(score, 0, 100)
    out["suportes_perto"] = near
    return out


def entries(score: pd.Series, threshold: float) -> np.ndarray:
    """Bars where the score crosses the threshold from below (one entry per episode)."""
    s = score.to_numpy()
    prev = np.r_[0.0, s[:-1]]
    return np.flatnonzero((s >= threshold) & (prev < threshold))


def live_score(df: pd.DataFrame, study: dict[str, Any] | None, tf: str) -> dict[str, Any]:
    from mqmas.ta.signals import all_events

    ev, f, _ = all_events(df)
    sf = score_frame(df, ev, f, weights_from_verdicts(study, tf))
    last, prev = sf.iloc[-1], sf.iloc[-2]
    w = weights_from_verdicts(study, tf)
    recent = ev.iloc[-RECENT:]
    counted = [{"padrao": k, "peso": round(w[k], 2)} for k in w if k in ev.columns and bool(recent[k].any())]
    return {"score": round(float(last["score"]), 1), "score_anterior": round(float(prev["score"]), 1),
            "componentes": {k: round(float(last[k]) * WEIGHTS[k], 1) for k in WEIGHTS},
            "penalidade": round(-float(last["penalidade"]) * WEIGHTS["padroes"], 1),
            "suportes_perto": int(last["suportes_perto"]), "padroes_com_historico": counted,
            "atr": float(f["atr"].iloc[-1]), "preco": float(df["close"].iloc[-1])}


def order_proposal(sc: dict[str, Any], tf: str, validation: dict[str, Any] | None, equity: float,
                   risk_pct: float, price_map: dict[str, Any] | None = None, min_notional: float = 5.0) -> dict[str, Any]:
    """Turn the live score into an executable bracket order (or say why not)."""
    v = (validation or {}).get("tf", {}).get(tf, {})
    th = v.get("limiar")
    price, a = sc["preco"], sc["atr"]
    entry = price
    stop = entry - RULE.stop_atr * a
    risk = entry - stop
    tp = entry + RULE.target_r * risk
    qty = equity * risk_pct / risk if risk > 0 else 0.0
    qty = min(qty, equity / entry) if entry > 0 else 0.0
    order: dict[str, Any] = {
        "lado": "compra", "tipo": "mercado na abertura do próximo candle", "entrada": round(entry, 8),
        "stop": round(stop, 8), "stop_pct": round(risk / entry * 100, 2) if entry else None,
        "alvo": round(tp, 8), "risco_retorno": f"1:{RULE.target_r:g}", "saida_por_tempo_barras": RULE.hold_for(tf),
        "quantidade": round(qty, 8), "valor_usdt": round(qty * entry, 2), "risco_usdt": round(qty * risk, 2)}
    alerts = []
    for z in (price_map or {}).get("resistencias", []):
        if z["forca"] >= 2 and z["preco"] < tp:
            alerts.append(f"resistência de força {z['forca']} em {z['preco']:.6g}, antes do alvo")
            break
    if qty * entry < min_notional:
        alerts.append(f"valor abaixo do mínimo de US${min_notional:g}")
    crossed = th is not None and sc["score"] >= th > sc["score_anterior"]
    if th is None:
        decision, why = "OBSERVAR", "o score ainda não foi validado fora da amostra neste tempo gráfico"
    elif crossed and qty * entry >= min_notional:
        decision, why = "EXECUTAR", f"score {sc['score']} cruzou o limiar validado {th}"
    elif sc["score"] >= th:
        decision, why = "AGUARDAR", f"score {sc['score']} ≥ {th}, mas o cruzamento já aconteceu antes (sem perseguir preço)"
    else:
        decision, why = "NÃO OPERAR", f"score {sc['score']} abaixo do limiar {th}"
    return {"decisao": decision, "motivo": why, "limiar": th, "ordem": order, "alertas": alerts}
