"""Indicator signals as events, and the full technical report of one market.

``all_events(df)`` returns one boolean column per signal (candles, structure, chart patterns and
indicators) — the input of the historical study. ``analyze(df)`` returns what a human analyst would
write about the last closed bar: trend, momentum, volatility, volume, structure, active zones,
recent patterns and a confluence score; when the study report is available, each recent pattern is
annotated with how it actually performed historically.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mqmas.ta.candles import BEARISH, BULLISH, NEUTRAL, candle_patterns
from mqmas.ta.indicators import all_indicators, volume_profile
from mqmas.ta.structure import BEARISH_STRUCT, BULLISH_STRUCT, structure_events

RECENT_BARS = 3


def _xup(a: pd.Series, b: pd.Series | float) -> pd.Series:
    return (a > b) & (a.shift(1) <= (b.shift(1) if isinstance(b, pd.Series) else b))


def _xdn(a: pd.Series, b: pd.Series | float) -> pd.Series:
    return (a < b) & (a.shift(1) >= (b.shift(1) if isinstance(b, pd.Series) else b))


def indicator_events(df: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    cloud_top = f[["span_a", "span_b"]].max(axis=1)
    cloud_bot = f[["span_a", "span_b"]].min(axis=1)
    e = pd.DataFrame(index=df.index)
    e["golden_cross"] = _xup(f["ema50"], f["ema200"])
    e["death_cross"] = _xdn(f["ema50"], f["ema200"])
    e["macd_cruza_alta"] = _xup(f["macd"], f["macd_signal"])
    e["macd_cruza_baixa"] = _xdn(f["macd"], f["macd_signal"])
    e["rsi_sai_sobrevenda"] = _xup(f["rsi"], 30.0)
    e["rsi_sai_sobrecompra"] = _xdn(f["rsi"], 70.0)
    e["estocastico_alta_sobrevenda"] = _xup(f["stoch_k"], f["stoch_d"]) & (f["stoch_k"] < 20)
    e["estocastico_baixa_sobrecompra"] = _xdn(f["stoch_k"], f["stoch_d"]) & (f["stoch_k"] > 80)
    e["cci_sai_de_baixo"] = _xup(f["cci"], -100.0)
    e["cci_sai_de_cima"] = _xdn(f["cci"], 100.0)
    e["roc_vira_positivo"] = _xup(f["roc"], 0.0)
    e["roc_vira_negativo"] = _xdn(f["roc"], 0.0)
    e["adx_tendencia_alta"] = _xup(f["adx"], 25.0) & (f["plus_di"] > f["minus_di"])
    e["adx_tendencia_baixa"] = _xup(f["adx"], 25.0) & (f["minus_di"] > f["plus_di"])
    e["sar_vira_alta"] = _xup(c, f["sar"])
    e["sar_vira_baixa"] = _xdn(c, f["sar"])
    e["ichimoku_rompe_nuvem_alta"] = _xup(c, cloud_top)
    e["ichimoku_perde_nuvem"] = _xdn(c, cloud_bot)
    e["ichimoku_tk_alta"] = _xup(f["tenkan"], f["kijun"]) & (c > cloud_top)
    e["ichimoku_tk_baixa"] = _xdn(f["tenkan"], f["kijun"]) & (c < cloud_bot)
    e["bollinger_reentrada_alta"] = (c > f["bb_lo"]) & (c.shift(1) < f["bb_lo"].shift(1))
    e["bollinger_rompe_topo"] = _xup(c, f["bb_up"])
    e["ttm_squeeze_dispara_alta"] = f["squeeze_fired"] & (f["squeeze_mom"] > 0)
    e["ttm_squeeze_dispara_baixa"] = f["squeeze_fired"] & (f["squeeze_mom"] < 0)
    e["obv_cruza_alta"] = _xup(f["obv"], f["obv_ema"])
    e["obv_cruza_baixa"] = _xdn(f["obv"], f["obv_ema"])
    e["vwap_recupera"] = _xup(c, f["vwap"])
    e["vwap_perde"] = _xdn(c, f["vwap"])
    return e.fillna(False).astype(bool)


IND_BULLISH = ("golden_cross", "macd_cruza_alta", "rsi_sai_sobrevenda", "estocastico_alta_sobrevenda",
               "cci_sai_de_baixo", "roc_vira_positivo", "adx_tendencia_alta", "sar_vira_alta",
               "ichimoku_rompe_nuvem_alta", "ichimoku_tk_alta", "bollinger_reentrada_alta", "bollinger_rompe_topo",
               "ttm_squeeze_dispara_alta", "obv_cruza_alta", "vwap_recupera")
IND_BEARISH = ("death_cross", "macd_cruza_baixa", "rsi_sai_sobrecompra", "estocastico_baixa_sobrecompra",
               "cci_sai_de_cima", "roc_vira_negativo", "adx_tendencia_baixa", "sar_vira_baixa", "ichimoku_perde_nuvem",
               "ichimoku_tk_baixa", "ttm_squeeze_dispara_baixa", "obv_cruza_baixa", "vwap_perde")
BULL = set(BULLISH) | set(BULLISH_STRUCT) | set(IND_BULLISH)
BEAR = set(BEARISH) | set(BEARISH_STRUCT) | set(IND_BEARISH)
CATEGORY = {**{k: "candle" for k in BULLISH + BEARISH + NEUTRAL},
            **{k: "estrutura" for k in ("bos_alta", "bos_baixa", "choch_alta", "choch_baixa", "ob_alta_reteste",
                                        "ob_baixa_reteste", "fvg_alta_reteste", "fvg_baixa_reteste",
                                        "varredura_fundo", "varredura_topo")},
            **{k: "indicador" for k in IND_BULLISH + IND_BEARISH}}


def all_events(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    f = all_indicators(df)
    struct, state = structure_events(df)
    ev = pd.concat([candle_patterns(df), struct, indicator_events(df, f)], axis=1)
    return ev, f, state


def _num(x: Any, d: int = 2) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, d) if np.isfinite(v) else None


def analyze(df: pd.DataFrame, study: dict[str, Any] | None = None) -> dict[str, Any]:
    """Technical report of the last closed bar of ``df`` (OHLCV, closed candles only)."""
    ev, f, state = all_events(df)
    last, c = f.iloc[-1], float(df["close"].iloc[-1])
    cloud_top = max(last["span_a"], last["span_b"]) if np.isfinite(last["span_a"]) else np.nan
    trend_votes = {
        "EMA50 > EMA200": bool(last["ema50"] > last["ema200"]), "preço > EMA200": bool(c > last["ema200"]),
        "preço acima da nuvem": bool(c > cloud_top) if np.isfinite(cloud_top) else None,
        "+DI > −DI": bool(last["plus_di"] > last["minus_di"]), "MACD > 0": bool(last["macd"] > 0),
        "preço > SAR": bool(c > last["sar"]) if np.isfinite(last["sar"]) else None}
    ups = sum(1 for v in trend_votes.values() if v is True)
    downs = sum(1 for v in trend_votes.values() if v is False)
    bias = "alta" if ups >= 5 else ("baixa" if downs >= 5 else "lateral/indefinida")
    recent = ev.iloc[-RECENT_BARS:]
    found = [k for k in ev.columns if bool(recent[k].any())]
    perf = (study or {}).get("events", {})

    def describe(k: str) -> dict[str, Any]:
        item: dict[str, Any] = {"padrao": k, "categoria": CATEGORY.get(k, "padrão gráfico"),
                                "vies": "alta" if k in BULL else ("baixa" if k in BEAR else "neutro")}
        if k in perf:
            item["historico"] = perf[k]
        return item

    bull = [describe(k) for k in found if k in BULL]
    bear = [describe(k) for k in found if k in BEAR]
    neutral = [describe(k) for k in found if k not in BULL and k not in BEAR]
    vp = volume_profile(df.iloc[-120:])
    near = [z for z in state.zones if abs((z["de"] + z["ate"]) / 2 / c - 1) < 0.05]
    return {
        "preco": c, "tendencia": {"vies": bias, "votos": trend_votes, "adx": _num(last["adx"], 1),
                                   "forca": "forte" if last["adx"] >= 25 else ("fraca" if last["adx"] < 20 else "moderada")},
        "momento": {"rsi": _num(last["rsi"], 1), "macd_hist": _num(last["macd_hist"], 6),
                    "estocastico_k": _num(last["stoch_k"], 1), "cci": _num(last["cci"], 0), "roc_pct": _num(last["roc"], 2)},
        "volatilidade": {"atr_pct": _num(last["atr"] / c * 100, 2), "largura_bollinger_pct": _num(last["bb_width"] * 100, 2),
                         "squeeze_ttm": bool(last["squeeze_on"]),
                         "posicao_bollinger": "acima" if c > last["bb_up"] else ("abaixo" if c < last["bb_lo"] else "dentro")},
        "volume": {"obv_acima_media": bool(last["obv"] > last["obv_ema"]), "acima_vwap": bool(c > last["vwap"]),
                   "poc": _num(vp["poc"], 8), "value_area": [_num(vp["val"], 8), _num(vp["vah"], 8)]},
        "estrutura": {"tendencia_smc": {1: "alta", -1: "baixa", 0: "indefinida"}[state.trend],
                      "ultimo_topo": _num(state.last_high.price, 8) if state.last_high else None,
                      "ultimo_fundo": _num(state.last_low.price, 8) if state.last_low else None,
                      "zonas_proximas": near[-6:]},
        "padroes_recentes": {"alta": bull, "baixa": bear, "neutros": neutral},
        "confluencia": {"sinais_alta": len(bull), "sinais_baixa": len(bear), "saldo": len(bull) - len(bear)},
    }
