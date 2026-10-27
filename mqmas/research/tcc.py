"""The four operating models of Prates (2021, UFU) — "Análise de Indicadores Técnicos para Negociação em
Bolsa de Valores" — rebuilt on Binance spot and reported in the same format as the thesis.

Models, with the thesis' parameters (signal on the closed candle, order at the next candle's open):

* Cruzamento de Médias — EMA20 crosses the SMA50 (up = buy, down = sell);
* Cruzamento do MACD — MACD(12, 26) crosses its 9-period signal line;
* Rompimento do Estocástico — slow stochastic (5, 3, 3, MetaTrader's default) %K crosses 20 up = buy,
  crosses 80 down = sell;
* Fechou Fora Fechou Dentro (FFFD) — Bollinger(20, 2): close below the lower band then close back
  inside = buy; close above the upper band then back inside = sell.

Differences from the thesis, stated in the report: spot has no short, so a sell signal closes the long
(the thesis also sold short); costs are included (0.10 % fee + 0.05 % slippage per side), which the thesis
left out; the test covers several years and coins instead of one asset in 2019. Each model also runs
behind our macro filter (BTC above SMA100, coin above EMA200) to show what that filter adds.

Format, as in the thesis: per-operation table (tipo, resultado, saldo, rentabilidade acumulada), the
averages table (média operacional, média de ganho, média de perda, rentabilidade média), hit rate, the
saldo × capital líquido curve, and the comparison table against the asset's buy & hold.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mqmas.research.data import load_history
from mqmas.research.engine import Costs
from mqmas.research.strategies import ema

log = logging.getLogger(__name__)

VERSION = 1
ASSETS = ("BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT")
SINCE = pd.Timestamp("2018-01-01", tz="UTC")
START = pd.Timestamp("2019-01-01", tz="UTC")  # same first year as the thesis; 2018 is indicator warm-up
CAPITAL = 10_000.0  # the thesis' account size
MODELS = ("Cruzamento de Médias", "Cruzamento do MACD", "Rompimento do Estocástico", "FFFD")


def signals(df: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    """(buy, sell) on each closed candle, per model."""
    c, h, lo = df["close"], df["high"], df["low"]

    def up(a: pd.Series, b: pd.Series | float) -> pd.Series:
        bp = b.shift(1) if isinstance(b, pd.Series) else b
        return (a > b) & (a.shift(1) < bp)

    def down(a: pd.Series, b: pd.Series | float) -> pd.Series:
        bp = b.shift(1) if isinstance(b, pd.Series) else b
        return (a < b) & (a.shift(1) > bp)

    fast, slow = ema(c, 20), c.rolling(50).mean()
    macd = ema(c, 12) - ema(c, 26)
    sig = macd.ewm(span=9, adjust=False).mean()
    raw_k = 100 * (c - lo.rolling(5).min()) / (h.rolling(5).max() - lo.rolling(5).min()).replace(0, np.nan)
    k = raw_k.rolling(3).mean()  # slowing 3 = the slow %K
    mid, sd = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
    upper, lower = mid + 2 * sd, mid - 2 * sd
    return {
        "Cruzamento de Médias": (up(fast, slow), down(fast, slow)),
        "Cruzamento do MACD": (up(macd, sig), down(macd, sig)),
        "Rompimento do Estocástico": (up(k, 20.0), down(k, 80.0)),
        "FFFD": ((c.shift(1) < lower.shift(1)) & (c > lower), (c.shift(1) > upper.shift(1)) & (c < upper)),
    }


def macro(df: pd.DataFrame, btc: pd.DataFrame) -> pd.Series:
    """Our gate, known at the close of each candle: BTC above its SMA100 and the coin above its EMA200."""
    b = btc["close"]
    btc_ok = (b > b.rolling(100).mean()).reindex(df.index).fillna(False)
    return (df["close"] > ema(df["close"], 200)) & btc_ok.astype(bool)


def backtest(df: pd.DataFrame, buy: pd.Series, sell: pd.Series, costs: Costs,
             gate: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long-only, all-in, compounding: buy signal → enter at the next open; sell signal → exit at the next open.
    Returns the operations table and the daily saldo / capital líquido curve."""
    o, c = df["open"].to_numpy(), df["close"].to_numpy()
    b, s = buy.fillna(False).to_numpy(), sell.fillna(False).to_numpy()
    g = gate.fillna(False).to_numpy() if gate is not None else np.ones(len(df), dtype=bool)
    idx = df.index
    first = int(np.searchsorted(idx, START))
    saldo, qty, entry_px, entry_t = CAPITAL, 0.0, 0.0, None
    ops: list[dict[str, Any]] = []
    curve = []
    side = costs.fee + costs.slippage
    for i in range(first, len(df)):
        if i > first:  # orders from yesterday's signal fill at today's open
            if qty == 0 and b[i - 1] and g[i - 1]:
                entry_px, entry_t = o[i] * (1 + side), idx[i]
                qty = saldo / entry_px
            elif qty > 0 and s[i - 1]:
                exit_px = o[i] * (1 - side)
                pnl = qty * exit_px - saldo
                saldo += pnl
                ops.append({"entrada": entry_t, "saida": idx[i], "tipo": "Compra", "resultado": pnl,
                            "retorno_pct": (exit_px / entry_px - 1) * 100, "saldo": saldo,
                            "rentabilidade_pct": (saldo / CAPITAL - 1) * 100})
                qty = 0.0
        liquid = qty * c[i] * (1 - side) if qty > 0 else saldo
        curve.append((idx[i], saldo, liquid))
    eq = pd.DataFrame(curve, columns=["data", "saldo", "capital_liquido"]).set_index("data")
    return pd.DataFrame(ops), eq


def averages(ops: pd.DataFrame, eq: pd.DataFrame, df: pd.DataFrame) -> dict[str, Any]:
    if ops.empty:
        return {"operacoes": 0}
    r = ops["resultado"]
    wins, losses = r[r > 0], r[r <= 0]
    liq = eq["capital_liquido"]
    peak = liq.cummax()
    first_px = float(df.loc[df.index >= START, "open"].iloc[0])
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    final = float(liq.iloc[-1])
    return {"operacoes": int(len(r)), "taxa_acerto": round(float((r > 0).mean()), 4),
            "media_operacional": round(float(r.mean()), 2),
            "media_ganho": round(float(wins.mean()), 2) if len(wins) else 0.0,
            "media_perda": round(float(losses.mean()), 2) if len(losses) else 0.0,
            "media_ganho_pct": round(float(ops.loc[r > 0, "retorno_pct"].mean()), 2) if len(wins) else 0.0,
            "media_perda_pct": round(float(ops.loc[r <= 0, "retorno_pct"].mean()), 2) if len(losses) else 0.0,
            "rentabilidade_media_pct": round(float(ops["retorno_pct"].mean()), 2),
            "rentabilidade_total_pct": round((final / CAPITAL - 1) * 100, 2),
            "rentabilidade_ao_ano_pct": round(((final / CAPITAL) ** (1 / years) - 1) * 100, 2) if final > 0 else -100.0,
            "buy_hold_pct": round((float(df["close"].iloc[-1]) / first_px - 1) * 100, 2),
            "drawdown_max_pct": round(float(((peak - liq) / peak).max()) * 100, 2),
            "tempo_posicionado_pct": round(float((eq["capital_liquido"] != eq["saldo"]).mean()) * 100, 1),
            "por_ano": {str(y): round(float(g["retorno_pct"].sum()), 1)
                        for y, g in ops.groupby(ops["saida"].dt.year)}}


def run(root: Path, ex: Any = None) -> dict[str, Any]:
    if ex is None:
        import ccxt

        ex = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    since = int(SINCE.timestamp() * 1000)
    data = {a: load_history(ex, a, "1d", since, root / "tcc") for a in ASSETS}
    btc = data["BTC/USDT"]
    out: dict[str, Any] = {"version": VERSION, "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
                           "periodo": f"{START.date()} → {btc.index[-1].date()}", "capital": CAPITAL, "ativos": {}}
    for asset, df in data.items():
        if len(df) < 300 or df.index[0] > START:
            log.info("tcc: %s skipped (history starts %s)", asset, df.index[0].date() if len(df) else "-")
            continue
        sig = signals(df)
        gate = macro(df, btc)
        res: dict[str, Any] = {}
        for model, (buy, sell) in sig.items():
            for label, costs, g in (("sem custos (como no TCC)", Costs(fee=0.0, slippage=0.0), None),
                                    ("com custos", Costs(), None), ("com custos + filtro macro", Costs(), gate)):
                ops, eq = backtest(df, buy, sell, costs, g)
                entry: dict[str, Any] = {"medias": averages(ops, eq, df)}
                if asset == "BTC/USDT" and label == "com custos":
                    entry["operacoes"] = [{**{k: v for k, v in r.items() if k not in ("entrada", "saida")},
                                           "entrada": str(r["entrada"].date()), "saida": str(r["saida"].date())}
                                          for r in ops.round(2).to_dict("records")]
                    entry["curva"] = eq.resample("W").last().round(2).reset_index().assign(
                        data=lambda x: x["data"].dt.strftime("%Y-%m-%d")).to_dict("records")
                res.setdefault(model, {})[label] = entry
        out["ativos"][asset] = res
        log.info("tcc: %s done", asset)
    return out


def summarize(rep: dict[str, Any]) -> list[str]:
    lines = [f"TCC v{rep['version']} | {rep['periodo']} | capital {rep['capital']:.0f} | modelos de Prates (2021)"]
    for asset, models in rep["ativos"].items():
        for model, variants in models.items():
            for label, v in variants.items():
                m = v["medias"]
                if not m.get("operacoes"):
                    lines.append(f"TCC {asset} {model} [{label}]: sem operações")
                    continue
                lines.append(
                    f"TCC {asset} {model} [{label}] | n={m['operacoes']} acerto {m['taxa_acerto']:.0%} | "
                    f"M.oper {m['media_operacional']:+.2f} M.ganho {m['media_ganho']:+.2f} ({m['media_ganho_pct']:+.1f}%) "
                    f"M.perda {m['media_perda']:+.2f} ({m['media_perda_pct']:+.1f}%) | rent. total "
                    f"{m['rentabilidade_total_pct']:+.1f}% ({m['rentabilidade_ao_ano_pct']:+.1f}%/ano) vs buy&hold "
                    f"{m['buy_hold_pct']:+.1f}% | DD {m['drawdown_max_pct']:.1f}% | posicionado "
                    f"{m['tempo_posicionado_pct']:.0f}% | por ano {m['por_ano']}")
    return lines
