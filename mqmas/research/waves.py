"""Big-wave study: can a long-only spot desk catch the +10/20/30 % days, and does it pay after costs?

Every rule below was fixed from published research *before* running it on our data (Binance daily candles,
2019-01 → today, the same 40-coin pool and point-in-time universe as the trend study):

1. **Event study.** After a coin closes a day up ≥ 10 / 20 / 30 % (or down ≥ 10 %), what does it do over the
   next 1, 5, 10 and 20 days, bought at that close with 0.30 % round-trip cost? Compared with the average
   forward return of all coin-days, so a bull market is not mistaken for an edge. Literature is split:
   continuation (Caporale & Plastun 2019) vs. short-term reversal (Kosc et al. 2019).
2. **Weekly momentum** (Liu, Tsyvinski & Wu): every Monday hold the 5 coins with the best 3-week return
   (middle of the paper's 1–4 week range), equal weight, long only (spot cannot short).
3. **MAX momentum** (Li, Urquhart, Wang & Zhang 2021): every Monday hold the 5 coins with the largest single
   daily gain of the past week.
4. **Momentum + regime filter:** strategy 2, but in USDT while BTC closes under its 100-day average (the regime
   filter used across our earlier research).
5. **Wave capture of the trend ensemble:** of all +10 % days in the universe, on how many was the trend desk
   already holding that coin, and how much of its profit came from those days?

Costs: 0.10 % fee + 0.05 % slippage per side, charged on turnover. Survivorship bias (today's liquid coins)
flatters every long-only line here, benchmarks included.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mqmas.research.engine import Costs
from mqmas.research.trend_study import COMPARABLE_START, TrendStudy, by_year, curve_metrics
from mqmas.trend.model import MIN_HISTORY, TrendParams, eligibility, weights_frame

log = logging.getLogger(__name__)

VERSION = 1
THRESHOLDS = (0.10, 0.20, 0.30, -0.10)
HORIZONS = (1, 5, 10, 20)
UNIVERSE = 30
TOP_K = 5
MOM_DAYS = 21
MAX_DAYS = 7
BTC_SMA = 100


def event_study(closes: pd.DataFrame, elig: pd.DataFrame, start: pd.Timestamp, costs: Costs) -> dict[str, Any]:
    ret = closes.pct_change()
    rt = 2 * (costs.fee + costs.slippage)
    out: dict[str, Any] = {}
    base = {}
    for h in HORIZONS:
        fwd = closes.shift(-h) / closes - 1
        mask = (elig & fwd.notna()).to_numpy() & (closes.index >= start)[:, None]
        base[h] = float(fwd.to_numpy()[mask].mean())
    out["media_incondicional"] = {f"{h}d": base[h] for h in HORIZONS}
    for thr in THRESHOLDS:
        hit = (ret >= thr) if thr > 0 else (ret <= thr)
        hit &= elig & (closes.index >= start)[:, None]
        row: dict[str, Any] = {"eventos": int(hit.to_numpy().sum())}
        for h in HORIZONS:
            fwd = closes.shift(-h) / closes - 1
            x = fwd.to_numpy(dtype=float)[(hit & fwd.notna()).to_numpy()] - rt
            if len(x) < 5:
                continue
            sd = float(x.std(ddof=1))
            row[f"{h}d"] = {"n": len(x), "media": float(x.mean()), "mediana": float(np.median(x)),
                            "acerto": float((x > 0).mean()), "excesso": float(x.mean() + rt - base[h]),
                            "t": float(x.mean() / sd * math.sqrt(len(x))) if sd > 0 else 0.0,
                            "pior": float(x.min()), "melhor": float(x.max())}
        out[f"{'+' if thr > 0 else ''}{thr:.0%}"] = row
    return out


def weekly_weights(closes: pd.DataFrame, elig: pd.DataFrame, score: pd.DataFrame,
                   regime: pd.Series | None = None) -> pd.DataFrame:
    """Equal-weight top-K by ``score`` on each Monday close, held until the next Monday."""
    w = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
    for day in closes.index[closes.index.dayofweek == 0]:
        row = pd.Series(0.0, index=closes.columns)
        if regime is None or bool(regime.get(day, False)):
            s = score.loc[day].where(elig.loc[day]).dropna()
            s = s[s > 0]  # long only: never buy a coin whose own score is negative
            top = s.sort_values(ascending=False).index[:TOP_K]
            if len(top):
                row[top] = 1.0 / len(top)
        w.loc[day] = row
    return w.ffill().fillna(0.0)


def run_weights(closes: pd.DataFrame, w: pd.DataFrame, start: pd.Timestamp, costs: Costs) -> pd.Series:
    """Equity curve (1.0 at start) of daily weights decided on each close and held over the next day."""
    rets = closes.pct_change().fillna(0.0)
    w = w.loc[start:]
    held = w.shift(1).fillna(0.0)
    turn = w.diff().abs().sum(axis=1)
    turn.iloc[0] = w.iloc[0].abs().sum()
    port = (held * rets.loc[start:]).sum(axis=1) - turn * (costs.fee + costs.slippage)
    return (1 + port).cumprod()


def wave_capture(closes: pd.DataFrame, elig: pd.DataFrame, w: pd.DataFrame, start: pd.Timestamp) -> dict[str, Any]:
    ret = closes.pct_change()
    held = w.shift(1).reindex(closes.index).fillna(0.0) > 0
    live = elig & (closes.index >= start)[:, None]
    contrib = (w.shift(1).reindex(closes.index).fillna(0.0) * ret.fillna(0.0)).to_numpy()
    out: dict[str, Any] = {}
    for name, cond in (("dias_+10%", ret >= 0.10), ("dias_+20%", ret >= 0.20), ("dias_-10%", ret <= -0.10)):
        c = cond & live
        n = int(c.to_numpy().sum())
        out[name] = {"dias": n, "posicionado_pct": float((c & held).to_numpy().sum() / n) if n else 0.0,
                     "contribuicao": float(contrib[c.to_numpy()].sum())}
    out["contribuicao_total"] = float(contrib[live.to_numpy()].sum())
    return out


def run(root: Path, ex: Any = None, daily: dict[str, pd.DataFrame] | None = None) -> dict[str, Any]:
    daily = daily if daily is not None else TrendStudy(root).load(ex)
    closes = pd.DataFrame({s: d["close"] for s, d in daily.items()}).sort_index()
    btc = closes["BTC/USDT"]
    start = btc.dropna().index[0] + pd.Timedelta(days=MIN_HISTORY)
    costs = Costs()
    elig = eligibility(daily, UNIVERSE)
    rep: dict[str, Any] = {"version": VERSION, "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
                           "periodo": f"{start.date()} → {closes.index[-1].date()}", "universo": UNIVERSE,
                           "moedas": sorted(daily)}
    rep["eventos"] = event_study(closes, elig, start, costs)
    regime = btc > btc.rolling(BTC_SMA).mean()
    mom = closes / closes.shift(MOM_DAYS) - 1
    mx = closes.pct_change().rolling(MAX_DAYS).max()
    strategies = {
        "momentum_3sem_top5": weekly_weights(closes, elig, mom),
        "max_momentum_top5": weekly_weights(closes, elig, mx),
        "momentum_3sem_top5_filtro_btc": weekly_weights(closes, elig, mom, regime),
    }
    w_trend, _ = weights_frame(daily, TrendParams())
    w_waves, _ = weights_frame(daily, TrendParams(vol_target=0.60, top_n=20, max_weight=0.35))
    strategies["tendencia_base"] = w_trend
    strategies["tendencia_ondas_top20_vol60"] = w_waves
    rep["estrategias"] = {}
    for name, w in strategies.items():
        w = w.reindex(closes.index).fillna(0.0)
        row: dict[str, Any] = {}
        for per, s in (("completo", start), ("desde_2023_07", COMPARABLE_START)):
            eq = run_weights(closes, w, s, costs)
            m: dict[str, Any] = dict(curve_metrics(eq, 1.0))
            m["by_year"] = by_year(eq, 1.0)
            m["exposicao"] = float(w.loc[s:].sum(axis=1).mean())
            row[per] = m
        row["captura_de_ondas"] = wave_capture(closes, elig, w, start)
        rep["estrategias"][name] = row
    hold = (btc.loc[start:] / btc.loc[start:].iloc[0])
    rep["segurar_btc"] = {"completo": {**curve_metrics(hold, 1.0), "by_year": by_year(hold, 1.0)},
                          "desde_2023_07": curve_metrics(btc.loc[COMPARABLE_START:] / btc.loc[COMPARABLE_START:].iloc[0], 1.0)}
    return rep


def summarize(rep: dict[str, Any]) -> list[str]:
    out = [f"WAVES v{rep['version']} | {rep['periodo']} | universo top {rep['universo']}"]
    ev = rep["eventos"]
    out.append("WAVES média incondicional (todas as moedas-dia): " +
               ", ".join(f"{k} {v:+.2%}" for k, v in ev["media_incondicional"].items()))
    for k, row in ev.items():
        if k == "media_incondicional":
            continue
        parts = [f"{h} média {r['media']:+.2%} mediana {r['mediana']:+.2%} acerto {r['acerto']:.0%} "
                 f"excesso {r['excesso']:+.2%} t {r['t']:+.1f}" for h, r in row.items() if h != "eventos"]
        out.append(f"WAVES depois de dia {k} (n={row['eventos']}): " + " | ".join(parts))
    for name, row in rep["estrategias"].items():
        for per in ("completo", "desde_2023_07"):
            m = row[per]
            out.append(f"WAVES {name} {per}: CAGR {m['cagr']:+.1%} DD {m['max_drawdown']:.1%} Sharpe {m['sharpe']:.2f} "
                       f"mês médio {m['monthly_mean']:+.2%} meses+ {m['positive_months']:.0%} exposição "
                       f"{m['exposicao']:.0%}" + (f" | anos {m['by_year']}" if per == "completo" else ""))
        cap = row["captura_de_ondas"]
        out.append(f"WAVES {name} captura: " + " | ".join(
            f"{k}: {v['dias']} dias, posicionado {v['posicionado_pct']:.0%}, contribuição {v['contribuicao']:+.2f}"
            for k, v in cap.items() if isinstance(v, dict)) + f" | contribuição total {cap['contribuicao_total']:+.2f}")
    b = rep["segurar_btc"]
    out.append(f"WAVES segurar BTC completo: CAGR {b['completo']['cagr']:+.1%} DD {b['completo']['max_drawdown']:.1%} "
               f"Sharpe {b['completo']['sharpe']:.2f} | desde 2023-07: CAGR {b['desde_2023_07']['cagr']:+.1%} "
               f"DD {b['desde_2023_07']['max_drawdown']:.1%} Sharpe {b['desde_2023_07']['sharpe']:.2f}")
    return out
