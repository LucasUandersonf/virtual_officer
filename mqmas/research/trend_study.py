"""Trend ensemble study: the reverse-engineered system on our own Binance data, with the real account's rules.

Nothing is fitted: the nine lookbacks, the volatility window, the 40 % volatility target and the 3-ATR
crash stop are fixed before looking at the data (they come from the paper or from our mandate). The few
alternatives below are reported as *sensitivity*, never used to pick the live configuration.

The account is simulated day by day exactly as the live allocator trades:

* decisions on the daily close, orders at the next daily open (fee + slippage on every leg);
* Binance's US$ 5 minimum per order and a 25 % rebalancing band (``mqmas.trend.rebalance``);
* an exchange-side crash stop 3 ATR under the close, ratcheted up daily; a coin that hits it is blocked
  until its trend signal strengthens again (a fresh breakout);
* the accounts start at US$ 50, 120, 500 and 1 000, plus a large one where the minimum order never binds.

Benchmarks over the same days: holding BTC, holding an equal-weight basket of the same universe
(rebalanced monthly) and weekly DCA into BTC.

Survivorship bias: the coin pool is today's most liquid pairs, so coins that died since 2019 are missing.
That flatters every long-only result, the benchmarks included.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mqmas.research.data import candidate_pool, load_history, research_exchange
from mqmas.research.engine import Costs
from mqmas.research.templates import dca_benchmark
from mqmas.trend.model import CRASH_STOP_ATR, MIN_HISTORY, TrendParams, atr, eligibility, weights_frame
from mqmas.trend.rebalance import MIN_ORDER_USDT, plan_orders

log = logging.getLogger(__name__)

VERSION = 1
HISTORY_START = pd.Timestamp("2019-01-01", tz="UTC")
COMPARABLE_START = pd.Timestamp("2023-07-01", tz="UTC")  # the other studies' out-of-sample period
POOL_SIZE = 40
ACCOUNTS = (50.0, 120.0, 500.0, 1000.0)
IDEAL_EQUITY = 1_000_000.0  # the minimum order never binds


def simulate(daily: dict[str, pd.DataFrame], weights: pd.DataFrame, signal: pd.DataFrame, equity0: float,
             costs: Costs, start: pd.Timestamp, crash_stop: bool = True, min_order: float = MIN_ORDER_USDT
             ) -> dict[str, Any]:
    """Day-by-day account: returns the daily equity curve, fees, turnover, exposure and round trips."""
    opens = pd.DataFrame({s: d["open"] for s, d in daily.items()}).reindex(weights.index)
    lows = pd.DataFrame({s: d["low"] for s, d in daily.items()}).reindex(weights.index)
    closes = pd.DataFrame({s: d["close"] for s, d in daily.items()}).reindex(weights.index)
    atrs = pd.DataFrame({s: atr(d) for s, d in daily.items()}).reindex(weights.index)
    days = weights.index[weights.index >= start]
    side_cost = costs.fee + costs.slippage
    cash = equity0
    units: dict[str, float] = {}
    stops: dict[str, float] = {}
    blocked: dict[str, float] = {}  # coin -> signal level at the stop-out
    spent: dict[str, float] = {}  # USDT paid into the open position (all its buys)
    received: dict[str, float] = {}  # USDT taken out of it (all its sells)
    trips: list[float] = []  # net return of each position, first buy to last sell

    def _close_trip(sym: str) -> None:
        paid = spent.pop(sym, 0.0)
        trips.append(received.pop(sym, 0.0) / paid - 1 if paid > 0 else 0.0)
        units.pop(sym, None)
        stops.pop(sym, None)

    fees = turnover = 0.0
    curve, expo = [], []
    pending: list[Any] = []
    for day in days:
        o, lo, c = opens.loc[day], lows.loc[day], closes.loc[day]
        # 1) yesterday's orders at today's open
        for od in pending:
            px = float(o.get(od.symbol, np.nan))
            if not np.isfinite(px) or px <= 0:
                continue
            if od.side == "sell":
                held = units.get(od.symbol, 0.0)
                q = held if od.close_all else min(od.value / px, held)
                if q <= 0:
                    continue
                got = q * px * (1 - side_cost)
                cash += got
                fees += q * px * costs.fee
                turnover += q * px
                received[od.symbol] = received.get(od.symbol, 0.0) + got
                units[od.symbol] = held - q
                if od.close_all or units[od.symbol] <= 0:
                    _close_trip(od.symbol)
            else:
                spend = min(od.value, cash)
                if spend < min_order:
                    continue
                cash -= spend
                fees += spend * costs.fee
                turnover += spend
                units[od.symbol] = units.get(od.symbol, 0.0) + spend * (1 - side_cost) / px
                spent[od.symbol] = spent.get(od.symbol, 0.0) + spend
        pending = []
        # 2) crash stops during the day
        if crash_stop:
            for sym in list(units):
                st = stops.get(sym)
                if st is None or not np.isfinite(lo.get(sym, np.nan)) or lo[sym] > st:
                    continue
                px = min(float(o[sym]), st) * (1 - costs.slippage)
                got = units[sym] * px * (1 - costs.fee)
                cash += got
                fees += units[sym] * px * costs.fee
                turnover += units[sym] * px
                received[sym] = received.get(sym, 0.0) + got
                _close_trip(sym)
                blocked[sym] = float(signal.at[day, sym]) if np.isfinite(signal.at[day, sym]) else 0.0
        # 3) mark to market on the close, plan tomorrow
        values = {s: q * float(c[s]) for s, q in units.items() if np.isfinite(c.get(s, np.nan))}
        equity = cash + sum(values.values())
        curve.append(equity)
        expo.append(sum(values.values()) / equity if equity > 0 else 0.0)
        for sym in list(blocked):
            if float(signal.at[day, sym]) > blocked[sym] + 1e-9:
                blocked.pop(sym)
            else:
                blocked[sym] = min(blocked[sym], float(signal.at[day, sym]))
        w = weights.loc[day]
        targets = {s: float(x) for s, x in w.items() if x > 0 and s not in blocked}
        pending = plan_orders(targets, values, cash, min_order=min_order)
        if crash_stop:
            for sym in units:
                a = float(atrs.at[day, sym])
                if np.isfinite(a) and a > 0:
                    lvl = float(c[sym]) - CRASH_STOP_ATR * a
                    stops[sym] = max(stops.get(sym, lvl), lvl)
    eq = pd.Series(curve, index=days, dtype=float)
    return {"equity": eq, "fees": fees, "turnover": turnover, "exposure": float(np.mean(expo)) if expo else 0.0,
            "trips": trips}


def curve_metrics(eq: pd.Series, equity0: float) -> dict[str, float]:
    if len(eq) < 2:
        return {}
    rets = eq.pct_change().fillna(eq.iloc[0] / equity0 - 1)
    years = len(eq) / 365.0
    final = float(eq.iloc[-1])
    peak = eq.cummax()
    dd = float(((peak - eq) / peak).max())
    sd = float(rets.std(ddof=1))
    down = float(np.sqrt((np.minimum(rets, 0) ** 2).mean()))
    months = eq.resample("ME").last()
    monthly = months.pct_change()
    monthly.iloc[0] = months.iloc[0] / equity0 - 1
    cagr = (final / equity0) ** (1 / years) - 1 if final > 0 else -1.0
    return {
        "total_return": final / equity0 - 1, "cagr": cagr, "max_drawdown": dd,
        "sharpe": float(rets.mean()) / sd * math.sqrt(365) if sd > 0 else 0.0,
        "sortino": float(rets.mean()) / down * math.sqrt(365) if down > 0 else 0.0,
        "calmar": cagr / dd if dd > 0 else 0.0, "vol": sd * math.sqrt(365),
        "monthly_mean": float(monthly.mean()), "monthly_median": float(monthly.median()),
        "positive_months": float((monthly > 0).mean()), "worst_month": float(monthly.min()),
        "best_month": float(monthly.max()), "months": float(len(monthly)), "final": final,
    }


def by_year(eq: pd.Series, equity0: float) -> dict[str, float]:
    ye = eq.resample("YE").last()
    prev = ye.shift(1)
    prev.iloc[0] = equity0
    return {str(d.year): float(v / p - 1) for d, v, p in zip(ye.index, ye, prev, strict=True)}


def summary(sim: dict[str, Any], equity0: float) -> dict[str, Any]:
    eq = sim["equity"]
    trips = np.asarray(sim["trips"], dtype=float)
    years = max(len(eq) / 365.0, 1e-9)
    out: dict[str, Any] = dict(curve_metrics(eq, equity0))
    wins, losses = trips[trips > 0], trips[trips <= 0]
    out.update({
        "exposure": sim["exposure"], "fees": sim["fees"], "fees_per_year_pct": sim["fees"] / equity0 / years,
        "turnover_per_year": sim["turnover"] / max(float(eq.mean()), 1e-9) / years,
        "round_trips": float(len(trips)), "trips_per_month": float(len(trips)) / years / 12,
        "win_rate": float(wins.size) / max(float(trips.size), 1.0),
        "profit_factor": float(wins.sum()) / -float(losses.sum()) if losses.sum() < 0 else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0, "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "by_year": by_year(eq, equity0),
    })
    return out


def hold_curve(closes: pd.DataFrame, start: pd.Timestamp, equity0: float) -> pd.Series:
    c = closes.loc[start:].ffill()
    return c / c.iloc[0] * equity0


def basket_curve(daily: dict[str, pd.DataFrame], top_n: int, start: pd.Timestamp, costs: Costs) -> pd.Series:
    """Equal-weight hold of the same point-in-time universe, rebalanced on the first day of each month."""
    closes = pd.DataFrame({s: d["close"] for s, d in daily.items()}).sort_index()
    elig = eligibility(daily, top_n)
    rets = closes.pct_change().loc[start:]
    el = elig.loc[start:]
    w = el.div(el.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    m = w.index.tz_localize(None).to_period("M")
    first = np.r_[True, m[1:] != m[:-1]]
    w = w.where(pd.Series(first, index=w.index), np.nan).ffill().fillna(0.0)
    turn = w.diff().abs().sum(axis=1).fillna(w.iloc[0].abs().sum())
    port = (w.shift(1) * rets.fillna(0.0)).sum(axis=1) - turn * (costs.fee + costs.slippage)
    return (1 + port).cumprod()


class TrendStudy:
    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, ex: Any = None) -> dict[str, pd.DataFrame]:
        ex = ex if ex is not None else research_exchange()
        pool = candidate_pool(ex, POOL_SIZE)
        log.info("trend pool (%d): %s", len(pool), ", ".join(pool))
        since = int(HISTORY_START.timestamp() * 1000)
        daily: dict[str, pd.DataFrame] = {}
        for sym in pool:
            df = load_history(ex, sym, "1d", since, self.root / "trend_candles")
            if len(df) > 60:
                daily[sym] = df
        return daily

    def run(self, ex: Any = None, daily: dict[str, pd.DataFrame] | None = None) -> dict[str, Any]:
        daily = daily if daily is not None else self.load(ex)
        btc = daily["BTC/USDT"]
        start = btc.index[0] + pd.Timedelta(days=MIN_HISTORY)
        end = btc.index[-1]
        base = TrendParams()
        costs = Costs()
        rep: dict[str, Any] = {"version": VERSION, "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
                               "config": base.key, "periodo": f"{start.date()} → {end.date()}",
                               "moedas": sorted(daily), "custo_por_lado": costs.fee + costs.slippage}
        cache: dict[TrendParams, tuple[pd.DataFrame, pd.DataFrame]] = {}

        def frames(p: TrendParams) -> tuple[pd.DataFrame, pd.DataFrame]:
            if p not in cache:
                cache[p] = weights_frame(daily, p)
            return cache[p]

        w, sig = frames(base)
        periods = {"completo": start, "desde_2023_07": COMPARABLE_START}
        rep["sistema"] = {}
        rep["referencias"] = {}
        btc_close = pd.DataFrame({"BTC": btc["close"]})
        for name, s in periods.items():
            row: dict[str, Any] = {}
            for eq0 in (*ACCOUNTS, IDEAL_EQUITY):
                label = "ideal" if eq0 == IDEAL_EQUITY else f"US${eq0:g}"
                row[label] = summary(simulate(daily, w, sig, eq0, costs, s), eq0)
            rep["sistema"][name] = row
            refs: dict[str, Any] = {
                "segurar_btc": curve_metrics(hold_curve(btc_close, s, 1.0)["BTC"], 1.0),
                "cesta_igual_top10": curve_metrics(basket_curve(daily, base.top_n, s, costs), 1.0),
            }
            refs["segurar_btc"]["by_year"] = by_year(hold_curve(btc_close, s, 1.0)["BTC"], 1.0)
            refs["dca_btc"] = dca_benchmark(daily, s, end, {"BTC/USDT": 1.0})
            rep["referencias"][name] = refs
        # sensitivity: never used to choose the live configuration
        sens: dict[str, Any] = {}
        variants = {
            "custo_dobrado": (base, Costs(fee=0.002, slippage=0.001), True),
            "sem_stop_de_crash": (base, costs, False),
            "vol_alvo_25": (replace(base, vol_target=0.25), costs, True),
            "vol_alvo_60": (replace(base, vol_target=0.60), costs, True),
            "top5": (replace(base, top_n=5, max_weight=0.5), costs, True),
            "top20": (replace(base, top_n=20, max_weight=0.25), costs, True),
        }
        for name, (p, c, cs) in variants.items():
            wv, sv = frames(p)
            sens[name] = {lab: summary(simulate(daily, wv, sv, eq0, c, s, crash_stop=cs), eq0)
                          for lab, eq0, s in (("ideal_completo", IDEAL_EQUITY, start),
                                              ("ideal_desde_2023_07", IDEAL_EQUITY, COMPARABLE_START),
                                              ("US$120_completo", 120.0, start))}
        rep["sensibilidade"] = sens
        last = w.iloc[-1]
        rep["carteira_hoje"] = {"dia": str(w.index[-1].date()),
                                "pesos": {s: round(float(x), 4) for s, x in last.items() if x > 0},
                                "sinal": {s: round(float(x), 3) for s, x in sig.iloc[-1].items()
                                          if np.isfinite(x) and x > 0}}
        return rep


def _fmt(m: dict[str, Any]) -> str:
    return (f"CAGR {m['cagr']:+.1%} DD {m['max_drawdown']:.1%} Sharpe {m['sharpe']:.2f} "
            f"mês médio {m['monthly_mean']:+.2%} meses+ {m['positive_months']:.0%}")


def summarize(rep: dict[str, Any]) -> list[str]:
    out = [f"TREND v{rep['version']} {rep['config']} | {rep['periodo']} | custo {rep['custo_por_lado']:.2%}/lado"]
    for per, rows in rep["sistema"].items():
        for lab, m in rows.items():
            out.append(f"TREND {per} {lab}: {_fmt(m)} | exposição {m['exposure']:.0%} viagens {m['round_trips']:.0f} "
                       f"({m['trips_per_month']:.1f}/mês) acerto {m['win_rate']:.0%} PF {m['profit_factor']:.2f} "
                       f"taxas {m['fees_per_year_pct']:.2%}/ano | anos {m['by_year']}")
        refs = rep["referencias"][per]
        out.append(f"TREND {per} segurar BTC: {_fmt(refs['segurar_btc'])} | anos {refs['segurar_btc']['by_year']}")
        out.append(f"TREND {per} cesta igual: {_fmt(refs['cesta_igual_top10'])}")
        d = refs["dca_btc"]
        out.append(f"TREND {per} DCA BTC: retorno sobre o investido {d['total_return_on_invested']:+.1%} "
                   f"pior DD {d['worst_drawdown_vs_cost']:.1%}")
    for name, rows in rep["sensibilidade"].items():
        for lab, m in rows.items():
            out.append(f"TREND sens {name} {lab}: {_fmt(m)} exposição {m['exposure']:.0%}")
    out.append(f"TREND carteira hoje ({rep['carteira_hoje']['dia']}): {rep['carteira_hoje']['pesos']}")
    return out
