"""Walk-forward research driver.

Protocol
--------
1. Universe: top liquid USDT pairs today (candidate pool); each month the
   tradable set is re-selected from trailing 30-day dollar volume and range.
2. 120 bracket configurations (4 families × 2 timeframes × params × exits)
   and 36 momentum-rotation configurations are simulated on every coin.
3. Anchored walk-forward: six-month test windows from mid-2023. For each
   window the configuration of each family is chosen on *all data before
   the window* (t-statistic of net per-trade returns), then traded blind
   inside the window.
4. Out-of-sample windows are stitched into one portfolio curve with the
   risk rules (0.6 % per trade, max 3 positions, daily loss 1.5 %, 24 h lock).
5. Stress: doubled costs, Monte-Carlo bootstrap of monthly returns, regime
   breakdown (BTC bull/bear/sideways, high volatility) and the worst days.
6. Deflated Sharpe Ratio (Bailey & López de Prado) corrects the OOS Sharpe
   for the number of configurations tried.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from mqmas.research.data import candidate_pool, load_history
from mqmas.research.engine import Costs, RiskRules, bracket_trades, simulate_portfolio
from mqmas.research.rotation import RotationConfig, returns_metrics, rotation_configs, rotation_returns
from mqmas.research.strategies import (
    FAMILIES,
    WARMUP,
    Config,
    align_daily,
    btc_regime,
    build_configs,
    daily_regime,
    daily_trend,
    features,
    monthly_eligibility,
    relative_strength,
)
from mqmas.research.templates import TEMPLATES, dca_benchmark, grid_returns, structured_trades

log = logging.getLogger(__name__)

VERSION = 5
HISTORY_START = pd.Timestamp("2022-01-01", tz="UTC")
FIRST_TEST = pd.Timestamp("2023-07-01", tz="UTC")
WINDOW_MONTHS = 6
MIN_TRAIN_TRADES = 30

CRITERIA = {
    "max_drawdown": 0.15, "profit_factor": 1.3, "payoff": 1.0, "sharpe": 1.0,
    "positive_months": 0.55, "dsr": 0.90, "trades": 60, "stressed_profit_factor": 1.1,
}


def test_windows(end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    out = []
    s = FIRST_TEST
    while s < end:
        e = min(s + pd.DateOffset(months=WINDOW_MONTHS), end)
        out.append((s, e))
        s = e
    return out


def trade_tstat(ret: np.ndarray) -> float:
    if ret.size < MIN_TRAIN_TRADES:
        return -math.inf
    sd = float(ret.std(ddof=1))
    return float(ret.mean()) / sd * math.sqrt(ret.size) if sd > 0 else -math.inf


def window_slice(tr: pd.DataFrame, s: pd.Timestamp, e: pd.Timestamp) -> pd.DataFrame:
    return tr[(tr["t_in"] >= s) & (tr["t_in"] < e)]


def deflated_sharpe(sr: float, n_obs: float, skew: float, kurt: float, trial_srs: list[float]) -> float:
    """Probability that the true (per-period) Sharpe is > 0 after selecting the best of N trials."""
    n = max(len(trial_srs), 1)
    var = float(np.var(trial_srs, ddof=1)) if n > 1 else 0.0
    g = 0.5772156649
    sr0 = math.sqrt(var) * ((1 - g) * stats.norm.ppf(1 - 1 / n) + g * stats.norm.ppf(1 - 1 / (n * math.e))) if n > 1 else 0.0
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr * sr, 1e-12))
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(max(n_obs - 1, 1)) / denom))


def monte_carlo(monthly: pd.Series, n: int = 3000, seed: int = 7) -> dict[str, float]:
    if len(monthly) < 3:
        return {}
    rng = np.random.default_rng(seed)
    m = monthly.to_numpy()
    sims = rng.choice(m, size=(n, len(m)), replace=True)
    eq = np.cumprod(1 + sims, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((n, 1)), eq], axis=1), axis=1)[:, 1:]
    dd = ((peak - eq) / peak).max(axis=1)
    yearly = np.prod(1 + rng.choice(m, size=(n, 12), replace=True), axis=1) - 1
    return {"dd_p50": float(np.median(dd)), "dd_p95": float(np.quantile(dd, 0.95)),
            "year_p5": float(np.quantile(yearly, 0.05)), "year_p50": float(np.median(yearly)),
            "year_p95": float(np.quantile(yearly, 0.95)), "prob_losing_year": float((yearly < 0).mean())}


def regime_breakdown(monthly: pd.Series, btc_daily: pd.DataFrame) -> dict[str, dict[str, float]]:
    c = btc_daily["close"]
    btc_m = c.resample("ME").last().pct_change()
    vol_m = c.pct_change().resample("ME").std() * math.sqrt(365)
    hi_vol = vol_m.quantile(0.75)
    labels: dict[str, list[float]] = {"alta (BTC > +10%)": [], "queda (BTC < -10%)": [], "lateral": [],
                                      "volatilidade alta": []}
    for ts, r in monthly.items():
        b = btc_m.get(ts, np.nan)
        if not np.isfinite(b):
            continue
        key = "alta (BTC > +10%)" if b > 0.10 else ("queda (BTC < -10%)" if b < -0.10 else "lateral")
        labels[key].append(float(r))
        if vol_m.get(ts, 0) >= hi_vol:
            labels["volatilidade alta"].append(float(r))
    return {k: {"months": len(v), "mean": float(np.mean(v)) if v else 0.0,
                "positive": float(np.mean([x > 0 for x in v])) if v else 0.0} for k, v in labels.items()}


class Lab:
    def __init__(self, root: Path, pool_size: int = 40, top_n: int = 15,
                 min_range: float = 0.025, max_range: float = 0.15) -> None:
        self.root = root
        self.pool_size, self.top_n, self.min_range, self.max_range = pool_size, top_n, min_range, max_range
        self.costs = Costs()
        # Percent results are scale-free; a large notional base keeps Binance's $5 minimum from
        # silently censoring wide-stop setups. Feasibility for the real $50 is reported separately.
        self.rules = RiskRules(equity0=10_000.0)

    # ------------------------------------------------------------------ data

    def load(self, ex: Any = None) -> None:
        if ex is None:
            import ccxt

            ex = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
        self.pool = candidate_pool(ex, self.pool_size)
        log.info("research pool (%d): %s", len(self.pool), ", ".join(self.pool))
        since = int((HISTORY_START - pd.Timedelta(days=60)).timestamp() * 1000)
        self.data: dict[str, dict[str, pd.DataFrame]] = {"1h": {}, "4h": {}, "1d": {}}
        cache = self.root / "candles"
        for sym in self.pool:
            for tf in ("1d", "4h", "1h"):
                df = load_history(ex, sym, tf, since, cache)
                if len(df) > WARMUP + 50:
                    self.data[tf][sym] = df
        log.info("history loaded: %s", {tf: len(v) for tf, v in self.data.items()})
        self.btc_daily = self.data["1d"]["BTC/USDT"]
        self.end = max(d.index[-1] for d in self.data["1h"].values()).normalize()
        self.eligible = monthly_eligibility(self.data["1d"], self.top_n, self.min_range, self.max_range)
        self.rs = relative_strength(self.data["1d"], self.eligible)
        self.daily_trend = {coin: daily_trend(d) for coin, d in self.data["1d"].items()}
        self.coin_regime = {coin: daily_regime(d) for coin, d in self.data["1d"].items()}

    # ------------------------------------------------------------------ bracket strategies

    def bracket_trade_table(self, configs: list[Config]) -> dict[str, pd.DataFrame]:
        costs = self.costs
        regime = btc_regime(self.btc_daily)
        rows: dict[str, list[pd.DataFrame]] = {c.key: [] for c in configs}
        by_tf: dict[str, list[Config]] = {}
        for c in configs:
            by_tf.setdefault(c.tf, []).append(c)
        for tf, cfgs in by_tf.items():
            for coin, df in self.data[tf].items():
                f = features(df[df.index >= HISTORY_START - pd.Timedelta(days=40)])
                reg = align_daily(regime, f.index)
                rs = self.rs[coin] if coin in self.rs.columns else pd.Series(dtype=float)
                f["rs"] = rs.reindex(f.index.normalize()).to_numpy()
                f["d_trend"] = self.daily_trend.get(coin, pd.Series(dtype=bool)).reindex(
                    f.index.normalize()).fillna(False).to_numpy(dtype=bool)
                months = f.index.tz_localize(None).to_period("M")
                elig = np.array([coin in self.eligible.get(m, ()) for m in months], dtype=bool)
                o, h, lo, cl = (f[k].to_numpy() for k in ("open", "high", "low", "close"))
                a = f["atr"].to_numpy()
                for c in cfgs:
                    sig = FAMILIES[c.family](f, c.p, reg) & elig
                    sig[:WARMUP] = False
                    t = bracket_trades(o, h, lo, cl, a, sig, c.exit, costs)
                    if t["entry"].size == 0:
                        continue
                    ein = t["entry"].astype(int)
                    eout = t["exit"].astype(int)
                    rows[c.key].append(pd.DataFrame({
                        "coin": coin, "t_in": f.index[ein], "t_out": f.index[eout] + pd.Timedelta(TF_DELTA[tf]),
                        "ret": t["ret"], "stop_pct": t["stop_pct"], "reason": t["reason"],
                        "regime": self._regime_at(coin, f.index[ein])}))
        empty = pd.DataFrame(columns=["coin", "t_in", "t_out", "ret", "stop_pct", "reason"])
        return {k: (pd.concat(v, ignore_index=True) if v else empty) for k, v in rows.items()}

    def _regime_at(self, coin: str, ts: pd.DatetimeIndex) -> np.ndarray:
        reg = self.coin_regime.get(coin)
        if reg is None:
            return np.full(len(ts), "lateral", dtype=object)
        out: np.ndarray = reg.reindex(ts.normalize()).fillna("lateral").to_numpy(dtype=object)
        return out

    def regime_router(self, wf: dict[str, Any], table: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
        """"O mercado dita a regra": in every test window each setup only trades the market phases where,
        on data *before* the window, its chosen configuration had a positive edge (n ≥ 20, t-stat ≥ 1)."""
        parts = []
        for name, r in wf.items():
            for pk in r["picks"]:
                tr = table[pk["config"]]
                train = window_slice(tr, HISTORY_START, pk["start"])
                allowed = [g for g, grp in train.groupby("regime")
                           if trade_tstat(grp["ret"].to_numpy(dtype=float)) >= 1.0]
                pk["regimes"] = allowed
                test = window_slice(tr, pk["start"], pk["end"])
                parts.append(test[test["regime"].isin(allowed)].assign(setup=name))
        return pd.concat(parts, ignore_index=True) if parts else None

    @staticmethod
    def by_regime(trades: pd.DataFrame) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        if "regime" not in trades:
            return out
        for g, grp in trades.groupby("regime"):
            r = grp["ret"].to_numpy(dtype=float)
            gain, loss = r[r > 0].sum(), -r[r <= 0].sum()
            out[str(g)] = {"trades": float(r.size), "win_rate": float((r > 0).mean()),
                           "expectancy_pct": float(r.mean()),
                           "profit_factor": float(min(gain / loss, 99.0)) if loss > 0 else 99.0}
        return out

    def walk_forward(self, groups: dict[str, list[str]], table: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """For each group of competing configurations, pick on the past and trade the next window blind."""
        windows = test_windows(self.end)
        results: dict[str, Any] = {}
        is_sr: dict[str, float] = {}
        for keys in groups.values():  # in-sample SR of every trial, for the deflated Sharpe
            for k in keys:
                res = simulate_portfolio(table[k], self.rules, self.costs, HISTORY_START, FIRST_TEST)
                is_sr[k] = res.metrics["daily_sr"]
        for name, cands in groups.items():
            oos_parts, picks = [], []
            for s, e in windows:
                best, best_t = None, -math.inf
                for k in cands:
                    score = trade_tstat(window_slice(table[k], HISTORY_START, s)["ret"].to_numpy(dtype=float))
                    if score > best_t:
                        best, best_t = k, score
                if best is None:
                    continue
                picks.append({"window": f"{s.date()}→{e.date()}", "start": s, "end": e, "config": best,
                              "train_tstat": best_t})
                oos_parts.append(window_slice(table[best], s, e))
            results[name] = {"picks": picks, "oos_trades": pd.concat(oos_parts) if oos_parts else None,
                             "trial_srs": [is_sr[k] for k in cands]}
        return results

    def template_trade_table(self) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
        """Trades for every playbook template configuration (structural stops, limits, indicator exits)."""
        regime = btc_regime(self.btc_daily)
        rows: dict[str, list[pd.DataFrame]] = {}
        groups: dict[str, list[str]] = {}
        for tpl in TEMPLATES:
            for key, tf, _p in tpl.configs():
                rows.setdefault(key, [])
                groups.setdefault(f"tpl:{tpl.name}|{tf}", []).append(key)
        for tf in sorted({tf for t in TEMPLATES for tf in t.tfs}):
            for coin, df in self.data[tf].items():
                f = features(df[df.index >= HISTORY_START - pd.Timedelta(days=40)])
                idx = f.index.normalize()
                f["d_trend"] = self.daily_trend.get(coin, pd.Series(dtype=bool)).reindex(idx).fillna(
                    False).to_numpy(dtype=bool)
                reg = align_daily(regime, f.index)
                months = f.index.tz_localize(None).to_period("M")
                elig = np.array([coin in self.eligible.get(m, ()) for m in months], dtype=bool)
                o, h, lo, cl = (f[k].to_numpy() for k in ("open", "high", "low", "close"))
                a = f["atr"].to_numpy()
                for tpl in TEMPLATES:
                    if tf not in tpl.tfs:
                        continue
                    for key, ktf, p in tpl.configs():
                        if ktf != tf:
                            continue
                        plan = tpl.fn(f, p, reg)
                        keep = elig[plan.sig_idx] & (plan.sig_idx >= WARMUP)
                        plan.sig_idx, plan.stop = plan.sig_idx[keep], plan.stop[keep]
                        if plan.limit is not None:
                            plan.limit = plan.limit[keep]
                        t = structured_trades(o, h, lo, cl, a, plan, self.costs)
                        if t["entry"].size == 0:
                            continue
                        ein, eout = t["entry"].astype(int), t["exit"].astype(int)
                        rows[key].append(pd.DataFrame({
                            "coin": coin, "t_in": f.index[ein], "t_out": f.index[eout] + pd.Timedelta(TF_DELTA[tf]),
                            "ret": t["ret"], "stop_pct": t["stop_pct"], "reason": t["reason"],
                        "regime": self._regime_at(coin, f.index[ein])}))
        empty = pd.DataFrame(columns=["coin", "t_in", "t_out", "ret", "stop_pct", "reason"])
        return {k: (pd.concat(v, ignore_index=True) if v else empty) for k, v in rows.items()}, groups

    def grid_candidates(self, trials: list[float]) -> list[dict[str, Any]]:
        """#6 grid bot on a fixed blue-chip basket, each configuration reported out of sample."""
        basket = [s for s in ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT") if s in self.data["1h"]]
        out = []
        for levels, stop_below in ((10, 0.05), (20, 0.10)):
            hourly = pd.concat([grid_returns(self.data["1h"][s][self.data["1h"][s].index >= HISTORY_START],
                                             levels, stop_below, self.costs) for s in basket], axis=1).mean(axis=1)
            daily = (1 + hourly).groupby(hourly.index.normalize()).prod() - 1
            oos = daily[daily.index >= FIRST_TEST]
            st = oos - (daily.abs() * 0)  # costs already per fill; stress handled by fee in grid
            out.append(self.evaluate(f"tpl:grid|1h|niveis={levels},stop={stop_below}", oos, trials,
                                     returns_metrics(st, self.rules.equity0)["profit_factor"], [],
                                     f"grid {levels} níveis, re-armado mensalmente, cesta {'/'.join(basket)}"))
        return out

    def benchmarks(self, ex: Any) -> dict[str, Any]:
        """#10 DCA and #5 funding-rate carry are measured, not traded (spot-only mandate)."""
        out: dict[str, Any] = {}
        d = self.data["1d"]
        out["dca_btc"] = dca_benchmark(d, FIRST_TEST, self.end, {"BTC/USDT": 1.0})
        if "ETH/USDT" in d:
            out["dca_btc_eth"] = dca_benchmark(d, FIRST_TEST, self.end, {"BTC/USDT": 0.5, "ETH/USDT": 0.5})
        btc = d["BTC/USDT"]["close"].loc[FIRST_TEST:self.end]
        out["buy_hold_btc"] = {"total_return": float(btc.iloc[-1] / btc.iloc[0] - 1),
                               "max_drawdown": float(((btc.cummax() - btc) / btc.cummax()).max())}
        try:
            out["funding"] = funding_carry(FIRST_TEST, self.end)
        except Exception as exc:  # derivatives API may be unreachable; the benchmark is optional
            out["funding"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        return out

    # ------------------------------------------------------------------ evaluation

    def evaluate(self, name: str, oos_daily_or_trades: Any, trial_srs: list[float], stressed_pf: float,
                 picks: list[dict[str, Any]], live_config: str) -> dict[str, Any]:
        if isinstance(oos_daily_or_trades, pd.Series):
            m = returns_metrics(oos_daily_or_trades, self.rules.equity0)
            eq = self.rules.equity0 * (1 + oos_daily_or_trades).cumprod()
            monthly = eq.resample("ME").last().pct_change().dropna()
            worst_days = oos_daily_or_trades.nsmallest(3)
        else:
            res = simulate_portfolio(oos_daily_or_trades, self.rules, self.costs, FIRST_TEST, self.end)
            m = res.metrics
            daily = res.equity.groupby(res.equity.index.normalize()).last()
            daily = daily.reindex(pd.date_range(FIRST_TEST, self.end, freq="D", tz="UTC")).ffill().fillna(
                self.rules.equity0)
            monthly = daily.resample("ME").last().pct_change().dropna()
            worst_days = daily.pct_change().nsmallest(3)
        regimes = {} if isinstance(oos_daily_or_trades, pd.Series) else self.by_regime(oos_daily_or_trades)
        dsr = deflated_sharpe(m["daily_sr"], m["days"], m["daily_skew"], m["daily_kurt"], trial_srs) \
            if m["days"] > 2 and np.isfinite(m["daily_sr"]) else 0.0
        if not isinstance(oos_daily_or_trades, pd.Series) and len(oos_daily_or_trades):
            stop = float(oos_daily_or_trades["stop_pct"].median()) + self.costs.round_trip
            m["min_capital_usd"] = 5.0 * stop / RiskRules().risk_per_trade
            m["median_stop_pct"] = float(oos_daily_or_trades["stop_pct"].median())
            for risk in (0.01, 0.015):  # sensitivity: same signals, bigger risk per trade
                alt = simulate_portfolio(oos_daily_or_trades, RiskRules(equity0=self.rules.equity0,
                                                                        risk_per_trade=risk),
                                         self.costs, FIRST_TEST, self.end).metrics
                m[f"monthly_mean_risk{risk}"] = alt["monthly_mean"]
                m[f"max_drawdown_risk{risk}"] = alt["max_drawdown"]
        else:
            m["min_capital_usd"] = 5.0 * 3  # rotation: top-k slots of at least $5
        m["dsr"] = dsr
        m["stressed_profit_factor"] = stressed_pf
        fails = []
        for k, lim in CRITERIA.items():
            v = m.get(k, 0.0)
            bad = v > lim if k == "max_drawdown" else v < lim
            if bad:
                fails.append(f"{k}={v:.3f} (limite {lim})")
        return {"name": name, "metrics": m, "approved": not fails, "fails": fails, "picks": picks,
                "by_regime": regimes,
                "live_config": live_config, "monte_carlo": monte_carlo(monthly),
                "regimes": regime_breakdown(monthly, self.btc_daily),
                "monthly": {str(k.date()): round(float(v), 5) for k, v in monthly.items()},
                "worst_days": {str(k.date()): round(float(v), 5) for k, v in worst_days.items()}}

    def run(self, ex: Any = None) -> dict[str, Any]:
        t0 = time.time()
        self.load(ex)
        configs = build_configs()
        table = self.bracket_trade_table(configs)
        # doubling costs = subtracting one more round trip from every net return
        extra = self.costs.round_trip
        stressed = {k: v.assign(ret=v["ret"] - extra) for k, v in table.items()}
        log.info("trade tables built: %d configs in %.0fs", len(configs), time.time() - t0)
        groups: dict[str, list[str]] = {}
        for c in configs:
            groups.setdefault(f"{c.family}|{c.tf}", []).append(c.key)
        tpl_table, tpl_groups = self.template_trade_table()
        table.update(tpl_table)
        stressed.update({k: v.assign(ret=v["ret"] - extra) for k, v in tpl_table.items()})
        groups.update(tpl_groups)
        log.info("template tables built: %d configs", len(tpl_table))
        wf = self.walk_forward(groups, table)
        candidates: list[dict[str, Any]] = []
        routed = self.regime_router(wf, table)
        if routed is not None and len(routed):
            st_r = simulate_portfolio(routed.assign(ret=routed["ret"] - extra), self.rules,
                                      Costs(self.costs.fee * 2, self.costs.slippage * 2), FIRST_TEST,
                                      self.end).metrics["profit_factor"]
            all_srs = [x for r in wf.values() for x in r["trial_srs"]]
            candidates.append(self.evaluate("roteador-regime", routed, all_srs, st_r,
                                            [{"setups_usados": sorted(routed["setup"].unique().tolist())}],
                                            "cada setup só opera nos regimes onde provou vantagem no passado"))
        for name, r in wf.items():
            if r["oos_trades"] is None or r["oos_trades"].empty:
                continue
            last_cfg = r["picks"][-1]["config"]
            st_oos = pd.concat([window_slice(stressed[p["config"]], p["start"], p["end"]) for p in r["picks"]])
            st_res = simulate_portfolio(st_oos, self.rules, Costs(self.costs.fee * 2, self.costs.slippage * 2),
                                        FIRST_TEST, self.end)
            for p in r["picks"]:
                p.pop("start"), p.pop("end")
            candidates.append(self.evaluate(name, r["oos_trades"], r["trial_srs"],
                                            st_res.metrics["profit_factor"], r["picks"], last_cfg))
        # Ensemble fixed *a priori* (before seeing v4 results): every confluence/MTF family runs together
        # on one book, each with its own walk-forward picks. More signals per day, shared risk limits.
        members = [n for n in wf if n.startswith(("confluence|", "mtf|"))]
        parts = [wf[n]["oos_trades"] for n in members if wf[n]["oos_trades"] is not None]
        if parts:
            ens = pd.concat(parts, ignore_index=True)
            st_ens = ens.assign(ret=ens["ret"] - extra)
            st_pf = simulate_portfolio(st_ens, self.rules, Costs(self.costs.fee * 2, self.costs.slippage * 2),
                                       FIRST_TEST, self.end).metrics["profit_factor"]
            trial_srs = [x for n in members for x in wf[n]["trial_srs"]]
            candidates.append(self.evaluate("ensemble-confluencia", ens, trial_srs, st_pf,
                                            [{"members": members}], " + ".join(members)))
        all_rot = rotation_configs()
        candidates.append(self.rotation("rotation|1d", all_rot, all_rot))
        candidates.append(self.rotation("rotation-safe|1d", [c for c in all_rot if c.trend_exit], all_rot))
        candidates.append(self.rotation("rotation-dominancia|1d", [c for c in all_rot if c.dominance], all_rot))
        grid_trials = [x for r in wf.values() for x in r["trial_srs"]]
        candidates.extend(self.grid_candidates(grid_trials))
        all_trials = sum(len(r["trial_srs"]) for r in wf.values()) + len(rotation_configs()) + 2
        candidates.sort(key=lambda x: (x["approved"], x["metrics"]["sortino"]), reverse=True)
        report = {"version": VERSION, "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
                  "elapsed_sec": round(time.time() - t0), "pool": self.pool, "trials": all_trials,
                  "oos_period": [str(FIRST_TEST.date()), str(self.end.date())],
                  "rules": asdict(self.rules), "costs": asdict(self.costs), "criteria": CRITERIA,
                  "candidates": candidates, "benchmarks": self.benchmarks(ex),
                  "playbook_verdicts": PLAYBOOK_NOTES}
        return report

    # ------------------------------------------------------------------ rotation

    def rotation(self, name: str, cfgs: list[RotationConfig], trials: list[RotationConfig]) -> dict[str, Any]:
        closes = pd.DataFrame({c: d["close"] for c, d in self.data["1d"].items()}).sort_index()
        closes = closes[closes.index >= HISTORY_START - pd.Timedelta(days=120)]
        btc = closes["BTC/USDT"]
        if not hasattr(self, "_rot_cache"):
            self._rot_cache = {c.key: rotation_returns(closes, btc, self.eligible, c, self.costs) for c in trials}
        series = self._rot_cache
        stressed_costs = Costs(self.costs.fee * 2, self.costs.slippage * 2)
        is_sr: dict[str, float] = {}
        for k, s in series.items():
            tr = s[(s.index >= HISTORY_START) & (s.index < FIRST_TEST)]
            is_sr[k] = float(tr.mean() / tr.std(ddof=1)) if tr.std(ddof=1) > 0 else 0.0
        oos, st_oos, picks = [], [], []
        by_key = {c.key: c for c in cfgs}
        def train_sr(key: str, until: pd.Timestamp) -> float:
            x = series[key][(series[key].index >= HISTORY_START) & (series[key].index < until)]
            sd = float(x.std(ddof=1))
            return float(x.mean()) / sd if sd > 0 else -math.inf

        for s, e in test_windows(self.end):
            best = max(cfgs, key=lambda c: train_sr(c.key, s))
            picks.append({"window": f"{s.date()}→{e.date()}", "config": best.key})
            seg = series[best.key]
            oos.append(seg[(seg.index >= s) & (seg.index < e)])
            st = rotation_returns(closes, btc, self.eligible, by_key[best.key], stressed_costs)
            st_oos.append(st[(st.index >= s) & (st.index < e)])
        oos_s = pd.concat(oos)
        st_m = returns_metrics(pd.concat(st_oos), self.rules.equity0)
        return self.evaluate(name, oos_s, list(is_sr.values()), st_m["profit_factor"], picks,
                             picks[-1]["config"])


TF_DELTA = {"1h": "1h", "4h": "4h", "1d": "1D"}

PLAYBOOK_NOTES = {
    "1_range_breakout": "testado (tpl:range_breakout)",
    "2_golden_cross": "testado (tpl:golden_cross)",
    "3_rsi_divergence": "testado só o lado comprado (tpl:rsi_divergence); vender a descoberto exige futuros",
    "4_orderbook_scalping": "não testável: não existe histórico público do livro de ofertas; alvos de 0,2–0,5% "
                            "contra 0,2–0,3% de custo por ida e volta não deixam margem no varejo",
    "5_funding_arbitrage": "medido (benchmarks.funding); exige futuros + spot, fora do mandato spot-only",
    "6_spot_grid": "testado (tpl:grid)",
    "7_support_flip": "testado (tpl:support_flip)",
    "8_bollinger_squeeze": "testado nas rodadas anteriores (squeeze 15m/1h/4h/1d); só o rompimento de alta",
    "9_altcoin_rotation": "testado (rotation-dominancia: proxy de BTC.D = alts vs BTC em 30 dias)",
    "10_smart_dca": "medido como referência (benchmarks.dca_*), não é setup de trade",
}


def funding_carry(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    """Historical funding of the delta-neutral carry (long spot + short perp), net of 4 legs of costs."""
    import ccxt

    fx = ccxt.binanceusdm({"enableRateLimit": True})
    out: dict[str, Any] = {}
    for sym in ("BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"):
        rates: list[tuple[int, float]] = []
        since = int(start.timestamp() * 1000)
        while since < int(end.timestamp() * 1000):
            batch = fx.fetch_funding_rate_history(sym, since=since, limit=1000)
            if not batch:
                break
            rates += [(int(r["timestamp"]), float(r["fundingRate"])) for r in batch]
            since = int(batch[-1]["timestamp"]) + 1
            if len(batch) < 1000:
                break
        if not rates:
            continue
        s = pd.Series([r for _, r in rates], index=pd.to_datetime([t for t, _ in rates], unit="ms", utc=True))
        years = max((s.index[-1] - s.index[0]).days / 365.0, 1e-9)
        gross = float(s.sum())
        net = gross - 4 * (0.001 + 0.0005)  # open + close on both legs, once over the whole period
        monthly = s.resample("ME").sum()
        out[sym.split(":")[0]] = {"apr_gross": gross / years, "apr_net": net / years,
                                  "positive_months": float((monthly > 0).mean()),
                                  "worst_month": float(monthly.min()), "samples": len(s)}
    return out


def summarize(report: dict[str, Any]) -> list[str]:
    lines = [f"RESEARCH v{report['version']} | OOS {report['oos_period'][0]}→{report['oos_period'][1]} | "
             f"{report['trials']} configurações testadas | {len(report['pool'])} moedas | {report['elapsed_sec']}s"]
    for c in report["candidates"]:
        m = c["metrics"]
        lines.append(
            f"{'APROVADO ' if c['approved'] else 'reprovado'} {c['name']:<14} "
            f"trades/mês {m['trades'] / max(m['months'], 1):.1f} mês médio {m['monthly_mean']:+.2%} "
            f"mediana {m['monthly_median']:+.2%} pior {m['monthly_worst']:+.2%} meses+ {m['positive_months']:.0%} | "
            f"trades {m['trades']:.0f} acerto {m['win_rate']:.0%} PF {m['profit_factor']:.2f} payoff {m['payoff']:.2f} "
            f"DD {m['max_drawdown']:.1%} Sharpe {m['sharpe']:.2f} Sortino {m['sortino']:.2f} DSR {m['dsr']:.2f} "
            f"PFx2custo {m['stressed_profit_factor']:.2f} capmin ${m.get('min_capital_usd', 0):.0f} | "
            f"MC DD95 {c['monte_carlo'].get('dd_p95', 0):.1%} "
            f"ano p5 {c['monte_carlo'].get('year_p5', 0):+.1%} | falhas: {'; '.join(c['fails']) or '-'}")
        lines.append(f"    config ao vivo: {c['live_config']}")
        if "monthly_mean_risk0.01" in m:
            lines.append(f"    stop mediano {m['median_stop_pct']:.2%} | risco 1%: mês {m['monthly_mean_risk0.01']:+.2%} "
                         f"DD {m['max_drawdown_risk0.01']:.1%} | risco 1,5%: mês {m['monthly_mean_risk0.015']:+.2%} "
                         f"DD {m['max_drawdown_risk0.015']:.1%}")
        lines.append("    regimes: " + " | ".join(f"{k}: {v['months']}m média {v['mean']:+.2%} pos {v['positive']:.0%}"
                                                  for k, v in c["regimes"].items()))
        if c.get("by_regime"):
            lines.append("    setup×regime: " + " | ".join(
                f"{k}: n={v['trades']:.0f} acerto {v['win_rate']:.0%} PF {v['profit_factor']:.2f} "
                f"exp {v['expectancy_pct']:+.2%}" for k, v in sorted(c["by_regime"].items())))
    return lines


def save(report: dict[str, Any], root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "report.json"
    path.write_text(json.dumps(report, default=str, ensure_ascii=False))
    return path
