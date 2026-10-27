"""Cross-sectional momentum rotation on daily bars (portfolio strategy).

Every ``rebalance`` days, rank the point-in-time universe by trailing
``lookback``-day return and hold the top ``k`` with positive momentum, equally
weighted, only while BTC is above its SMA (otherwise 100% cash/USDT).
Optional volatility targeting scales gross exposure down when the basket's
trailing volatility exceeds the target. Costs are charged on turnover.

Unlike the bracket setups this is an *exposure* strategy: there is no stop per
position, so its risk is controlled by the regime filter, diversification and
the volatility target instead of a fixed 0.6 % loss per trade.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mqmas.research.engine import Costs


@dataclass(frozen=True)
class RotationConfig:
    lookback: int
    top_k: int
    sma: int
    vol_target: float | None
    rebalance: int = 7
    trend_exit: bool = False  # drop a holding as soon as it closes below its own SMA20
    dominance: bool = False  # alts only while they beat BTC over the lookback (BTC.D falling), else BTC

    @property
    def key(self) -> str:
        return (f"rotation|1d|lookback={self.lookback},top={self.top_k},sma={self.sma},"
                f"voltarget={self.vol_target},rebal={self.rebalance},trendexit={self.trend_exit},dominance={self.dominance}")


def rotation_configs() -> list[RotationConfig]:
    v1 = [RotationConfig(lb, k, sma, vt) for lb, k, sma, vt in
          itertools.product((14, 30, 60), (1, 3, 5), (50, 100), (None, 0.5))]
    v2 = [RotationConfig(lb, k, sma, vt, 7, te) for lb, k, sma, vt, te in
          itertools.product((30, 60, 90), (3, 5), (100, 200), (0.25, 0.4), (False, True))]
    v3 = [RotationConfig(lb, k, 100, 0.4, 7, True, True) for lb, k in itertools.product((30, 60), (3, 5))]
    return v1 + v2 + v3


def rotation_returns(closes: pd.DataFrame, btc_close: pd.Series, eligible: dict[pd.Period, set[str]],
                     cfg: RotationConfig, costs: Costs) -> pd.Series:
    """Daily net portfolio returns (index = day the return is earned)."""
    rets = closes.pct_change()
    mom = closes / closes.shift(cfg.lookback) - 1
    below_trend = closes < closes.rolling(20).mean()
    regime = btc_close > btc_close.rolling(cfg.sma).mean()
    weights = pd.Series(0.0, index=closes.columns)
    out = np.zeros(len(closes))
    days = closes.index
    for i in range(max(cfg.lookback, cfg.sma) + 1, len(days)):
        d_prev = days[i - 1]
        r_today = rets.iloc[i].fillna(0.0)
        gross = float((weights * r_today).sum())
        cost = 0.0
        if (i - 1) % cfg.rebalance == 0:  # decide at the close of d_prev, trade into day i
            target = pd.Series(0.0, index=closes.columns)
            if bool(regime.get(d_prev, False)):
                pool = eligible.get(d_prev.tz_localize(None).to_period("M"), set())
                m = mom.loc[d_prev, [c for c in closes.columns if c in pool]].dropna()
                picks = m[m > 0].sort_values(ascending=False).head(cfg.top_k).index
                if cfg.dominance:
                    alts = m.drop(labels=[btc_close.name], errors="ignore")
                    btc_mom = mom.at[d_prev, btc_close.name] if btc_close.name in mom.columns else 0.0
                    alt_season = len(alts) > 0 and float(alts.median()) > float(btc_mom)
                    if not alt_season:  # BTC dominance rising: park in BTC instead of alts
                        picks = pd.Index([btc_close.name]) if btc_close.name in closes.columns else pd.Index([])
                if len(picks):
                    w = 1.0 / cfg.top_k  # unfilled slots stay in cash
                    target[picks] = w
                    if cfg.vol_target is not None:
                        basket = (rets.loc[:d_prev, picks].iloc[-30:] * w).sum(axis=1)
                        vol = float(basket.std(ddof=1)) * math.sqrt(365)
                        if vol > 0:
                            target *= min(1.0, cfg.vol_target / vol)
            turnover = float((target - weights).abs().sum())
            cost = turnover * (costs.fee + costs.slippage)
            gross = float((target * r_today).sum())
            weights = target
        elif cfg.trend_exit and (weights > 0).any():
            held = weights[weights > 0].index
            exits = [c for c in held if bool(below_trend.at[d_prev, c])]
            if exits:
                cost = float(weights[exits].sum()) * (costs.fee + costs.slippage)
                gross = float((weights.drop(exits) * r_today.drop(exits)).sum())
                weights[exits] = 0.0
        out[i] = gross - cost
        # drift weights with prices so turnover is measured honestly
        grown = weights * (1 + r_today)
        total = 1 + float((weights * r_today).sum())
        weights = grown / total if total > 0 else grown * 0
    return pd.Series(out, index=days)


def returns_metrics(daily: pd.Series, equity0: float = 50.0) -> dict[str, float]:
    eq = equity0 * (1 + daily).cumprod()
    months = eq.resample("ME").last()
    monthly = months.pct_change()
    if len(months):
        monthly.iloc[0] = months.iloc[0] / equity0 - 1
    peak = eq.cummax()
    sd = float(daily.std(ddof=1))
    downside = float(np.sqrt((np.minimum(daily, 0) ** 2).mean()))
    years = max(len(daily) / 365.0, 1e-9)
    final = float(eq.iloc[-1]) if len(eq) else equity0
    return {
        "trades": float((daily != 0).sum()), "win_rate": float((daily > 0).sum() / max((daily != 0).sum(), 1)),
        "profit_factor": float(daily[daily > 0].sum() / -daily[daily < 0].sum()) if (daily < 0).any() else 99.0,
        "payoff": float(daily[daily > 0].mean() / -daily[daily < 0].mean()) if (daily < 0).any() else 0.0,
        "expectancy_pct": float(daily.mean()),
        "max_drawdown": float(((peak - eq) / peak).max()) if len(eq) else 0.0,
        "sharpe": float(daily.mean()) / sd * math.sqrt(365) if sd > 0 else 0.0,
        "sortino": float(daily.mean()) / downside * math.sqrt(365) if downside > 0 else 0.0,
        "total_return": final / equity0 - 1, "cagr": (final / equity0) ** (1 / years) - 1 if final > 0 else -1.0,
        "monthly_mean": float(monthly.mean()), "monthly_median": float(monthly.median()),
        "monthly_worst": float(monthly.min()), "monthly_best": float(monthly.max()),
        "positive_months": float((monthly > 0).mean()), "months": float(len(monthly)),
        "daily_sr": float(daily.mean()) / sd if sd > 0 else 0.0,
        "daily_skew": float(daily.skew()), "daily_kurt": float(daily.kurt()) + 3, "days": float(len(daily)),
    }
