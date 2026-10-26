"""Hard risk constraints — deterministic Python, *not* LLM opinion.

The LLM agents can only ever **veto**. An entry is possible only if every
gate in this module passes *and* all four agents approve. No prompt, model
hallucination or provider outage can bypass these checks.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from mqmas.config import Settings
from mqmas.models import ExitReason, GateResult, QuantSnapshot, SetupStats, SizingPlan
from mqmas.persistence.db import Database, start_of_utc_day

log = logging.getLogger(__name__)

KILL_SWITCH_KEY = "kill_switch"
EPS = 1e-9


class RiskManager:
    def __init__(self, settings: Settings, db: Database) -> None:
        self._s = settings
        self._db = db

    @property
    def mode(self) -> str:
        return self._s.trading_mode.value

    # ------------------------------------------------------------------ circuit breakers

    async def kill_switch_engaged(self) -> bool:
        return (await self._db.get_kv(KILL_SWITCH_KEY)) == "1"

    async def set_kill_switch(self, engaged: bool) -> None:
        await self._db.set_kv(KILL_SWITCH_KEY, "1" if engaged else "0")
        await self._db.log_risk_event("kill_switch", {"engaged": engaged})

    async def consecutive_stops_today(self) -> int:
        trades = await self._db.closed_since(self.mode, start_of_utc_day())
        streak = 0
        for p in reversed(trades):  # most recent first
            if p.exit_reason is ExitReason.STOP_LOSS or (p.pnl_usdt is not None and p.pnl_usdt < 0):
                streak += 1
            else:
                break
        return streak

    async def on_trade_closed(self) -> datetime | None:
        """Called after every exit. Arms the global breaker when needed."""
        streak = await self.consecutive_stops_today()
        if streak >= self._s.consecutive_loss_limit:
            until = datetime.now(timezone.utc) + timedelta(hours=self._s.lockout_hours)
            await self._db.log_risk_event(
                "circuit_breaker", {"consecutive_stops": streak, "rule": "consecutive_loss_limit"}, until
            )
            log.warning("CIRCUIT BREAKER armed until %s (%d consecutive stops)", until, streak)
            return until
        day_pnl = await self._db.pnl_since(self.mode, start_of_utc_day())
        equity = await self.equity_estimate()
        if day_pnl <= -self._s.max_daily_loss_pct * equity:
            until = start_of_utc_day() + timedelta(days=1)
            await self._db.log_risk_event("daily_loss_limit", {"day_pnl": day_pnl}, until)
            return until
        return None

    async def lock_until(self) -> datetime | None:
        return await self._db.active_lock_until()

    async def equity_estimate(self) -> float:
        return self._s.initial_capital_usdt + await self._db.realised_pnl(self.mode)

    # ------------------------------------------------------------------ entry gates

    async def evaluate_entry(
        self,
        snap: QuantSnapshot,
        stats: SetupStats,
        plan: SizingPlan,
        *,
        feed_stale: bool,
        min_notional: float,
        available_cash: float,
    ) -> list[GateResult]:
        s = self._s
        g: list[GateResult] = []

        def gate(name: str, passed: bool, detail: str) -> None:
            g.append(GateResult(name=name, passed=bool(passed), detail=detail))

        lock = await self.lock_until()
        active = await self._db.active_positions(self.mode)
        kill = await self.kill_switch_engaged()

        gate("spot_only", True, "conta e ordens estritamente SPOT, sem margem/derivativos")
        gate("kill_switch", not kill, "kill switch manual " + ("ATIVO" if kill else "inativo"))
        gate("circuit_breaker", lock is None,
             f"bloqueado até {lock.isoformat()}" if lock else "sem bloqueio ativo")
        gate("single_position", not active, f"{len(active)} posição(ões) aberta(s)")
        gate("fresh_data", not feed_stale, "feed de mercado " + ("DEFASADO" if feed_stale else "ok"))
        gate("zscore_oversold", snap.zscore <= s.zscore_entry,
             f"z={snap.zscore:.2f} (limiar {s.zscore_entry})")
        gate("below_vwap", snap.price < snap.vwap,
             f"preço {snap.price:.2f} vs VWAP {snap.vwap:.2f}")
        vp = snap.volume_profile
        gate("structural_support", snap.price >= vp.value_area_low * (1 - s.stop_loss_max_pct / 2),
             f"VAL {vp.value_area_low:.2f} / POC {vp.poc:.2f}")
        gate("orderbook_pressure", snap.orderbook.imbalance >= s.ob_min_imbalance,
             f"imbalance {snap.orderbook.imbalance:+.3f} (mín {s.ob_min_imbalance:+.2f})")
        gate("spread", snap.orderbook.spread_pct <= s.max_spread_pct,
             f"spread {snap.orderbook.spread_pct:.5%}")
        fc = snap.forecast
        gate("forecast_supports",
             fc.expected_price > plan.stop_price and fc.upper_95 >= plan.take_profit_price,
             f"E[P]={fc.expected_price:.2f}, IC95=[{fc.lower_95:.2f}, {fc.upper_95:.2f}]")
        n = stats.backtest_trades + stats.live_trades
        gate("sample_size", n >= s.min_sample_size, f"n={n} (mín {s.min_sample_size})")
        gate("win_rate", stats.win_rate >= s.min_win_rate,
             f"p̂={stats.win_rate:.1%} (IC95 inf {stats.win_rate_lower_95:.1%}, mín {s.min_win_rate:.0%})")
        gate("trade_sharpe", stats.trade_sharpe >= s.min_trade_sharpe,
             f"Sharpe/trade={stats.trade_sharpe:.3f} (mín {s.min_trade_sharpe})")
        gate("kelly_positive", stats.kelly_full > 0, f"Kelly f*={stats.kelly_full:.4f}")
        gate("stop_bounds", s.stop_loss_min_pct - EPS <= plan.stop_pct <= s.stop_loss_max_pct + EPS,
             f"stop {plan.stop_pct:.3%}")
        gate("tp_bounds", s.take_profit_min_pct - EPS <= plan.take_profit_pct <= s.take_profit_max_pct + EPS,
             f"TP {plan.take_profit_pct:.3%}")
        max_loss_allowed = plan.equity * s.max_risk_per_trade_pct
        gate("max_risk", plan.max_loss_usdt <= max_loss_allowed + EPS,
             f"perda máx {plan.max_loss_usdt:.4f} USDT (teto {max_loss_allowed:.4f})")
        gate("min_notional", plan.notional >= min_notional,
             f"notional {plan.notional:.2f} (mín corretora {min_notional:.2f})")
        gate("no_leverage", plan.notional <= available_cash + EPS,
             f"notional {plan.notional:.2f} ≤ caixa {available_cash:.2f}")
        return g
