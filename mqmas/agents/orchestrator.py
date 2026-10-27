"""The firm's main loop: telemetry → (maybe) debate → unanimous consensus → order.

Token economy matters on free LLM tiers, so the expensive debate is convened
only when a setup *approved by the Setup League* (walk-forward tournament
over years of history) fires on a freshly closed candle. Several setups
firing together are presented as confluence. The rest of the time the office
stays alive with rules-based status lines plus a periodic LLM commentary.

Debate protocol (short-circuits on the first veto, which also saves tokens):

    Chief Quant (ToT) → Econometrician (stats) → CRO (hard gates + pre-mortem)
                      → Execution (slippage) → UNANIMITY? → limit order
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from mqmas.agents.chief_quant import ChiefQuant
from mqmas.agents.cro import ChiefRiskOfficer
from mqmas.agents.econometrician import Econometrician
from mqmas.agents.execution import ExecutionSpecialist
from mqmas.agents.personas import PERSONAS
from mqmas.config import Settings
from mqmas.data.market_feed import MarketFeed
from mqmas.execution.broker import Broker
from mqmas.execution.position_manager import PositionManager
from mqmas.league.service import SetupLeague
from mqmas.llm.router import LLMRouter
from mqmas.models import (
    AgentMessage,
    AgentRole,
    Decision,
    FiredSetup,
    GateResult,
    QuantSnapshot,
    SetupStats,
    SizingPlan,
    Vote,
)
from mqmas.persistence.db import Database
from mqmas.quant.evidence import BacktestResult, compute_setup_stats
from mqmas.quant.pipeline import QuantPipeline
from mqmas.quant.sizing import build_sizing_plan
from mqmas.risk.constraints import RiskManager
from mqmas.ws.hub import WebSocketHub

log = logging.getLogger(__name__)

SPEECH_PACING_SEC = 1.5  # lets the office render bubbles one after another


class Orchestrator:
    def __init__(self, settings: Settings, db: Database, feed: MarketFeed, broker: Broker,
                 router: LLMRouter, risk: RiskManager, positions: PositionManager,
                 hub: WebSocketHub, league: SetupLeague | None = None) -> None:
        self._s = settings
        self._db = db
        self._feed = feed
        self._broker = broker
        self._router = router
        self._risk = risk
        self._positions = positions
        self._hub = hub
        self._pipeline = QuantPipeline(settings)
        self.chief = ChiefQuant(router)
        self.econ = Econometrician(router)
        self.cro = ChiefRiskOfficer(router)
        self.execution = ExecutionSpecialist(router)
        self.league = league
        self._last_signal_bar: Any = None
        self.watching: list[FiredSetup] = []  # fired but not approved: shown, never traded
        self._last_debate = 0.0
        self._last_commentary = 0.0
        self._idle_turn = 0
        self.last_snapshot: QuantSnapshot | None = None
        self.last_decision: Decision | None = None

    # ------------------------------------------------------------------ helpers

    async def _say(self, m: AgentMessage, pace: bool = True) -> None:
        await self._db.save_message(m)
        await self._hub.publish("agent_message", m.model_dump(mode="json"))
        if pace:
            await asyncio.sleep(SPEECH_PACING_SEC)

    async def _system(self, text: str, debate_id: str | None = None) -> None:
        p = PERSONAS[AgentRole.SYSTEM]
        await self._say(AgentMessage(debate_id=debate_id, role=AgentRole.SYSTEM, agent_name=p.name,
                                     text=text, provider="rules"), pace=False)

    async def _evidence(self, snap: QuantSnapshot, lead: FiredSetup) -> tuple[SetupStats, SizingPlan]:
        """Out-of-sample league trades + forward (paper/live) trades of the same setup."""
        s = self._s
        stop_pct = min(max(lead.profile.stop_atr_mult * snap.atr_pct, s.stop_loss_min_pct), s.stop_loss_max_pct)
        tp_pct = lead.profile.tp_pct
        n_live, w_live = await self._db.win_loss_counts(setup=lead.key)
        oos = BacktestResult(trades=lead.oos.trades, wins=lead.oos.wins, net_returns=())
        stats = compute_setup_stats(oos, n_live, w_live, stop_pct, tp_pct, s)
        equity = await self._risk.equity_estimate()
        plan = build_sizing_plan(equity, snap.orderbook.best_ask, stop_pct, tp_pct, stats.kelly_applied, s)
        return stats, plan

    async def _memory(self) -> list[dict[str, Any]]:
        closed = await self._db.closed_positions(self._s.trading_mode.value, limit=10)
        return [{"entrada": round(p.entry_price, 2), "saida": round(p.exit_price or 0, 2),
                 "pnl_usdt": round(p.pnl_usdt or 0, 4), "motivo": p.exit_reason.value if p.exit_reason else None,
                 "fechado_em": p.closed_at.isoformat() if p.closed_at else None} for p in closed]

    # ------------------------------------------------------------------ cycle

    async def cycle(self) -> None:
        if self._feed.is_stale():
            await self._hub.publish("system", {"status": "stale_feed"})
            return
        df = self._feed.candles()
        bids, asks = self._feed.book()
        snap = await self._pipeline.compute_async(df, bids, asks)
        self.last_snapshot = snap
        equity = await self._risk.equity_estimate()
        lock = await self._risk.lock_until()
        await self._hub.publish("telemetry", {
            "snapshot": snap.model_dump(mode="json"), "equity": equity,
            "mode": self._s.trading_mode.value, "lock_until": lock.isoformat() if lock else None,
            "llm": self._router.status(),
            "spark": [round(float(c), 2) for c in df["close"].iloc[-96:]],
        })

        active = await self._db.active_positions(self._s.trading_mode.value)
        now = time.monotonic()
        fired = await self._new_signals(df)
        approved = [f for f in fired if f.approved]
        if approved and not active and lock is None and now - self._last_debate >= self._s.debate_cooldown_sec:
            self._last_debate = now
            await self.debate(snap, approved)
            return
        if now - self._last_commentary >= self._s.commentary_interval_sec:
            self._last_commentary = now
            if m := await self.chief.commentary(snap):
                await self._say(m, pace=False)
                return
        await self._idle_chatter(snap, bool(active), lock is not None)

    async def _new_signals(self, df: Any) -> list[FiredSetup]:
        """Evaluate the league once per closed candle (no tokens, a few ms of CPU)."""
        if self.league is None or len(df) < 2 or df.index[-2] == self._last_signal_bar:
            return []
        self._last_signal_bar = df.index[-2]
        fired = await asyncio.to_thread(self.league.fired, df)
        self.watching = [f for f in fired if not f.approved]
        return fired

    def _league_line(self) -> str:
        report = self.league.report if self.league else None
        if report is None:
            return "Liga de Setups ainda montando o ranking com o histórico — sem setup aprovado, sem operação."
        approved = report.approved
        if not approved:
            best = report.entries[0]
            return (f"Liga: 0/{len(report.entries)} setups aprovados no walk-forward. Melhor: {best.name} "
                    f"(acerto OOS {best.oos.win_rate:.0%}, {best.oos.expectancy_pct:+.3%}/trade). Mesa em espera.")
        names = ", ".join(e.name for e in approved[:3])
        return f"Liga: {len(approved)}/{len(report.entries)} setups aprovados ({names}). Vigiando cada candle de 15m."

    async def _idle_chatter(self, snap: QuantSnapshot, in_position: bool, locked: bool) -> None:
        """Deterministic status lines (zero tokens) so the office never goes quiet."""
        s, ob, fc = self._s, snap.orderbook, snap.forecast
        lines = [
            (self.chief, self._league_line() if self._idle_turn % 8 == 0 else
                         (f"Observando {', '.join(w.name for w in self.watching)} — disparou, mas não é aprovado na Liga."
                          if self.watching else
                          f"Z-Score(50) em {snap.zscore:+.2f}. Preço {'abaixo' if snap.price < snap.vwap else 'acima'} "
                          f"da VWAP ({snap.vwap:,.0f}).")),
            (self.econ, f"Regressão grau {fc.degree} (R² {fc.r_squared:.2f}) projeta {fc.expected_price:,.0f} "
                        f"em {fc.horizon} barras, IC95 [{fc.lower_95:,.0f}; {fc.upper_95:,.0f}]."),
            (self.cro, "Circuit breaker ATIVO — mesa bloqueada." if locked else
                       ("Posição aberta sob vigilância: stop nativo + monitor de 2s." if in_position else
                        f"Travas OK. Risco máx/trade {s.max_risk_per_trade_pct:.1%}, ATR {snap.atr_pct:.3%}.")),
            (self.execution, f"Book top-{s.ob_levels}: imbalance {ob.imbalance:+.2f}, spread {ob.spread_pct:.4%}, "
                             f"microprice {ob.microprice:,.2f}."),
        ]
        agent, text = lines[self._idle_turn % len(lines)]
        self._idle_turn += 1
        await self._hub.publish("agent_message", agent.message(text).model_dump(mode="json"))

    async def debate(self, snap: QuantSnapshot, fired: list[FiredSetup] | None = None) -> Decision:
        debate_id = uuid.uuid4().hex[:12]
        fired = sorted(fired or [], key=lambda f: f.score, reverse=True)
        lead = fired[0] if fired else None
        confluence = [f.key for f in fired]
        await self._hub.publish("debate_start", {"debate_id": debate_id, "zscore": snap.zscore, "price": snap.price,
                                                 "setups": [f.name for f in fired]})
        names = " + ".join(f.name for f in fired) or "nenhum setup"
        await self._system(f"Reunião convocada: {names} disparou no candle fechado"
                           + (f" — confluência de {len(fired)} setups." if len(fired) > 1 else "."), debate_id)
        votes: dict[str, Vote] = {}
        plan: SizingPlan | None = None
        stats: SetupStats | None = None
        gates: list[GateResult] = []
        limit: float | None = None

        def outcome(reason: str) -> Decision:
            return Decision(debate_id=debate_id, approved=False, reason=reason, gates=gates,
                            votes=votes, sizing=plan, setup=lead.key if lead else None, confluence=confluence)

        if lead is None:
            empty = outcome("Nenhum setup aprovado disparou")
            self.last_decision = empty
            await self._db.save_debate(empty, snap.model_dump_json())
            await self._hub.publish("decision", empty.model_dump(mode="json"))
            return empty

        # 1. Chief Quant — Tree of Thoughts over the fired setups
        _, msg = await self.chief.deliberate(debate_id, snap, await self._db.recent_debates(5), fired)
        votes[AgentRole.CHIEF_QUANT.value] = msg.vote or Vote.ABSTAIN
        await self._say(msg)
        decision: Decision | None = None
        if msg.vote is not Vote.APPROVE:
            decision = outcome("Chief Quant não endossou a tese")

        # 2. Econometrician — evidence + memory
        if decision is None:
            stats, plan = await self._evidence(snap, lead)
            msg = await self.econ.deliberate(debate_id, snap, stats, await self._memory(), self._s, lead)
            votes[AgentRole.ECONOMETRICIAN.value] = msg.vote or Vote.ABSTAIN
            await self._say(msg)
            if msg.vote is not Vote.APPROVE:
                decision = outcome("Evidência estatística insuficiente")

        # 3. CRO — hard constraints, then pre-mortem
        if decision is None and plan is not None and stats is not None:
            gates = await self._risk.evaluate_entry(
                snap, stats, plan, feed_stale=self._feed.is_stale(),
                min_notional=await self._broker.min_notional(),
                available_cash=await self._broker.available_cash(plan.equity),
                setup=lead, league_age_hours=self.league.age_hours() if self.league else None)
            msg = await self.cro.deliberate(debate_id, snap, plan, gates)
            votes[AgentRole.CRO.value] = msg.vote or Vote.ABSTAIN
            await self._say(msg)
            if msg.vote is not Vote.APPROVE:
                decision = outcome("Veto do CRO")

        # 4. Execution — slippage
        if decision is None and plan is not None:
            _, asks = self._feed.book()
            msg, limit = await self.execution.deliberate(debate_id, snap, plan, asks, self._s)
            votes[AgentRole.EXECUTION.value] = msg.vote or Vote.ABSTAIN
            await self._say(msg)
            if msg.vote is not Vote.APPROVE or limit is None:
                decision = outcome("Execução reprovada")

        if decision is None:
            unanimous = len(votes) == 4 and all(v is Vote.APPROVE for v in votes.values())
            decision = Decision(debate_id=debate_id, approved=unanimous,
                                reason="Consenso unânime" if unanimous else "Sem unanimidade",
                                gates=gates, votes=votes, sizing=plan, setup=lead.key, confluence=confluence)

        self.last_decision = decision
        await self._db.save_debate(decision, snap.model_dump_json())
        await self._hub.publish("decision", decision.model_dump(mode="json"))
        await self._system(("✅ APROVADO por unanimidade — enviando ordem LIMIT." if decision.approved
                            else f"❌ Sem operação: {decision.reason}."), debate_id)

        if decision.approved and plan is not None and limit is not None:
            position = await self._positions.open_position(debate_id, plan, limit, setup=lead.key)
            await self._system("Ordem executada e stop nativo armado." if position
                               else "Ordem não preenchida no prazo — cancelada sem exposição.", debate_id)
        return decision

    async def run(self, stop: asyncio.Event) -> None:
        await self._system(f"MQ-MAS online em modo {self._s.trading_mode.value.upper()} "
                           f"({len(self._router.status())} provedores LLM configurados).")
        while not stop.is_set():
            started = time.monotonic()
            try:
                await self.cycle()
            except Exception:
                log.exception("orchestrator cycle failed")
                await self._hub.publish("system", {"status": "cycle_error"})
            elapsed = time.monotonic() - started
            try:
                await asyncio.wait_for(stop.wait(), timeout=max(1.0, self._s.decision_interval_sec - elapsed))
            except TimeoutError:
                pass
