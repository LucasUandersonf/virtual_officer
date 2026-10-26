"""Agent 4 — Execution Specialist (slippage gate + limit order plan)."""

from __future__ import annotations

from collections.abc import Sequence

from mqmas.agents.base import Agent, snapshot_brief
from mqmas.config import Settings
from mqmas.llm import prompts
from mqmas.models import AgentMessage, AgentRole, ExecutionOpinion, QuantSnapshot, SizingPlan, Vote
from mqmas.quant.orderbook import Level, estimate_buy_slippage


class ExecutionSpecialist(Agent):
    role = AgentRole.EXECUTION
    system_prompt = prompts.EXECUTION

    @staticmethod
    def limit_price(snap: QuantSnapshot, s: Settings) -> float:
        """Marketable limit at the best ask, never above mid·(1+max_slippage)."""
        return min(snap.orderbook.best_ask, snap.orderbook.mid * (1 + s.max_slippage_pct))

    async def deliberate(self, debate_id: str, snap: QuantSnapshot, plan: SizingPlan,
                         asks: Sequence[Level], s: Settings) -> tuple[AgentMessage, float | None]:
        try:
            avg_fill, slip = estimate_buy_slippage(asks, plan.quantity)
        except ValueError as exc:
            return self.message(f"⚡ Book raso: {exc}. Não executo.", debate_id=debate_id,
                                vote=Vote.REJECT), None
        limit = self.limit_price(snap, s)
        data = {"slippage_pct": slip, "avg_fill": avg_fill, "limit_price": limit,
                "quantity": plan.quantity, "notional": plan.notional}
        if slip > s.max_slippage_pct:
            return self.message(f"⚡ Slippage estimado {slip:.4%} > limite {s.max_slippage_pct:.2%}. Abortado.",
                                debate_id=debate_id, vote=Vote.REJECT, data=data), None
        res = await self.ask({"snapshot": snapshot_brief(snap), "plano": plan.model_dump(),
                              "execucao": data}, ExecutionOpinion)
        if res is None:
            return self.abstain(debate_id), None
        op, meta = res
        text = (f"⚡ LIMIT BUY {plan.quantity:.5f} BTC @ {limit:.2f} (slippage est. {slip:.4%}). "
                f"Stop {plan.stop_price:.2f} / TP {plan.take_profit_price:.2f}. {op.plan}")
        return self.message(text, debate_id=debate_id, vote=op.vote, provider=meta.provider, data=data), limit
