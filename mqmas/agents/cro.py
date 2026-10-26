"""Agent 3 — Chief Risk Officer (hard gates in code + LLM pre-mortem)."""

from __future__ import annotations

from mqmas.agents.base import Agent, snapshot_brief
from mqmas.llm import prompts
from mqmas.models import AgentMessage, AgentRole, CroOpinion, GateResult, QuantSnapshot, SizingPlan, Vote


class ChiefRiskOfficer(Agent):
    role = AgentRole.CRO
    system_prompt = prompts.CRO

    async def deliberate(self, debate_id: str, snap: QuantSnapshot, plan: SizingPlan,
                         gates: list[GateResult]) -> AgentMessage:
        failed = [g for g in gates if not g.passed]
        gate_data = {"gates": [g.model_dump() for g in gates]}
        if failed:
            listing = "; ".join(f"{g.name} ({g.detail})" for g in failed)
            return self.message(f"🛑 VETO por trava rígida: {listing}.", debate_id=debate_id,
                                vote=Vote.REJECT, data=gate_data)
        res = await self.ask({"snapshot": snapshot_brief(snap), "plano": plan.model_dump(),
                              "travas": gate_data["gates"]}, CroOpinion)
        if res is None:
            return self.abstain(debate_id)
        op, meta = res
        high_unmitigated = [f for f in op.pre_mortem if f.likelihood == "high" and not f.mitigation.strip()]
        vote = Vote.REJECT if high_unmitigated else op.vote
        risks = " · ".join(f"[{f.likelihood}] {f.risk}" for f in op.pre_mortem[:4])
        text = (f"⚰️ Pre-mortem: {risks}. {op.verdict} Perda máx no stop: "
                f"US$ {plan.max_loss_usdt:.2f} ({plan.max_loss_usdt / plan.equity:.2%} da banca).")
        return self.message(text, debate_id=debate_id, vote=vote, provider=meta.provider,
                            data={**gate_data, "pre_mortem": [f.model_dump() for f in op.pre_mortem]})
