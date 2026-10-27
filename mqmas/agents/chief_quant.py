"""Agent 1 — Chief Quant & Market Microstructure (Tree of Thoughts)."""

from __future__ import annotations

import json
from typing import Any

from mqmas.agents.base import Agent, snapshot_brief
from mqmas.llm import prompts
from mqmas.llm.router import LLMUnavailableError
from mqmas.models import AgentMessage, AgentRole, ChiefQuantOpinion, FiredSetup, QuantSnapshot, Vote


def normalise(op: ChiefQuantOpinion) -> ChiefQuantOpinion:
    total = sum(sc.probability for sc in op.scenarios) or 1.0
    for sc in op.scenarios:
        sc.probability = sc.probability / total
    return op


def setups_brief(fired: list[FiredSetup]) -> list[dict[str, Any]]:
    """What the league knows about each fired setup, rounded to save tokens."""
    return [{"setup": f.name, "familia": f.family, "tese": f.thesis,
             "alvo_pct": round(f.profile.tp_pct * 100, 2), "stop_atr_x": f.profile.stop_atr_mult,
             "oos_trades": f.oos.trades, "oos_acerto_pct": round(f.oos.win_rate * 100, 1),
             "oos_expectativa_pct": round(f.oos.expectancy_pct * 100, 3),
             "oos_profit_factor": round(f.oos.profit_factor, 2),
             "oos_drawdown_pct": round(f.oos.max_drawdown_pct * 100, 2)} for f in fired]


class ChiefQuant(Agent):
    role = AgentRole.CHIEF_QUANT
    system_prompt = prompts.CHIEF_QUANT

    async def deliberate(self, debate_id: str, snap: QuantSnapshot,
                         recent_debates: list[dict[str, Any]],
                         fired: list[FiredSetup] | None = None,
                         technical: dict[str, Any] | None = None) -> tuple[ChiefQuantOpinion | None, AgentMessage]:
        res = await self.ask({"snapshot": snapshot_brief(snap), "setups_disparados": setups_brief(fired or []),
                              "analise_tecnica": technical or {}, "ultimos_debates": recent_debates[:5]},
                             ChiefQuantOpinion)
        if res is None:
            return None, self.abstain(debate_id)
        op, meta = res
        op = normalise(op)
        probs = {sc.name: sc.probability for sc in op.scenarios}
        bull = probs.get("bullish", 0.0)
        # Code-level consistency check: an APPROVE must be backed by a dominant bullish branch.
        vote = op.vote if (op.vote is not Vote.APPROVE or bull >= max(probs.values())) else Vote.REJECT
        tree = " | ".join(f"{k}: {v:.0%}" for k, v in probs.items())
        return op, self.message(f"🌳 Árvore de cenários → {tree}. {op.thesis}", debate_id=debate_id,
                                vote=vote, provider=meta.provider,
                                data={"scenarios": [sc.model_dump() for sc in op.scenarios]})

    async def commentary(self, snap: QuantSnapshot) -> AgentMessage | None:
        if not self._router.enabled:
            return None
        try:
            res = await self._router.complete(prompts.COMMENTARY,
                                              json.dumps(snapshot_brief(snap), ensure_ascii=False),
                                              max_tokens=160, temperature=0.5)
        except LLMUnavailableError:
            return None
        return self.message(res.content.strip(), provider=res.provider)
