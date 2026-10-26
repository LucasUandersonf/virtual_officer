"""Shared agent plumbing: persona, LLM access, fail-closed message helpers."""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel

from mqmas.agents.personas import PERSONAS
from mqmas.llm.router import LLMResult, LLMRouter, LLMUnavailableError
from mqmas.models import AgentMessage, AgentRole, QuantSnapshot, Vote

T = TypeVar("T", bound=BaseModel)


def snapshot_brief(s: QuantSnapshot) -> dict[str, Any]:
    """Compact, rounded view of the snapshot for prompts (saves tokens)."""
    ob, fc, vp = s.orderbook, s.forecast, s.volume_profile
    return {
        "preco": round(s.price, 2),
        "zscore_50": round(s.zscore, 3),
        "vwap": round(s.vwap, 2),
        "vwap_banda_inf": round(s.vwap_lower, 2),
        "poc": round(vp.poc, 2),
        "value_area": [round(vp.value_area_low, 2), round(vp.value_area_high, 2)],
        "atr_pct": round(s.atr_pct * 100, 3),
        "book_imbalance_top5": round(ob.imbalance, 3),
        "bid_ask_ratio": round(ob.bid_ask_ratio, 3),
        "depth_delta": round(ob.depth_delta, 3),
        "spread_pct": round(ob.spread_pct * 100, 4),
        "microprice": round(ob.microprice, 2),
        "regressao": {
            "grau": fc.degree, "r2": round(fc.r_squared, 3), "horizonte_barras_15m": fc.horizon,
            "preco_esperado": round(fc.expected_price, 2),
            "ic95": [round(fc.lower_95, 2), round(fc.upper_95, 2)],
            "retorno_esperado_pct": round(fc.expected_return_pct * 100, 3),
        },
    }


class Agent:
    role: AgentRole
    system_prompt: str

    def __init__(self, router: LLMRouter) -> None:
        self._router = router
        self.persona = PERSONAS[self.role]

    def message(self, text: str, *, debate_id: str | None = None, vote: Vote | None = None,
                provider: str | None = "rules", data: dict[str, Any] | None = None) -> AgentMessage:
        return AgentMessage(debate_id=debate_id, role=self.role, agent_name=self.persona.name,
                            text=text, vote=vote, provider=provider, data=data or {})

    async def ask(self, payload: dict[str, Any], schema: type[T]) -> tuple[T, LLMResult] | None:
        """Structured LLM call; ``None`` means every provider failed (→ abstain)."""
        if not self._router.enabled:
            return None
        try:
            return await self._router.structured(
                self.system_prompt, json.dumps(payload, ensure_ascii=False, default=str), schema)
        except LLMUnavailableError:
            return None

    def abstain(self, debate_id: str) -> AgentMessage:
        return self.message("LLMs indisponíveis no momento — por protocolo fail-closed, me abstenho.",
                            debate_id=debate_id, vote=Vote.ABSTAIN)
