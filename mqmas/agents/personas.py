"""Agent identities shown in the virtual office (names, desks, colours)."""

from __future__ import annotations

from dataclasses import dataclass

from mqmas.models import AgentRole


@dataclass(frozen=True)
class Persona:
    role: AgentRole
    name: str
    title: str


PERSONAS: dict[AgentRole, Persona] = {
    AgentRole.CHIEF_QUANT: Persona(AgentRole.CHIEF_QUANT, "Ada", "Chief Quant & Microstructure"),
    AgentRole.ECONOMETRICIAN: Persona(AgentRole.ECONOMETRICIAN, "Gauss", "Econometrista & Probabilidade"),
    AgentRole.CRO: Persona(AgentRole.CRO, "Vera", "Chief Risk Officer"),
    AgentRole.EXECUTION: Persona(AgentRole.EXECUTION, "Hex", "Execution Specialist"),
    AgentRole.SYSTEM: Persona(AgentRole.SYSTEM, "Sistema", "Orquestrador"),
}
