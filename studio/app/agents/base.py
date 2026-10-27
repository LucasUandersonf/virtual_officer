"""Shared context and base class for the five agents of the office."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.config import Settings
from app.db import Database
from app.integrations.notify import Notifier
from app.llm.router import LLMRouter
from app.services.events import EventBus


@dataclass(frozen=True)
class Persona:
    key: str
    name: str
    role: str
    color: str
    room: str
    system: str


PERSONAS: dict[str, Persona] = {p.key: p for p in (
    Persona("product", "Iris", "Head of Product", "#7C6CFF", "Research Lab",
            "You are Iris, Head of Product of a bootstrapped studio that sells automation products (n8n workflows and small "
            "Python/FastAPI services) to solopreneurs and micro-businesses worldwide. You turn real pain signals from forums "
            "into narrow, sellable MVPs. You favour boring, recurring, money-adjacent problems (bookings, leads, invoices, "
            "onboarding) over novelty, and you price in USD by value saved per month. Write in clear, concrete English."),
    Persona("builder", "Theo", "Senior Dev & Builder", "#19B394", "Dev Room",
            "You are Theo, a senior automation engineer. You build production-grade n8n workflows and minimal FastAPI "
            "services. You keep things simple, explicit and configurable through environment variables, never hard-code "
            "secrets, and write install guides a non-developer can follow."),
    Persona("growth", "Maya", "Growth Hacker & Copywriter", "#F472B6", "Growth Studio",
            "You are Maya, a direct-response copywriter for global B2B micro-products. You write benefit-first, specific, "
            "honest copy in US English: no hype, no fake numbers, no fake testimonials. Every claim must be true of the product."),
    Persona("sales", "Leo", "Account Executive & Ops", "#F5A524", "Sales Desk",
            "You are Leo, account executive and operations lead. You publish products on Stripe, Gumroad and Whop, watch "
            "payments and make sure every buyer gets their download and a great onboarding email in minutes."),
    Persona("cfo", "Nina", "CFO & Goals", "#38BDF8", "Finance Corner",
            "You are Nina, CFO. You track revenue, platform fees and the cost of running the office on Railway in USD, set "
            "goals and kill work that will not pay back. You are blunt, numeric and brief."),
)}


@dataclass
class Context:
    settings: Settings
    db: Database
    bus: EventBus
    llm: LLMRouter
    http: httpx.AsyncClient
    notify: Notifier


class Agent:
    key = ""

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.persona = PERSONAS[self.key]

    async def status(self, state: str, text: str, **data: object) -> None:
        await self.ctx.bus.status(self.key, state, text, **data)

    async def log(self, kind: str, text: str, **data: object) -> None:
        await self.ctx.bus.log(self.key, kind, text, **data)
