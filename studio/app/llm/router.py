"""Free-tier LLM cascade: Groq -> Gemini -> OpenRouter (:free models) -> built-in playbook.

All three expose OpenAI-compatible ``/chat/completions``, so one transport
serves them. Two tiers per provider: ``smart`` (analysis, code, copy) and
``fast`` (short comments in meetings). Answers must be JSON matching a
Pydantic model; a provider that answers garbage counts as a failure and the
next one is tried. When every provider fails (or none is configured, or the
daily budget is spent) the caller's ``fallback`` runs — the office never
stops because a free tier is rate limited.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.config import Settings, secret
from app.db import Database
from app.models import LLMCall

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
OFFLINE = "playbook"


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    models: dict[str, str]
    json_mode: bool = True
    headers: dict[str, str] = field(default_factory=dict)
    failures: int = 0
    open_until: float = 0.0  # circuit breaker

    def available(self) -> bool:
        return time.monotonic() >= self.open_until


def build_providers(s: Settings) -> list[Provider]:
    out: list[Provider] = []
    if s.groq_api_key:
        out.append(Provider("groq", "https://api.groq.com/openai/v1", secret(s.groq_api_key),
                            {"smart": s.groq_model, "fast": s.groq_fast_model}))
    if s.gemini_api_key:
        out.append(Provider("gemini", "https://generativelanguage.googleapis.com/v1beta/openai",
                            secret(s.gemini_api_key), {"smart": s.gemini_model, "fast": s.gemini_fast_model}))
    if s.openrouter_api_key:
        out.append(Provider("openrouter", "https://openrouter.ai/api/v1", secret(s.openrouter_api_key),
                            {"smart": s.openrouter_model, "fast": s.openrouter_fast_model}, json_mode=False,
                            headers={"HTTP-Referer": s.public_base_url, "X-Title": "Product Studio"}))
    return out


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    text = _FENCE.sub("", text.strip())
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = _OBJ.search(text)
        if not m:
            raise
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON is not an object")
    return obj


class LLMRouter:
    def __init__(self, settings: Settings, db: Database | None = None,
                 providers: list[Provider] | None = None, client: httpx.AsyncClient | None = None) -> None:
        self.s = settings
        self.db = db
        self.providers = build_providers(settings) if providers is None else providers
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(settings.llm_timeout_sec, connect=8.0))
        self._day = date.today()
        self.calls_today = 0
        self.budget_factor = 1.0  # Nina halves it in austerity mode

    @property
    def enabled(self) -> bool:
        return bool(self.providers)

    def status(self) -> dict[str, Any]:
        return {
            "providers": [{"name": p.name, "available": p.available(), "models": p.models} for p in self.providers],
            "calls_today": self.calls_today, "daily_budget": int(self.s.llm_daily_call_budget * self.budget_factor),
            "cost_usd": 0.0,  # free tiers only
        }

    def _budget_ok(self) -> bool:
        if date.today() != self._day:
            self._day, self.calls_today = date.today(), 0
        return self.calls_today < int(self.s.llm_daily_call_budget * self.budget_factor)

    async def _record(self, agent: str, p: Provider, model: str, ok: bool, ms: int, tokens: int) -> None:
        if self.db:
            async with self.db.session() as s:
                s.add(LLMCall(agent=agent, provider=p.name, model=model, ok=ok, latency_ms=ms, tokens=tokens))

    async def _call(self, p: Provider, model: str, messages: list[dict[str, str]], max_tokens: int,
                    temperature: float) -> tuple[str, int]:
        body: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if p.json_mode:
            body["response_format"] = {"type": "json_object"}
        r = await self.client.post(f"{p.base_url}/chat/completions", json=body,
                                   headers={"Authorization": f"Bearer {p.api_key}", **p.headers})
        if r.status_code != 200:
            raise RuntimeError(f"{p.name} HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        return content, int((data.get("usage") or {}).get("total_tokens") or 0)

    async def json(self, agent: str, system: str, user: str, model_cls: type[T], fallback: Callable[[], T],
                   tier: str = "smart", max_tokens: int = 1500, temperature: float = 0.6) -> tuple[T, str]:
        """Returns (validated answer, provider name). Never raises: falls back to the playbook."""
        schema = json.dumps(model_cls.model_json_schema(), separators=(",", ":"))
        messages = [
            {"role": "system", "content": f"{system}\n\nReply with ONE JSON object only, matching this JSON schema:\n{schema}"},
            {"role": "user", "content": user},
        ]
        for p in self.providers:
            if not p.available() or not self._budget_ok():
                continue
            model = p.models.get(tier) or p.models["smart"]
            t0 = time.monotonic()
            self.calls_today += 1
            try:
                content, tokens = await self._call(p, model, messages, max_tokens, temperature)
                obj = model_cls.model_validate(extract_json(content))
            except (httpx.HTTPError, RuntimeError, ValueError, ValidationError, KeyError, IndexError) as exc:
                p.failures += 1
                if p.failures >= 3:
                    p.open_until = time.monotonic() + 300
                log.warning("LLM %s/%s failed for %s: %s", p.name, model, agent, str(exc)[:200])
                await self._record(agent, p, model, False, int((time.monotonic() - t0) * 1000), 0)
                continue
            p.failures, p.open_until = 0, 0.0
            await self._record(agent, p, model, True, int((time.monotonic() - t0) * 1000), tokens)
            return obj, p.name
        return fallback(), OFFLINE

    async def aclose(self) -> None:
        await self.client.aclose()
