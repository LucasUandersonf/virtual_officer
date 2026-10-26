"""Fault-tolerant LLM router: Groq → Gemini → OpenRouter.

All three providers expose OpenAI-compatible ``/chat/completions`` endpoints,
so one transport serves them all. Resilience layers, from inner to outer:

1. **Per-request retries** with exponential backoff + jitter on 429/5xx and
   network timeouts, honouring ``Retry-After``.
2. **Per-provider circuit breaker**: after N consecutive failures the
   provider is skipped for a cooldown window (no wasted latency on a dead
   upstream), then half-opened with a single probe.
3. **Cascade fallback** to the next provider, including when a provider
   answers but its output fails JSON/Pydantic validation.
4. **Fail-closed**: if every provider fails, ``LLMUnavailableError`` is
   raised and the orchestrator treats the debate as *no consensus* — the
   firm never trades because an LLM was down.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from mqmas.config import Settings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class LLMUnavailableError(RuntimeError):
    """Every provider in the cascade failed."""


class ProviderError(RuntimeError):
    def __init__(self, message: str, retryable: bool, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    model: str
    supports_json_mode: bool = True
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass
class CircuitBreaker:
    threshold: int
    cooldown: float
    failures: int = 0
    opened_at: float | None = None

    def available(self) -> bool:
        if self.opened_at is None:
            return True
        if time.monotonic() - self.opened_at >= self.cooldown:
            return True  # half-open: allow one probe
        return False

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = time.monotonic()


@dataclass(frozen=True)
class LLMResult:
    provider: str
    model: str
    content: str
    latency_ms: float


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict[str, object]:
    """Parse a JSON object even when wrapped in prose or ```json fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(text)
        if not match:
            raise
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise json.JSONDecodeError("top-level JSON is not an object", text, 0)
    return obj


def build_providers(s: Settings) -> list[ProviderConfig]:
    providers: list[ProviderConfig] = []
    if s.groq_api_key:
        providers.append(ProviderConfig(
            "groq", "https://api.groq.com/openai/v1", s.groq_api_key.get_secret_value(), s.groq_model))
    if s.gemini_api_key:
        providers.append(ProviderConfig(
            "gemini", "https://generativelanguage.googleapis.com/v1beta/openai",
            s.gemini_api_key.get_secret_value(), s.gemini_model))
    if s.openrouter_api_key:
        providers.append(ProviderConfig(
            "openrouter", "https://openrouter.ai/api/v1", s.openrouter_api_key.get_secret_value(),
            s.openrouter_model, supports_json_mode=False,
            extra_headers={"HTTP-Referer": "https://github.com/mq-mas", "X-Title": "MQ-MAS"}))
    return providers


class LLMRouter:
    def __init__(
        self,
        providers: list[ProviderConfig],
        timeout: float = 25.0,
        max_retries: int = 2,
        breaker_threshold: int = 3,
        breaker_cooldown: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._providers = providers
        self._max_retries = max_retries
        self._breakers = {p.name: CircuitBreaker(breaker_threshold, breaker_cooldown) for p in providers}
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=5.0))

    @classmethod
    def from_settings(cls, s: Settings) -> "LLMRouter":
        return cls(build_providers(s), s.llm_timeout_sec, s.llm_max_retries,
                   s.llm_breaker_threshold, s.llm_breaker_cooldown_sec)

    @property
    def enabled(self) -> bool:
        return bool(self._providers)

    def status(self) -> dict[str, dict[str, object]]:
        return {
            name: {"available": b.available(), "consecutive_failures": b.failures}
            for name, b in self._breakers.items()
        }

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _call_once(self, p: ProviderConfig, messages: list[dict[str, str]],
                         temperature: float, max_tokens: int, json_mode: bool) -> str:
        body: dict[str, object] = {
            "model": p.model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens,
        }
        if json_mode and p.supports_json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {p.api_key}", **p.extra_headers}
        try:
            resp = await self._client.post(f"{p.base_url}/chat/completions", json=body, headers=headers)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ProviderError(f"{p.name} transport: {type(exc).__name__}", retryable=True) from exc
        if resp.status_code != 200:
            retry_after: float | None = None
            if ra := resp.headers.get("retry-after"):
                try:
                    retry_after = float(ra)
                except ValueError:
                    retry_after = None
            raise ProviderError(
                f"{p.name} HTTP {resp.status_code}: {resp.text[:200]}",
                retryable=resp.status_code in RETRYABLE_STATUS, retry_after=retry_after)
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(f"{p.name} malformed response", retryable=True) from exc
        if not content:
            raise ProviderError(f"{p.name} empty content", retryable=True)
        return str(content)

    async def _call_with_retries(self, p: ProviderConfig, messages: list[dict[str, str]],
                                 temperature: float, max_tokens: int, json_mode: bool) -> str:
        for attempt in range(self._max_retries + 1):
            try:
                return await self._call_once(p, messages, temperature, max_tokens, json_mode)
            except ProviderError as exc:
                if not exc.retryable or attempt == self._max_retries:
                    raise
                delay = exc.retry_after if exc.retry_after is not None else 0.5 * 2**attempt
                delay = min(delay, 10.0) + random.uniform(0, 0.25)
                log.warning("%s (attempt %d) → retry in %.2fs", exc, attempt + 1, delay)
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def complete(self, system: str, user: str, *, temperature: float = 0.3,
                       max_tokens: int = 700, json_mode: bool = False) -> LLMResult:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        errors: list[str] = []
        for p in self._providers:
            breaker = self._breakers[p.name]
            if not breaker.available():
                errors.append(f"{p.name}: circuit open")
                continue
            t0 = time.perf_counter()
            try:
                content = await self._call_with_retries(p, messages, temperature, max_tokens, json_mode)
            except ProviderError as exc:
                breaker.record_failure()
                errors.append(str(exc))
                log.warning("provider %s failed, cascading: %s", p.name, exc)
                continue
            breaker.record_success()
            return LLMResult(p.name, p.model, content, (time.perf_counter() - t0) * 1000)
        raise LLMUnavailableError("; ".join(errors) or "no LLM provider configured")

    async def structured(self, system: str, user: str, schema: type[T], *,
                         temperature: float = 0.3, max_tokens: int = 900) -> tuple[T, LLMResult]:
        """Complete and validate against ``schema``; invalid output cascades onward."""
        schema_hint = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        sys_prompt = (f"{system}\n\nRespond with ONE JSON object only, no prose, matching this "
                      f"JSON Schema:\n{schema_hint}")
        errors: list[str] = []
        for p in self._providers:
            breaker = self._breakers[p.name]
            if not breaker.available():
                errors.append(f"{p.name}: circuit open")
                continue
            t0 = time.perf_counter()
            try:
                content = await self._call_with_retries(
                    p, [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
                    temperature, max_tokens, json_mode=True)
                parsed = schema.model_validate(extract_json(content))
            except (ProviderError, json.JSONDecodeError, ValidationError) as exc:
                breaker.record_failure()
                errors.append(f"{p.name}: {type(exc).__name__}: {str(exc)[:160]}")
                log.warning("provider %s unusable for %s: %s", p.name, schema.__name__, exc)
                continue
            breaker.record_success()
            return parsed, LLMResult(p.name, p.model, content, (time.perf_counter() - t0) * 1000)
        raise LLMUnavailableError("; ".join(errors) or "no LLM provider configured")
