from __future__ import annotations

import httpx
import pytest

from mqmas.llm.router import LLMRouter, LLMUnavailableError, ProviderConfig, extract_json
from mqmas.models import ExecutionOpinion, Vote


def _ok(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def _router(handler, **kw) -> LLMRouter:
    providers = [ProviderConfig("groq", "https://groq.test", "k", "m"),
                 ProviderConfig("gemini", "https://gemini.test", "k", "m"),
                 ProviderConfig("openrouter", "https://or.test", "k", "m", supports_json_mode=False)]
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return LLMRouter(providers, max_retries=kw.get("retries", 0), breaker_threshold=kw.get("threshold", 3), client=client)


def test_extract_json_handles_fences():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Claro! {"a": 2} espero ajudar') == {"a": 2}


@pytest.mark.asyncio
async def test_cascade_on_http_error():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.host)
        if req.url.host == "groq.test":
            return httpx.Response(429, text="rate limited")
        return _ok('{"plan": "ok", "vote": "APPROVE"}')

    r = _router(handler)
    op, meta = await r.structured("sys", "user", ExecutionOpinion)
    assert meta.provider == "gemini" and op.vote is Vote.APPROVE
    assert calls == ["groq.test", "gemini.test"]


@pytest.mark.asyncio
async def test_cascade_on_invalid_schema():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "or.test":
            return _ok('{"plan": "limpo", "vote": "REJECT"}')
        return _ok('{"not": "the schema"}')

    op, meta = await _router(handler).structured("s", "u", ExecutionOpinion)
    assert meta.provider == "openrouter" and op.vote is Vote.REJECT


@pytest.mark.asyncio
async def test_all_down_fails_closed_and_opens_breakers():
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    r = _router(handler, threshold=1)
    with pytest.raises(LLMUnavailableError):
        await r.complete("s", "u")
    assert all(not v["available"] for v in r.status().values())
    with pytest.raises(LLMUnavailableError, match="circuit open"):
        await r.complete("s", "u")


@pytest.mark.asyncio
async def test_retry_then_success():
    n = {"i": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        n["i"] += 1
        return httpx.Response(503, headers={"retry-after": "0"}) if n["i"] == 1 else _ok("olá")

    res = await _router(handler, retries=1).complete("s", "u")
    assert res.provider == "groq" and res.content == "olá" and n["i"] == 2
