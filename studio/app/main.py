"""FastAPI app: the office UI, the dashboard API, webhooks and (by default) the agents' worker loop."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agents.base import Context
from app.agents.office import Office
from app.api import dashboard, public
from app.config import Settings, get_settings
from app.db import Database
from app.integrations.notify import Notifier
from app.llm.router import LLMRouter
from app.services.events import EventBus
from app.services.queue import RedisQueue, make_queue, scheduler, worker

STATIC = os.path.join(os.path.dirname(__file__), "web", "static")


def build_context(s: Settings) -> Context:
    db = Database(s.database_url)
    http = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=8.0), headers={"User-Agent": s.reddit_user_agent})
    return Context(settings=s, db=db, bus=EventBus(db), llm=LLMRouter(s, db), http=http, notify=Notifier(s, http))


def create_app(settings: Settings | None = None, run_agents: bool | None = None) -> FastAPI:
    s = settings or get_settings()
    logging.basicConfig(level=s.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        ctx = build_context(s)
        await ctx.db.init()
        office = Office(ctx)
        queue = make_queue(s.redis_url)
        app.state.settings, app.state.db, app.state.bus, app.state.llm = s, ctx.db, ctx.bus, ctx.llm
        app.state.office, app.state.queue = office, queue
        tasks: list[asyncio.Task[None]] = []
        # With Redis, a separate worker service can run the agents; this process then only relays its events.
        in_process = s.scheduler_enabled if run_agents is None else run_agents
        if isinstance(queue, RedisQueue):
            tasks.append(asyncio.create_task(queue.relay(ctx.bus.emit)))
            in_process = in_process and os.environ.get("AGENTS_IN_WEB", "true") == "true"
        if in_process:
            tasks += [asyncio.create_task(worker(queue, office)), asyncio.create_task(scheduler(queue, s.pipeline_interval_min))]
        yield
        for t in tasks:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t
        await ctx.llm.aclose()
        await ctx.http.aclose()
        await ctx.db.close()

    app = FastAPI(title="Product Studio — Autonomous Virtual Office", lifespan=lifespan)
    app.include_router(dashboard.router)
    app.include_router(public.router)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(os.path.join(STATIC, "index.html"))

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"ok": True, "last_tick": app.state.office.last_tick}

    @app.websocket("/ws")
    async def ws(sock: WebSocket) -> None:
        await sock.accept()
        bus: EventBus = app.state.bus
        q = bus.subscribe()
        try:
            await sock.send_json({"type": "hello", "agents": bus.agent_state, "meeting": bus.meeting})
            while True:
                await sock.send_json(await q.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            bus.unsubscribe(q)

    return app


app = create_app()
