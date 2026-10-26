"""FastAPI entrypoint: wires every component and owns the task lifecycle.

    uvicorn mqmas.main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import asyncio
import hmac
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from mqmas.agents.orchestrator import Orchestrator
from mqmas.agents.personas import PERSONAS
from mqmas.config import Settings, get_settings
from mqmas.data.market_feed import MarketFeed, build_feed
from mqmas.execution.broker import build_broker
from mqmas.execution.position_manager import PositionManager
from mqmas.llm.router import LLMRouter
from mqmas.logging_setup import configure_logging
from mqmas.persistence.db import Database
from mqmas.risk.constraints import RiskManager
from mqmas.ws.hub import WebSocketHub

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "web" / "static"


@dataclass
class Runtime:
    settings: Settings
    db: Database
    feed: MarketFeed
    router: LLMRouter
    risk: RiskManager
    hub: WebSocketHub
    positions: PositionManager
    orchestrator: Orchestrator
    stop: asyncio.Event
    tasks: list[asyncio.Task[None]]


hub = WebSocketHub()
runtime: Runtime | None = None


def _rt() -> Runtime:
    if runtime is None:
        raise HTTPException(503, "runtime not ready")
    return runtime


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global runtime
    s = get_settings()
    configure_logging(s.log_level)
    log.info("starting %s in %s mode on %s %s", s.app_name, s.trading_mode.value, s.exchange_id, s.symbol)

    db = Database(s.db_path)
    await db.connect()
    feed = build_feed(s)
    await feed.start()
    router = LLMRouter.from_settings(s)
    if not router.enabled:
        log.warning("no LLM API key configured: every debate will end in ABSTAIN (fail-closed)")
    risk = RiskManager(s, db)
    broker = build_broker(s, feed)
    positions = PositionManager(s, db, broker, feed, risk, hub)
    await positions.reconcile_on_boot()
    orch = Orchestrator(s, db, feed, broker, router, risk, positions, hub)
    stop = asyncio.Event()
    tasks = [
        asyncio.create_task(orch.run(stop), name="orchestrator"),
        asyncio.create_task(positions.run(stop), name="position_manager"),
    ]
    runtime = Runtime(s, db, feed, router, risk, hub, positions, orch, stop, tasks)
    try:
        yield
    finally:
        stop.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await feed.stop()
        await router.aclose()
        await db.close()
        runtime = None
        log.info("shutdown complete")


app = FastAPI(title="MQ-MAS", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    rt = _rt()
    return {"status": "ok", "mode": rt.settings.trading_mode.value, "feed_stale": rt.feed.is_stale(),
            "ws_clients": rt.hub.client_count, "llm": rt.router.status()}


@app.get("/api/state")
async def state() -> dict[str, Any]:
    rt = _rt()
    mode = rt.settings.trading_mode.value
    lock = await rt.risk.lock_until()
    n, w = await rt.db.win_loss_counts(mode)
    return {
        "mode": mode,
        "symbol": rt.settings.symbol,
        "equity": await rt.risk.equity_estimate(),
        "trades": n,
        "wins": w,
        "lock_until": lock.isoformat() if lock else None,
        "kill_switch": await rt.risk.kill_switch_engaged(),
        "open_positions": [p.model_dump(mode="json") for p in await rt.db.active_positions(mode)],
        "last_snapshot": rt.orchestrator.last_snapshot.model_dump(mode="json") if rt.orchestrator.last_snapshot else None,
        "last_decision": rt.orchestrator.last_decision.model_dump(mode="json") if rt.orchestrator.last_decision else None,
    }


@app.get("/api/trades")
async def trades(limit: int = 100) -> list[dict[str, Any]]:
    rt = _rt()
    closed = await rt.db.closed_positions(rt.settings.trading_mode.value, limit=min(limit, 500))
    return [p.model_dump(mode="json") for p in closed]


@app.get("/api/debates")
async def debates(limit: int = 20) -> list[dict[str, Any]]:
    return await _rt().db.recent_debates(min(limit, 200))


@app.post("/api/kill-switch")
async def kill_switch(engage: bool = True, x_admin_token: str | None = Header(default=None)) -> dict[str, Any]:
    rt = _rt()
    expected = rt.settings.admin_token.get_secret_value() if rt.settings.admin_token else None
    if not expected or not x_admin_token or not hmac.compare_digest(expected, x_admin_token):
        raise HTTPException(401, "invalid admin token")
    await rt.risk.set_kill_switch(engage)
    await rt.hub.publish("risk", {"event": "kill_switch", "engaged": engage})
    return {"kill_switch": engage}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    rt = _rt()
    hello = {
        "app": rt.settings.app_name, "mode": rt.settings.trading_mode.value, "symbol": rt.settings.symbol,
        "agents": [{"role": p.role.value, "name": p.name, "title": p.title} for p in PERSONAS.values()],
    }
    await hub.serve(ws, hello)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
