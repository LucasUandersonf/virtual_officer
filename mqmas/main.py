"""FastAPI entrypoint: wires every component and owns the task lifecycle.

    uvicorn mqmas.main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from mqmas.agents.orchestrator import Orchestrator
from mqmas.agents.personas import PERSONAS
from mqmas.book.engine import StrategyBook
from mqmas.config import Settings, get_settings
from mqmas.data.market_feed import MarketFeed, build_feed
from mqmas.execution.broker import build_broker
from mqmas.execution.position_manager import PositionManager
from mqmas.league.service import SetupLeague
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
    league: SetupLeague
    book: StrategyBook | None
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
    league = SetupLeague(s, db, feed, hub)
    await league.load_cached()
    orch = Orchestrator(s, db, feed, broker, router, risk, positions, hub, league)
    stop = asyncio.Event()
    tasks = [
        asyncio.create_task(orch.run(stop), name="orchestrator"),
        asyncio.create_task(positions.run(stop), name="position_manager"),
        asyncio.create_task(league.run(stop), name="setup_league"),
    ]
    book = StrategyBook(s, db, risk, router, hub, feed.exchange) if s.book_enabled and feed.exchange else None
    if book:
        tasks.append(asyncio.create_task(book.run(stop), name="strategy_book"))
    runtime = Runtime(s, db, feed, router, risk, hub, positions, orch, league, book, stop, tasks)
    if s.research_on_boot:
        tasks.append(asyncio.create_task(_research_once(s), name="research_lab"))
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


def _research_root(s: Settings) -> Path:
    return s.db_path.parent / "research"


async def _research_once(s: Settings) -> None:
    """Spawn the research lab in its own process (keeps the trading loop's GIL free)."""
    log.info("starting research lab subprocess (runs only the labs whose report is missing or outdated)")
    proc = await asyncio.create_subprocess_exec(sys.executable, "-m", "mqmas.research", str(_research_root(s)), "all")
    code = await proc.wait()
    log.info("research lab finished with exit code %s", code)


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


@app.get("/api/league")
async def league_report() -> dict[str, Any]:
    rt = _rt()
    report = rt.league.report
    return {"ready": report is not None, "running": rt.league.running, "age_hours": rt.league.age_hours(),
            "report": report.model_dump(mode="json") if report else None,
            "watching": [f.model_dump(mode="json") for f in rt.orchestrator.watching]}


@app.get("/api/book")
async def book_state() -> dict[str, Any]:
    rt = _rt()
    if rt.book is None:
        return {"enabled": False}
    return {"enabled": True, **await rt.book.snapshot()}


@app.get("/api/research")
async def research_report() -> dict[str, Any]:
    from mqmas.research.lab import summarize

    path = _research_root(_rt().settings) / "report.json"
    if not path.exists():
        return {"ready": False}
    report = json.loads(path.read_text())
    return {"ready": True, "summary": summarize(report), "report": report}


@app.get("/api/research/swing")
async def research_swing() -> dict[str, Any]:
    from mqmas.research.swing import summarize

    path = _research_root(_rt().settings) / "swing_report.json"
    if not path.exists():
        return {"ready": False}
    report = json.loads(path.read_text())
    return {"ready": True, "summary": summarize(report), "report": report}


@app.get("/api/ta/{base}")
async def technical_analysis(base: str, tf: str = "4h", quote: str = "USDT") -> dict[str, Any]:
    """Full technical report of BASE/QUOTE on ``tf`` (closed candles), annotated with the historical study."""
    from mqmas.ta.live import report as ta_report

    rt = _rt()
    if tf not in ("1h", "4h", "1d"):
        raise HTTPException(400, "tf must be 1h, 4h or 1d")
    ex = rt.feed.exchange
    if ex is None:
        raise HTTPException(503, "no exchange connection (synthetic feed)")
    symbol = f"{base.upper()}/{quote.upper()}"
    rows = await ex.fetch_ohlcv(symbol, tf, limit=400)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df.pop("ts").astype("int64"), unit="ms", utc=True)
    rep = await asyncio.to_thread(ta_report, df.iloc[:-1].astype(float), tf, _research_root(rt.settings))
    return {"symbol": symbol, "tf": tf, **rep}


@app.get("/api/levels/{base}")
async def price_levels(base: str, tf: str = "4h", quote: str = "USDT") -> dict[str, Any]:
    """Marked price map of BASE/QUOTE: confluence zones (support/resistance) and a long plan."""
    from mqmas.ta.levels import price_map

    rt = _rt()
    if tf not in ("1h", "4h", "1d"):
        raise HTTPException(400, "tf must be 1h, 4h or 1d")
    ex = rt.feed.exchange
    if ex is None:
        raise HTTPException(503, "no exchange connection (synthetic feed)")
    symbol = f"{base.upper()}/{quote.upper()}"
    rows = await ex.fetch_ohlcv(symbol, tf, limit=400)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df.pop("ts").astype("int64"), unit="ms", utc=True)
    return {"symbol": symbol, "tf": tf, **await asyncio.to_thread(price_map, df.iloc[:-1].astype(float))}


@app.get("/api/ta-study")
async def ta_study_report() -> dict[str, Any]:
    path = _research_root(_rt().settings) / "ta_report.json"
    if not path.exists():
        return {"ready": False}
    rep = json.loads(path.read_text())
    return {"ready": True, "rows": [{k: r[k] for k in ("padrao", "tf", "categoria", "vies", "ocorrencias", "veredito",
                                                         "operacao")} for r in rep["rows"]]}


@app.get("/api/research/season")
async def research_season() -> dict[str, Any]:
    path = _research_root(_rt().settings) / "season_report.json"
    if not path.exists():
        return {"ready": False}
    return {"ready": True, "report": json.loads(path.read_text())}


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
