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
from fastapi.responses import FileResponse, PlainTextResponse
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
    from mqmas.ops import Notifier, backup_loop, heartbeat_loop

    notifier = Notifier(s)
    hub.add_listener(notifier.on_event)
    core = [t for t in tasks if t.get_name() in ("orchestrator", "position_manager", "strategy_book")]
    tasks += [asyncio.create_task(notifier.run(stop), name="notifier"),
              asyncio.create_task(backup_loop(s, notifier, stop), name="backup"),
              asyncio.create_task(heartbeat_loop(s, lambda: not feed.is_stale() and not any(t.done() for t in core),
                                                 stop), name="heartbeat")]
    if notifier.enabled:
        await notifier.send(f"MQ-MAS iniciado em modo {s.trading_mode.value}"
                            + (" · carteira com ORDENS REAIS" if s.is_live and s.book_live_enabled else ""))
    if s.review_on_boot:
        tasks.append(asyncio.create_task(_review_once(s, db, router), name="review_on_boot"))
    runtime = Runtime(s, db, feed, router, risk, hub, positions, orch, league, book, stop, tasks)
    tasks.append(asyncio.create_task(_sync_library(s, db), name="pattern_library"))
    if s.research_on_boot:
        tasks.append(asyncio.create_task(_research_once(s, db), name="research_lab"))
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


async def _sync_library(s: Settings, db: Database) -> None:
    """Pattern catalog + measured record into SQLite (ta_patterns / ta_pattern_stats)."""
    from mqmas.ta.catalog import catalog, stats_rows

    try:
        path = _research_root(s) / "ta_report.json"
        study = json.loads(path.read_text()) if path.exists() else None
        await db.sync_pattern_library(await asyncio.to_thread(catalog), stats_rows(study))
    except Exception:
        log.exception("pattern library sync failed")


async def _review_once(s: Settings, db: Database, router: LLMRouter) -> None:
    """REVIEW_ON_BOOT: the team debates a trade report once; facts and opinions go to the log and the office."""
    from mqmas.agents.review import TradeReport, debate
    from mqmas.ta.live import load_study

    await asyncio.sleep(20)
    try:
        rep = TradeReport.model_validate_json(s.review_on_boot or "{}")
        res = await debate(router, rep, load_study(_research_root(s)))
        for m in res.pop("messages"):
            await db.save_message(m)
            await hub.publish("agent_message", m.model_dump(mode="json"))
        log.info("REVIEW fatos %s", json.dumps(res["fatos"], ensure_ascii=False))
        log.info("REVIEW evidencia %s", json.dumps(res["evidencia"], ensure_ascii=False))
        for d in res["debate"]:
            log.info("REVIEW %s", json.dumps(d, ensure_ascii=False))
    except Exception:
        log.exception("review on boot failed")


async def _research_once(s: Settings, db: Database) -> None:
    """Spawn the research lab in its own process (keeps the trading loop's GIL free)."""
    log.info("starting research lab subprocess (runs only the labs whose report is missing or outdated)")
    proc = await asyncio.create_subprocess_exec(sys.executable, "-m", "mqmas.research", str(_research_root(s)), "all")
    code = await proc.wait()
    log.info("research lab finished with exit code %s", code)
    await _sync_library(s, db)


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


@app.get("/api/confluence/{base}")
async def confluence_score(base: str, tf: str = "4h", quote: str = "USDT") -> dict[str, Any]:
    """Confluence score (0–100) of BASE/QUOTE, its components, the price map and the order it implies."""
    from mqmas.ta.confluence import live_score, order_proposal
    from mqmas.ta.levels import price_map
    from mqmas.ta.live import load_study

    rt = _rt()
    if tf not in ("4h", "1d"):
        raise HTTPException(400, "tf must be 4h or 1d")
    ex = rt.feed.exchange
    if ex is None:
        raise HTTPException(503, "no exchange connection (synthetic feed)")
    symbol = f"{base.upper()}/{quote.upper()}"
    rows = await ex.fetch_ohlcv(symbol, tf, limit=400)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df.pop("ts").astype("int64"), unit="ms", utc=True)
    df = df.iloc[:-1].astype(float)
    root = _research_root(rt.settings)
    sc, pm = await asyncio.to_thread(lambda: (live_score(df, load_study(root), tf), price_map(df)))
    path = root / "confluence_report.json"
    validation = json.loads(path.read_text()) if path.exists() else None
    prop = order_proposal(sc, tf, validation, await rt.risk.equity_estimate(), rt.settings.book_risk_per_trade, pm)
    return {"symbol": symbol, "tf": tf, "confluencia": sc, "proposta": prop, "mapa_de_precos": pm,
            "validacao": (validation or {}).get("tf", {}).get(tf)}


@app.get("/api/patterns")
async def pattern_library(tf: str | None = None) -> dict[str, Any]:
    """The pattern library: definition, parameters and measured record of every pattern."""
    return {"patterns": await _rt().db.pattern_library(tf)}


@app.get("/api/ta-signals")
async def ta_signals(limit: int = 50, min_score: float = 0.0) -> dict[str, Any]:
    """Audit trail of the confluence scans (every coin, every 4h/1d close)."""
    return {"signals": await _rt().db.recent_ta_signals(min(limit, 500), min_score)}


@app.get("/api/export/{name}.csv", response_class=PlainTextResponse)
async def export_csv(name: str) -> PlainTextResponse:
    """Spreadsheet view (Excel / Google Sheets: =IMPORTDATA(url)) of trades or scanned signals."""
    import csv
    import io

    rt = _rt()
    if name == "operacoes":
        rows = [json.loads(p.model_dump_json()) for p in await rt.db.closed_positions(rt.settings.trading_mode.value,
                                                                                     limit=5000)]
        rows += [json.loads(p.model_dump_json()) for p in await rt.db.active_positions(rt.settings.trading_mode.value)]
    elif name == "sinais":
        rows = [{**{k: v for k, v in r.items() if k not in ("componentes", "padroes", "mapa", "ordem")},
                 **{f"ordem_{k}": v for k, v in r["ordem"].items()}} for r in await rt.db.recent_ta_signals(5000)]
    else:
        raise HTTPException(404, "use operacoes.csv or sinais.csv")
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return PlainTextResponse(buf.getvalue(), media_type="text/csv; charset=utf-8",
                             headers={"Content-Disposition": f"inline; filename={name}.csv"})


@app.get("/api/ta-study")
async def ta_study_report() -> dict[str, Any]:
    path = _research_root(_rt().settings) / "ta_report.json"
    if not path.exists():
        return {"ready": False}
    rep = json.loads(path.read_text())
    return {"ready": True, "rows": [{k: r[k] for k in ("padrao", "tf", "categoria", "vies", "ocorrencias", "veredito",
                                                         "operacao")} for r in rep["rows"]]}


@app.get("/api/research/livro")
async def research_livro() -> dict[str, Any]:
    """Book strategies one by one, their confluences, and the walk-forward-selected setup."""
    path = _research_root(_rt().settings) / "livro_report.json"
    if not path.exists():
        return {"ready": False}
    rep = json.loads(path.read_text())
    return {"ready": True, "report": {k: v for k, v in rep.items() if k != "linhas"},
            "linhas": [{k: r[k] for k in ("key", "total", "periodo1", "periodo2")} for r in rep["linhas"]]}


@app.get("/api/research/arena")
async def research_arena() -> dict[str, Any]:
    """Every rule on 1h/2h/4h/1d/2d/3d, alone and in confluence, chosen blind; survivors with 40–60 % hit rate."""
    path = _research_root(_rt().settings) / "arena_report.json"
    if not path.exists():
        return {"ready": False}
    return {"ready": True, "report": json.loads(path.read_text())}


@app.get("/api/research/crosscheck")
async def research_crosscheck() -> dict[str, Any]:
    """Our engine vs backtesting.py on the same Binance candles and rules."""
    path = _research_root(_rt().settings) / "crosscheck_report.json"
    if not path.exists():
        return {"ready": False}
    return {"ready": True, "report": json.loads(path.read_text())}


@app.get("/api/research/tcc")
async def research_tcc() -> dict[str, Any]:
    """The four models of Prates (2021) on Binance, in the thesis' report format."""
    path = _research_root(_rt().settings) / "tcc_report.json"
    if not path.exists():
        return {"ready": False}
    return {"ready": True, "report": json.loads(path.read_text())}


@app.get("/api/research/season")
async def research_season() -> dict[str, Any]:
    path = _research_root(_rt().settings) / "season_report.json"
    if not path.exists():
        return {"ready": False}
    return {"ready": True, "report": json.loads(path.read_text())}


@app.post("/api/review")
async def review_report(report: dict[str, Any], x_admin_token: str | None = Header(default=None)) -> dict[str, Any]:
    """Team review of a backtest report: Python facts + Gauss/Ada/Vera debate (spends LLM quota → admin only)."""
    from mqmas.agents.review import TradeReport, debate
    from mqmas.ta.live import load_study

    rt = _rt()
    expected = rt.settings.admin_token.get_secret_value() if rt.settings.admin_token else None
    if not expected or not x_admin_token or not hmac.compare_digest(expected, x_admin_token):
        raise HTTPException(401, "invalid admin token")
    rep = TradeReport.model_validate(report)
    res = await debate(rt.router, rep, load_study(_research_root(rt.settings)))
    for m in res.pop("messages"):
        await rt.db.save_message(m)
        await rt.hub.publish("agent_message", m.model_dump(mode="json"))
    return res


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
