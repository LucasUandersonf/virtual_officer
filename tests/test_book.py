from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from mqmas.book.engine import Signal, StrategyBook
from mqmas.llm.router import LLMRouter, ProviderConfig
from mqmas.models import ExitReason
from mqmas.persistence.db import Database
from mqmas.risk.constraints import RiskManager
from mqmas.ws.hub import WebSocketHub

KEY = "confluence|1d|trigger=breakout,min_extra=2,rs_min=0.6|stop=2.5|tp=4.5|trail=None|hold=20"


class FakeEx:
    markets: dict = {}

    def __init__(self) -> None:
        self.px = {"SOL/USDT": 100.0}

    async def fetch_ticker(self, sym):
        p = self.px[sym]
        return {"bid": p, "ask": p, "last": p}

    async def fetch_tickers(self, syms=None):
        return {s: {"bid": p, "ask": p, "last": p} for s, p in self.px.items()}


def cro_router(vote: str) -> LLMRouter:
    body = {"pre_mortem": [{"risk": "correlação com BTC", "likelihood": "low", "mitigation": "stop"}],
            "verdict": "ok", "vote": vote}

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(body)}}]})

    return LLMRouter([ProviderConfig("groq", "https://g.test", "k", "m")],
                     client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def make(settings, router):
    db = Database(settings.db_path)
    await db.connect()
    ex = FakeEx()
    book = StrategyBook(settings, db, RiskManager(settings, db), router, WebSocketHub(), ex)
    return db, ex, book


def signal() -> Signal:
    return Signal("SOL/USDT", KEY, 100.0, 4.0, 0.9, {"rs_rank": 0.9})


@pytest.mark.asyncio
async def test_entry_sizes_by_risk_and_exits_on_target(settings):
    db, ex, book = await make(settings, cro_router("APPROVE"))
    assert [c.key for c in book.strategies] == [KEY]
    [pos] = await book.try_entries([signal()])
    stop_pct = (pos.entry_price - 90.0) / pos.entry_price
    loss = pos.quantity * pos.entry_price * (stop_pct + 2 * (settings.fee_rate + settings.max_slippage_pct))
    assert loss == pytest.approx(50 * settings.book_risk_per_trade, rel=1e-6)
    assert pos.stop_price == pytest.approx(90.0) and pos.take_profit_price == pytest.approx(118.0)
    assert await book.try_entries([signal()]) == []  # one position per coin
    ex.px["SOL/USDT"] = 119.0
    await book.supervise()
    [closed] = await db.closed_positions("paper")
    assert closed.exit_reason is ExitReason.TAKE_PROFIT and closed.pnl_usdt > 0
    await db.close()


@pytest.mark.asyncio
async def test_gap_through_stop_fills_below_stop(settings):
    db, ex, book = await make(settings, cro_router("APPROVE"))
    await book.try_entries([signal()])
    ex.px["SOL/USDT"] = 80.0  # gaps through the 90 stop
    await book.supervise()
    [closed] = await db.closed_positions("paper")
    assert closed.exit_reason is ExitReason.STOP_LOSS and closed.exit_price < 90.0
    await db.close()


@pytest.mark.asyncio
async def test_time_stop_after_hold(settings):
    db, _, book = await make(settings, cro_router("APPROVE"))
    [pos] = await book.try_entries([signal()])
    old = (datetime.now(timezone.utc) - timedelta(days=21)).isoformat()
    await db._write("UPDATE positions SET opened_at=? WHERE id=?", (old, pos.id))
    await book.supervise()
    [closed] = await db.closed_positions("paper")
    assert closed.exit_reason is ExitReason.TIME_STOP
    await db.close()


@pytest.mark.asyncio
async def test_cro_veto_and_llm_outage_block_entries(settings):
    db, _, book = await make(settings, cro_router("REJECT"))
    assert await book.try_entries([signal()]) == []
    await db.close()
    db, _, book = await make(settings, LLMRouter([]))  # no LLM at all → fail-closed
    assert await book.try_entries([signal()]) == []
    await db.close()
