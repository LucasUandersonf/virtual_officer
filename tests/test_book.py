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


@pytest.mark.asyncio
async def test_confluence_scan_logs_every_coin_and_enters_on_a_validated_crossing(settings):
    import numpy as np

    from mqmas.ta.confluence import live_score

    db, ex, book = await make(settings, cro_router("APPROVE"))
    n, step = 400, 4 * 3_600_000
    start = (int(datetime.now(timezone.utc).timestamp() * 1000) // step - n - 1) * step
    for seed in range(50):  # a market whose last bar raised the score
        rng = np.random.default_rng(seed)
        c = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n)))
        o = np.r_[c[0], c[:-1]]
        rows = [[start + i * step, o[i], max(o[i], c[i]) * 1.004, min(o[i], c[i]) * 0.996, c[i], 100.0 + i]
                for i in range(n)]
        import pandas as pd
        df = pd.DataFrame([r[1:] for r in rows], columns=["open", "high", "low", "close", "volume"],
                          index=pd.to_datetime([r[0] for r in rows], unit="ms", utc=True))
        sc = live_score(df, None, "4h")
        if sc["score"] > sc["score_anterior"] + 1:
            break
    th = sc["score"] - 0.5
    (settings.db_path.parent / "research").mkdir(parents=True, exist_ok=True)
    (settings.db_path.parent / "research" / "confluence_report.json").write_text(json.dumps({"tf": {
        "4h": {"limiar": th, "linhas": [{"limiar": th, "periodo2": {"trades": 80, "acerto": 0.5, "pf": 1.4}}]},
        "1d": {"limiar": None, "linhas": []}}}))

    async def fetch_ohlcv(sym, tf, limit=400):
        return rows

    ex.fetch_ohlcv = fetch_ohlcv
    ex.px["SOL/USDT"] = float(c[-1])
    book.state.universe = ["SOL/USDT"]
    signals = await book.scan_confluence()
    assert [s.symbol for s in signals] == ["SOL/USDT"]
    logged = await db.recent_ta_signals()
    assert {r["timeframe"] for r in logged} == {"4h", "1d"} and logged[0]["ordem"]["risco_retorno"] == "1:2"
    [pos] = await book.try_entries(signals)
    atr = signals[0].atr
    assert pos.stop_price == pytest.approx(signals[0].price - 1.5 * atr)
    assert pos.take_profit_price == pytest.approx(signals[0].price + 3.0 * atr)
    assert pos.setup.startswith("book:score-4h")
    assert await book.scan_confluence() == []  # same closed bar: not scanned twice
    await db.close()
