"""End-to-end: unanimous debate → paper fill → stop-loss → circuit breaker."""

from __future__ import annotations

import json

import httpx
import pytest

from mqmas.agents.orchestrator import Orchestrator
from mqmas.execution.broker import PaperBroker
from mqmas.execution.position_manager import PositionManager
from mqmas.llm.router import LLMRouter, ProviderConfig
from mqmas.models import (
    Forecast,
    OrderBookMetrics,
    PositionStatus,
    QuantSnapshot,
    SetupStats,
    VolumeProfile,
)
from mqmas.persistence.db import Database
from mqmas.quant.sizing import build_sizing_plan
from mqmas.risk.constraints import RiskManager
from mqmas.ws.hub import WebSocketHub

PRICE = 100_000.0

ANSWERS = {
    "scenarios": {"scenarios": [
        {"name": "bullish", "probability": 0.6, "rationale": "reversão"},
        {"name": "neutral", "probability": 0.25}, {"name": "bearish", "probability": 0.15}],
        "thesis": "Sobrevenda com absorção no book.", "vote": "APPROVE"},
    "assessment": {"assessment": "Evidência robusta.", "vote": "APPROVE"},
    "pre_mortem": {"pre_mortem": [{"risk": "notícia macro", "likelihood": "low", "mitigation": "stop nativo"}],
                   "verdict": "Aceitável.", "vote": "APPROVE"},
    "plan": {"plan": "Limit no ask.", "vote": "APPROVE"},
}


def llm_handler(req: httpx.Request) -> httpx.Response:
    system = json.loads(req.content)["messages"][0]["content"]
    for key in ("scenarios", "assessment", "pre_mortem", "plan"):
        if f'"{key}"' in system.split("JSON Schema:")[-1].split('"required"')[-1]:
            body = ANSWERS[key]
            break
    else:
        raise AssertionError("unknown schema")
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(body)}}]})


class FakeFeed:
    def __init__(self) -> None:
        self.price = PRICE

    def book(self):
        return ([[self.price - 1 - i, 2.0] for i in range(20)], [[self.price + 1 + i, 2.0] for i in range(20)])

    def is_stale(self) -> bool:
        return False

    exchange = None


def snapshot() -> QuantSnapshot:
    return QuantSnapshot(
        symbol="BTC/USDT", price=PRICE, zscore=-2.4, rolling_mean=101_000, rolling_std=400,
        vwap=100_900, vwap_upper=101_800, vwap_lower=100_100,
        volume_profile=VolumeProfile(poc=100_200, value_area_high=101_500, value_area_low=99_900),
        atr=350, atr_pct=0.0035,
        orderbook=OrderBookMetrics(best_bid=PRICE - 1, best_ask=PRICE + 1, mid=PRICE, spread_pct=0.00002,
                                   bid_volume=12, ask_volume=6, imbalance=0.33, bid_ask_ratio=2.0,
                                   depth_delta=0.05, microprice=PRICE + 0.3),
        forecast=Forecast(degree=1, horizon=4, r_squared=0.4, slope_per_bar=20, expected_price=100_300,
                          lower_95=99_500, upper_95=101_200, expected_return_pct=0.003),
    )


@pytest.mark.asyncio
async def test_unanimous_debate_opens_and_breaker_trips(settings, monkeypatch):
    db = Database(settings.db_path)
    await db.connect()
    hub, feed = WebSocketHub(), FakeFeed()
    risk = RiskManager(settings, db)
    broker = PaperBroker(settings, feed)  # type: ignore[arg-type]
    pm = PositionManager(settings, db, broker, feed, risk, hub)  # type: ignore[arg-type]
    router = LLMRouter([ProviderConfig("groq", "https://groq.test", "k", "m")],
                       client=httpx.AsyncClient(transport=httpx.MockTransport(llm_handler)))
    orch = Orchestrator(settings, db, feed, broker, router, risk, pm, hub)  # type: ignore[arg-type]
    monkeypatch.setattr("mqmas.agents.orchestrator.SPEECH_PACING_SEC", 0)

    stats = SetupStats(backtest_trades=40, backtest_wins=33, live_trades=0, live_wins=0, win_rate=0.80,
                       win_rate_lower_95=0.66, payoff_ratio=0.7, expected_return_pct=0.003, trade_sharpe=0.5,
                       kelly_full=0.5, kelly_applied=0.125)

    async def fake_evidence(snap):
        return stats, build_sizing_plan(50.0, snap.orderbook.best_ask, 0.006, 0.008, stats.kelly_applied, settings)

    monkeypatch.setattr(orch, "_evidence", fake_evidence)

    for _ in range(2):
        feed.price = PRICE
        decision = await orch.debate(snapshot())
        assert decision.approved, decision
        assert all(g.passed for g in decision.gates)
        [pos] = await db.active_positions("paper")
        assert pos.status is PositionStatus.OPEN
        assert pos.quantity * pos.entry_price <= 50.0
        feed.price = pos.stop_price - 5  # crash through the stop
        await pm.supervise_once()
        assert not await db.active_positions("paper")

    closed = await db.closed_positions("paper")
    assert len(closed) == 2 and all(p.pnl_usdt < 0 for p in closed)
    assert all(abs(p.pnl_usdt) <= 50.0 * settings.max_risk_per_trade_pct + 1e-6 for p in closed)
    assert await risk.lock_until() is not None  # 2 consecutive stops → 24h lockout

    feed.price = PRICE
    blocked = await orch.debate(snapshot())
    assert not blocked.approved
    assert any(g.name == "circuit_breaker" and not g.passed for g in blocked.gates)
    await router.aclose()
    await db.close()
