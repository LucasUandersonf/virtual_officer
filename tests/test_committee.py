from __future__ import annotations

import httpx
import numpy as np
import pandas as pd
import pytest

from mqmas.llm.router import LLMRouter
from mqmas.persistence.db import Database
from mqmas.trend.committee import MIN_FLAGS, Committee, combine, render_board
from mqmas.trend.live import TrendAllocator
from mqmas.trend.specialists import (
    ABSTAIN,
    CONFIRM,
    REDUCE,
    VETO,
    Fundamental,
    Opinion,
    Proposal,
    Risk,
    Sentiment,
    Technical,
    validated_signals,
)
from tests.test_trend import NOW, FakeEx, frame, make_alloc, market


class Fixed:
    def __init__(self, name: str, verdict: str) -> None:
        self.name, self.verdict = name, verdict

    async def opine(self, p, others):
        return Opinion(self.name, self.verdict, f"{self.name} diz {self.verdict}", {"vistos": len(others)})


def proposal(sym: str = "SOL/USDT", w: float = 0.2) -> Proposal:
    return Proposal(sym, w, 0.0, 0.8, 100.0, frame(400, 0.003, 3))


def test_only_active_specialists_change_the_order():
    ops = [Opinion("tecnico", REDUCE, ""), Opinion("risco", VETO, ""), Opinion("sentimento", CONFIRM, "")]
    assert combine(ops, set()) == 1.0  # all in shadow: the backtest's order goes through
    assert combine(ops, {"tecnico"}) == 0.5
    assert combine(ops, {"tecnico", "risco"}) == 0.0
    assert combine([Opinion("risco", ABSTAIN, "")], {"risco"}) == 1.0  # abstaining never blocks


@pytest.mark.asyncio
async def test_review_records_every_opinion_and_each_specialist_sees_the_previous_ones(settings):
    db = Database(settings.db_path)
    await db.connect()
    c = Committee([Fixed("tecnico", CONFIRM), Fixed("fundamentalista", REDUCE), Fixed("risco", VETO)],
                  {"fundamentalista"}, db)
    await c.ensure_schema()
    [r] = (await c.review("2026-10-26", [proposal()])).values()
    assert r.final == pytest.approx(0.1)  # active reduce halves the increase; the veto is in shadow
    assert [o.data["vistos"] for o in r.opinions] == [0, 1, 2]
    async with db.conn.execute("SELECT specialist, active FROM committee_opinions ORDER BY id") as cur:
        assert [tuple(x) for x in await cur.fetchall()] == [("tecnico", 0), ("fundamentalista", 1), ("risco", 0)]
    await db.close()


@pytest.mark.asyncio
async def test_scoreboard_measures_whether_flags_avoided_losses(settings):
    db = Database(settings.db_path)
    await db.connect()
    c = Committee([Fixed("tecnico", REDUCE)], set(), db)
    await c.ensure_schema()
    rng = np.random.default_rng(0)
    days = pd.date_range("2026-01-01", periods=120, freq="1D", tz="UTC")
    closes: dict[str, pd.Series] = {}
    for i in range(2 * MIN_FLAGS):  # flagged coins fall 10 % over 30 days, confirmed ones rise 10 %
        sym, flagged = f"C{i}/USDT", i % 2 == 0
        path = np.linspace(100, 90 if flagged else 110, 31) * (1 + rng.normal(0, 0.002, 31))
        closes[sym] = pd.Series(np.r_[path, np.full(len(days) - 31, path[-1])], index=days)
        c.specialists = [Fixed("tecnico", REDUCE if flagged else CONFIRM)]
        await c.review("2026-01-01", [Proposal(sym, 0.1, 0.0, 0.5, 100.0, pd.DataFrame())])
    assert await c.fill_outcomes(closes, "2026-03-01") == 2 * MIN_FLAGS
    [b] = await c.scoreboard()
    assert b["veredito"] == "ajuda" and b["vantagem"] == pytest.approx(0.2, abs=0.01)
    assert b["peso_x_retorno_evitado"] > 0
    assert "tecnico [sombra]" in render_board([b])
    await db.close()


def mock_http(routes: dict[str, object]) -> httpx.AsyncClient:
    def handler(req: httpx.Request) -> httpx.Response:
        for key, body in routes.items():
            if key in str(req.url):
                return body if isinstance(body, httpx.Response) else httpx.Response(200, json=body)
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_fundamental_flags_dilution_and_falling_tvl():
    coin = {"id": "solana", "market_cap_rank": 6, "categories": ["Layer 1"],
            "market_data": {"market_cap": {"usd": 50e9}, "fully_diluted_valuation": {"usd": 120e9},
                            "circulating_supply": 400e6, "total_supply": 600e6, "max_supply": None}}
    http = mock_http({"/search": {"coins": [{"id": "solana", "symbol": "sol", "market_cap_rank": 6}]},
                      "/coins/solana": coin, "llama.fi/protocols": [{"gecko_id": "solana", "tvl": 5e9, "change_1m": -40}]})
    op = await Fundamental(http).opine(proposal(), [])
    assert op.verdict == REDUCE and "FDV 2.4×" in op.reason and "TVL" in op.reason
    assert op.data["circulante_pct"] == pytest.approx(66.7)
    down = await Fundamental(mock_http({"/search": httpx.Response(429)})).opine(proposal(), [])
    assert down.verdict == ABSTAIN  # API down or rate-limited: abstain, never block


@pytest.mark.asyncio
async def test_sentiment_flags_euphoria_and_abstains_without_data():
    http = mock_http({"alternative.me": {"data": [{"value": "86"}]}, "premiumIndex": {"lastFundingRate": "0.0001"}})
    op = await Sentiment(http).opine(proposal(), [])
    assert op.verdict == REDUCE and "ganância extrema" in op.reason
    calm = mock_http({"alternative.me": {"data": [{"value": "40"}]}, "premiumIndex": {"lastFundingRate": "0.0001"}})
    assert (await Sentiment(calm).opine(proposal(), [])).verdict == CONFIRM
    assert (await Sentiment(mock_http({})).opine(proposal(), [])).verdict == ABSTAIN


@pytest.mark.asyncio
async def test_risk_abstains_without_an_llm():
    op = await Risk(LLMRouter([])).opine(proposal(), [Opinion("tecnico", CONFIRM, "ok")])
    assert op.verdict == ABSTAIN


@pytest.mark.asyncio
async def test_technical_reads_the_validated_signals():
    d = frame(400, 0.002, 5)
    d.iloc[-1, d.columns.get_loc("close")] = d["high"].iloc[-21:-1].max() * 1.05  # 20-day breakout
    d.iloc[-1, d.columns.get_loc("high")] = d["close"].iloc[-1] * 1.01
    d.iloc[-1, d.columns.get_loc("volume")] = d["volume"].iloc[-1] * 5  # with heavy volume
    sigs = validated_signals(d)
    assert "classico rompimento" in sigs and "swing 1d rompimento" in sigs
    op = await Technical().opine(Proposal("SOL/USDT", 0.2, 0.0, 0.9, float(d["close"].iloc[-1]), d), [])
    assert op.verdict in (CONFIRM, REDUCE) and "score_confluencia" in op.data


@pytest.mark.asyncio
async def test_allocator_trades_like_the_backtest_in_shadow_and_obeys_an_active_veto(trend_settings):
    async def run(active: set[str]):
        ex = FakeEx(market())
        await ex.load_markets()
        cfg = trend_settings.model_copy(update={"db_path": trend_settings.db_path.with_name(f"c{len(active)}.db")})
        db, _ = await make_alloc(cfg, ex)
        c = Committee([Fixed("risco", VETO)], active, db)
        await c.ensure_schema()
        rep = await TrendAllocator(cfg, db, ex, committee=c).run_once(NOW)
        await db.close()
        return rep

    shadow, active = await run(set()), await run({"risco"})
    assert shadow["holdings"] and not active["holdings"]  # same proposals; only the active veto blocks
    assert any(n.startswith("COMITÊ") for n in shadow["notes"])


@pytest.fixture
def trend_settings(settings):
    return settings.model_copy(update={"trend_capital_usdt": 200.0, "trend_top_n": 4})
