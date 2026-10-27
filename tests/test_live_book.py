"""Live book execution against a fake Binance: stop on the exchange, idempotent ids, reconciliation."""

from __future__ import annotations

import math

import ccxt
import pytest

from mqmas.book.engine import Signal, StrategyBook
from mqmas.config import LIVE_CONFIRMATION_PHRASE, Settings
from mqmas.models import ExitReason, PositionStatus
from mqmas.persistence.db import Database
from mqmas.risk.constraints import RiskManager
from mqmas.ws.hub import WebSocketHub
from tests.test_book import KEY, cro_router


class FakeBinance:
    def __init__(self, price: float = 100.0) -> None:
        self.px = {"SOL/USDT": price}
        self.orders: dict[str, dict] = {}
        self.sent: list[dict] = []
        self.fail_next_create = False
        self.markets = {"SOL/USDT": {"base": "SOL", "quote": "USDT", "limits": {"cost": {"min": 5.0}}}}

    def market(self, sym):
        return self.markets[sym]

    def amount_to_precision(self, sym, x):
        return str(math.floor(x * 1000) / 1000)

    def price_to_precision(self, sym, x):
        return str(round(x, 2))

    async def fetch_balance(self):
        return {"USDT": {"free": 1000.0}}

    async def fetch_ticker(self, sym):
        p = self.px[sym]
        return {"bid": p, "ask": p, "last": p}

    async def fetch_tickers(self, syms=None):
        return {s: {"bid": p, "ask": p, "last": p} for s, p in self.px.items()}

    async def create_order(self, sym, type_, side, amount, price=None, params=None):
        params = params or {}
        oid = str(len(self.orders) + 1)
        o = {"id": oid, "clientOrderId": params.get("newClientOrderId"), "symbol": sym, "side": side,
             "amount": amount, "price": price, "filled": 0.0, "status": "open", "fees": [],
             "stop": params.get("stopLossPrice")}
        if o["stop"] is None:  # plain orders fill at once at the current price
            px = price if type_ == "limit" else self.px[sym]
            o.update(status="closed", filled=amount, average=px)
            fee_cur = "SOL" if side == "buy" else "USDT"
            o["fees"] = [{"cost": amount * 0.001 if side == "buy" else amount * px * 0.001, "currency": fee_cur}]
        self.orders[oid] = o
        self.sent.append(o)
        if self.fail_next_create:  # the order reached the exchange but the response was lost
            self.fail_next_create = False
            raise ccxt.RequestTimeout("lost response")
        return dict(o)

    async def fetch_order(self, oid, sym, params=None):
        cid = (params or {}).get("origClientOrderId")
        for o in self.orders.values():
            if (cid and o["clientOrderId"] == cid) or (not cid and o["id"] == oid):
                return dict(o)
        raise ccxt.OrderNotFound(oid or cid)

    async def cancel_order(self, oid, sym):
        o = self.orders.get(oid)
        if not o:
            raise ccxt.OrderNotFound(oid)
        if o["status"] == "open":
            o["status"] = "canceled"

    def trigger_stop(self, oid):
        o = self.orders[oid]
        o.update(status="closed", filled=o["amount"], average=o["price"],
                 fees=[{"cost": o["amount"] * o["price"] * 0.001, "currency": "USDT"}])


def live_settings(tmp_path) -> Settings:
    return Settings(_env_file=None, db_path=tmp_path / "t.db", trading_mode="live",
                    live_trading_confirm=LIVE_CONFIRMATION_PHRASE, exchange_api_key="k", exchange_api_secret="s",
                    book_live_enabled=True, initial_capital_usdt=1000.0)


async def make(tmp_path, ex):
    s = live_settings(tmp_path)
    db = Database(s.db_path)
    await db.connect()
    return db, StrategyBook(s, db, RiskManager(s, db), cro_router("APPROVE"), WebSocketHub(), ex)


def signal() -> Signal:
    return Signal("SOL/USDT", KEY, 100.0, 4.0, 0.9, {"rs_rank": 0.9})


@pytest.mark.asyncio
async def test_live_entry_puts_the_stop_on_the_exchange_and_books_the_stop_fill(tmp_path):
    ex = FakeBinance()
    db, book = await make(tmp_path, ex)
    [pos] = await book.try_entries([signal()])
    buy, stop = ex.sent
    assert buy["side"] == "buy" and buy["clientOrderId"].startswith("mqe")
    assert stop["stop"] == pytest.approx(90.0, abs=0.06) and stop["clientOrderId"].startswith("mqs")
    assert pos.quantity == pytest.approx(math.floor(buy["amount"] * 0.999 * 1000) / 1000)  # base fee deducted
    assert pos.stop_order_id == stop["id"] and pos.status is PositionStatus.OPEN
    ex.trigger_stop(stop["id"])
    await book.supervise()
    [closed] = await db.closed_positions("live")
    assert closed.exit_reason is ExitReason.STOP_LOSS and closed.exit_price == pytest.approx(stop["price"])
    await db.close()


@pytest.mark.asyncio
async def test_take_profit_cancels_the_stop_before_selling(tmp_path):
    ex = FakeBinance()
    db, book = await make(tmp_path, ex)
    [pos] = await book.try_entries([signal()])
    ex.px["SOL/USDT"] = 119.0
    await book.supervise()
    assert ex.orders[pos.stop_order_id]["status"] == "canceled"
    [closed] = await db.closed_positions("live")
    assert closed.exit_reason is ExitReason.TAKE_PROFIT and closed.pnl_usdt > 0
    await db.close()


@pytest.mark.asyncio
async def test_lost_response_is_recovered_by_client_id_not_resent(tmp_path):
    ex = FakeBinance()
    db, book = await make(tmp_path, ex)
    ex.fail_next_create = True
    await book.try_entries([signal()])
    buys = [o for o in ex.sent if o["side"] == "buy"]
    assert len(buys) == 1  # never sent twice
    [pos] = await book.positions()
    assert pos.status is PositionStatus.OPEN and pos.stop_order_id
    await db.close()


@pytest.mark.asyncio
async def test_boot_reconciles_a_pending_entry_and_replaces_a_vanished_stop(tmp_path):
    ex = FakeBinance()
    db, book = await make(tmp_path, ex)
    [pos] = await book.try_entries([signal()])
    # simulate a crash right after the buy: row back to PENDING, stop gone from the exchange
    ex.orders[pos.stop_order_id]["status"] = "canceled"
    pos.status, pos.stop_order_id = PositionStatus.PENDING, None
    await db.update_position(pos)
    s = live_settings(tmp_path)
    book2 = StrategyBook(s, db, RiskManager(s, db), cro_router("APPROVE"), WebSocketHub(), ex)
    await book2.reconcile()
    [again] = await book2.positions()
    assert again.status is PositionStatus.OPEN and again.stop_order_id
    assert ex.orders[again.stop_order_id]["status"] == "open"
    assert sum(o["side"] == "buy" for o in ex.sent) == 1
    await db.close()


@pytest.mark.asyncio
async def test_below_binance_minimum_is_refused_without_sending(tmp_path):
    ex = FakeBinance()
    s = live_settings(tmp_path).model_copy(update={"initial_capital_usdt": 50.0})
    db = Database(s.db_path)
    await db.connect()
    book = StrategyBook(s, db, RiskManager(s, db), cro_router("APPROVE"), WebSocketHub(), ex)
    assert await book.try_entries([signal()]) == []  # 0.6 % of US$50 with a 10 % stop ≈ US$3 < US$5
    assert ex.sent == []
    await db.close()
