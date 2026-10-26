"""Position lifecycle: open after consensus, supervise exits, reconcile on boot."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from mqmas.config import Settings
from mqmas.data.market_feed import MarketFeed
from mqmas.execution.broker import Broker
from mqmas.models import ExitReason, Position, PositionStatus, SizingPlan
from mqmas.persistence.db import Database
from mqmas.risk.constraints import RiskManager
from mqmas.ws.hub import WebSocketHub

log = logging.getLogger(__name__)

SUPERVISION_INTERVAL_SEC = 2.0


class PositionManager:
    def __init__(self, settings: Settings, db: Database, broker: Broker, feed: MarketFeed,
                 risk: RiskManager, hub: WebSocketHub) -> None:
        self._s = settings
        self._db = db
        self._broker = broker
        self._feed = feed
        self._risk = risk
        self._hub = hub
        self._lock = asyncio.Lock()  # one entry/exit at a time

    @property
    def mode(self) -> str:
        return self._s.trading_mode.value

    async def _publish(self, p: Position, event: str) -> None:
        await self._hub.publish("position", {"event": event, "position": p.model_dump(mode="json")})

    async def open_position(self, debate_id: str, plan: SizingPlan, limit_price: float) -> Position | None:
        async with self._lock:
            if await self._db.active_positions(self.mode):
                log.warning("refusing entry: a position is already active")
                return None
            qty = self._broker.normalize_quantity(plan.quantity)
            if qty <= 0:
                return None
            p = Position(debate_id=debate_id, mode=self.mode, symbol=self._s.symbol,
                         status=PositionStatus.PENDING, quantity=qty, entry_price=limit_price,
                         stop_price=plan.stop_price, take_profit_price=plan.take_profit_price)
            p.id = await self._db.insert_position(p)  # persisted *before* the order is sent
            try:
                fill = await self._broker.buy_limit(qty, limit_price, self._s.entry_order_timeout_sec)
            except Exception:
                log.exception("entry order failed")
                await self._cancel_entry(p, "entry_failed")
                return None
            p.entry_order_id = fill.order_id
            if fill.quantity <= 0:
                await self._cancel_entry(p, "entry_unfilled")
                return None
            # Re-anchor stop/target on the *actual* fill price.
            p.quantity = fill.quantity
            p.entry_price = fill.avg_price
            p.stop_price = fill.avg_price * (1 - plan.stop_pct)
            p.take_profit_price = fill.avg_price * (1 + plan.take_profit_pct)
            p.fees_usdt = fill.fee_usdt
            p.status = PositionStatus.OPEN
            p.stop_order_id = await self._broker.place_protective_stop(p.quantity, p.stop_price)
            await self._db.update_position(p)
            await self._publish(p, "opened")
            log.info("position opened: %s", p.model_dump_json())
            return p

    async def _cancel_entry(self, p: Position, event: str) -> None:
        p.status, p.exit_reason = PositionStatus.CANCELLED, ExitReason.ENTRY_TIMEOUT
        p.closed_at = datetime.now(timezone.utc)
        await self._db.update_position(p)
        await self._publish(p, event)

    async def close_position(self, p: Position, reason: ExitReason, urgent: bool) -> None:
        async with self._lock:
            fill = await self._broker.sell_now(p.quantity, p.stop_order_id, urgent)
            await self._finalise(p, reason, fill.avg_price, fill.fee_usdt)

    async def _finalise(self, p: Position, reason: ExitReason, exit_price: float, exit_fee: float) -> None:
        p.exit_price = exit_price
        p.fees_usdt += exit_fee
        p.pnl_usdt = (exit_price - p.entry_price) * p.quantity - p.fees_usdt
        p.exit_reason = reason
        p.status = PositionStatus.CLOSED
        p.closed_at = datetime.now(timezone.utc)
        await self._db.update_position(p)
        await self._publish(p, "closed")
        log.info("position closed: %s", p.model_dump_json())
        lock = await self._risk.on_trade_closed()
        if lock:
            await self._hub.publish("risk", {"event": "circuit_breaker", "lock_until": lock.isoformat()})

    async def supervise_once(self) -> None:
        bids, _ = self._feed.book()
        if not bids:
            return
        bid = float(bids[0][0])
        kill = await self._risk.kill_switch_engaged()
        for p in await self._db.active_positions(self.mode):
            if p.status is not PositionStatus.OPEN:
                continue
            if p.stop_order_id:
                stop_fill = await self._broker.stop_was_triggered(p.stop_order_id)
                if stop_fill:
                    async with self._lock:
                        await self._finalise(p, ExitReason.STOP_LOSS, stop_fill.avg_price, stop_fill.fee_usdt)
                    continue
            if kill:
                await self.close_position(p, ExitReason.MANUAL, urgent=True)
            elif bid <= p.stop_price:
                await self.close_position(p, ExitReason.STOP_LOSS, urgent=True)
            elif bid >= p.take_profit_price:
                await self.close_position(p, ExitReason.TAKE_PROFIT, urgent=False)

    async def reconcile_on_boot(self) -> None:
        """A PENDING row at boot means we crashed mid-entry: never assume a fill."""
        for p in await self._db.active_positions(self.mode):
            if p.status is PositionStatus.PENDING:
                log.warning("pending position %s found at boot — marking cancelled; verify exchange manually", p.id)
                await self._cancel_entry(p, "orphan_pending_entry")
                await self._hub.publish("risk", {"event": "orphan_pending_entry", "position_id": p.id})

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                if not self._feed.is_stale():
                    await self.supervise_once()
            except Exception:
                log.exception("position supervision error")
            await asyncio.sleep(SUPERVISION_INTERVAL_SEC)
