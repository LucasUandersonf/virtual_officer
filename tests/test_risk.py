from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from mqmas.config import Settings
from mqmas.models import ExitReason, Position, PositionStatus
from mqmas.persistence.db import Database
from mqmas.risk.constraints import RiskManager


def test_mandate_cannot_be_loosened(tmp_path):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_risk_per_trade_pct=0.02)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, stop_loss_max_pct=0.01)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, trading_mode="live")  # no confirmation phrase / keys


async def _closed(db: Database, pnl: float, reason: ExitReason) -> None:
    p = Position(debate_id="d", mode="paper", symbol="BTC/USDT", status=PositionStatus.CLOSED,
                 quantity=0.0003, entry_price=100_000, stop_price=99_400, take_profit_price=100_600,
                 closed_at=datetime.now(timezone.utc), exit_reason=reason, pnl_usdt=pnl)
    pid = await db.insert_position(p)
    p.id = pid
    await db.update_position(p)


@pytest.mark.asyncio
async def test_circuit_breaker_after_two_stops(settings):
    db = Database(settings.db_path)
    await db.connect()
    risk = RiskManager(settings, db)
    await _closed(db, -0.25, ExitReason.STOP_LOSS)
    assert await risk.on_trade_closed() is None
    await _closed(db, -0.25, ExitReason.STOP_LOSS)
    until = await risk.on_trade_closed()
    assert until is not None
    assert await risk.lock_until() is not None
    await db.close()


@pytest.mark.asyncio
async def test_win_resets_streak(settings):
    db = Database(settings.db_path)
    await db.connect()
    risk = RiskManager(settings, db)
    await _closed(db, -0.25, ExitReason.STOP_LOSS)
    await _closed(db, 0.2, ExitReason.TAKE_PROFIT)
    await _closed(db, -0.25, ExitReason.STOP_LOSS)
    assert await risk.consecutive_stops_today() == 1
    assert await risk.on_trade_closed() is None
    await db.close()


@pytest.mark.asyncio
async def test_wal_mode(settings):
    db = Database(settings.db_path)
    await db.connect()
    async with db.conn.execute("PRAGMA journal_mode") as cur:
        assert (await cur.fetchone())[0] == "wal"
    await db.close()
