"""SQLite persistence (WAL mode) — the firm's long-term memory.

A single connection is shared by the process and guarded by an asyncio lock:
SQLite serialises writers anyway, and one connection avoids "database is
locked" storms. WAL lets the dashboard read while the agents write.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from mqmas.models import AgentMessage, Decision, ExitReason, Position, PositionStatus

SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    debate_id         TEXT NOT NULL,
    mode              TEXT NOT NULL,
    symbol            TEXT NOT NULL,
    status            TEXT NOT NULL,
    quantity          REAL NOT NULL,
    entry_price       REAL NOT NULL,
    stop_price        REAL NOT NULL,
    take_profit_price REAL NOT NULL,
    opened_at         TEXT NOT NULL,
    closed_at         TEXT,
    exit_price        REAL,
    exit_reason       TEXT,
    fees_usdt         REAL NOT NULL DEFAULT 0,
    pnl_usdt          REAL,
    entry_order_id    TEXT,
    stop_order_id     TEXT
);
CREATE INDEX IF NOT EXISTS ix_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS ix_positions_closed ON positions(closed_at);

CREATE TABLE IF NOT EXISTS debates (
    id         TEXT PRIMARY KEY,
    ts         TEXT NOT NULL,
    approved   INTEGER NOT NULL,
    reason     TEXT NOT NULL,
    decision   TEXT NOT NULL,
    snapshot   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    debate_id  TEXT,
    ts         TEXT NOT NULL,
    role       TEXT NOT NULL,
    agent      TEXT NOT NULL,
    text       TEXT NOT NULL,
    vote       TEXT,
    provider   TEXT
);
CREATE INDEX IF NOT EXISTS ix_messages_debate ON messages(debate_id);

CREATE TABLE IF NOT EXISTS risk_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    kind       TEXT NOT NULL,
    detail     TEXT NOT NULL,
    lock_until TEXT
);

CREATE TABLE IF NOT EXISTS candles (
    symbol    TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    ts        INTEGER NOT NULL,
    open      REAL NOT NULL,
    high      REAL NOT NULL,
    low       REAL NOT NULL,
    close     REAL NOT NULL,
    volume    REAL NOT NULL,
    PRIMARY KEY (symbol, timeframe, ts)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).isoformat() if dt else None


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class Database:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA synchronous=NORMAL;")
        await self._conn.execute("PRAGMA busy_timeout=5000;")
        await self._conn.execute("PRAGMA foreign_keys=ON;")
        await self._conn.executescript(SCHEMA)
        await self._migrate()
        await self._conn.commit()

    async def _migrate(self) -> None:
        async with self.conn.execute("PRAGMA table_info(positions)") as cur:
            cols = {r["name"] for r in await cur.fetchall()}
        if "setup" not in cols:
            await self.conn.execute("ALTER TABLE positions ADD COLUMN setup TEXT")

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("database not connected")
        return self._conn

    async def _write(self, sql: str, params: tuple[Any, ...]) -> int:
        async with self._lock:
            cur = await self.conn.execute(sql, params)
            await self.conn.commit()
            return int(cur.lastrowid or 0)

    # ------------------------------------------------------------------ kv state

    async def get_kv(self, key: str) -> str | None:
        async with self.conn.execute("SELECT value FROM kv WHERE key=?", (key,)) as cur:
            row = await cur.fetchone()
        return str(row["value"]) if row else None

    async def set_kv(self, key: str, value: str) -> None:
        await self._write(
            "INSERT INTO kv(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    # ------------------------------------------------------------------ positions

    async def insert_position(self, p: Position) -> int:
        return await self._write(
            "INSERT INTO positions(debate_id, mode, symbol, status, quantity, entry_price, "
            "stop_price, take_profit_price, opened_at, fees_usdt, entry_order_id, stop_order_id, setup) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                p.debate_id, p.mode, p.symbol, p.status.value, p.quantity, p.entry_price,
                p.stop_price, p.take_profit_price, _iso(p.opened_at), p.fees_usdt,
                p.entry_order_id, p.stop_order_id, p.setup,
            ),
        )

    async def update_position(self, p: Position) -> None:
        if p.id is None:
            raise ValueError("position has no id")
        await self._write(
            "UPDATE positions SET status=?, quantity=?, entry_price=?, stop_price=?, "
            "take_profit_price=?, closed_at=?, exit_price=?, exit_reason=?, fees_usdt=?, "
            "pnl_usdt=?, entry_order_id=?, stop_order_id=? WHERE id=?",
            (
                p.status.value, p.quantity, p.entry_price, p.stop_price, p.take_profit_price,
                _iso(p.closed_at), p.exit_price, p.exit_reason.value if p.exit_reason else None,
                p.fees_usdt, p.pnl_usdt, p.entry_order_id, p.stop_order_id, p.id,
            ),
        )

    @staticmethod
    def _row_to_position(r: aiosqlite.Row) -> Position:
        return Position(
            id=r["id"], debate_id=r["debate_id"], mode=r["mode"], symbol=r["symbol"],
            status=PositionStatus(r["status"]), quantity=r["quantity"],
            entry_price=r["entry_price"], stop_price=r["stop_price"],
            take_profit_price=r["take_profit_price"], opened_at=_dt(r["opened_at"]),
            closed_at=_dt(r["closed_at"]), exit_price=r["exit_price"],
            exit_reason=ExitReason(r["exit_reason"]) if r["exit_reason"] else None,
            fees_usdt=r["fees_usdt"], pnl_usdt=r["pnl_usdt"],
            entry_order_id=r["entry_order_id"], stop_order_id=r["stop_order_id"],
            setup=r["setup"],
        )

    async def active_positions(self, mode: str, symbol: str | None = None) -> list[Position]:
        sql = "SELECT * FROM positions WHERE mode=? AND status IN ('pending','open')"
        params: tuple[Any, ...] = (mode,)
        if symbol:
            sql += " AND symbol=?"
            params += (symbol,)
        async with self.conn.execute(sql + " ORDER BY id", params) as cur:
            return [self._row_to_position(r) for r in await cur.fetchall()]

    async def closed_positions(self, mode: str, limit: int = 500) -> list[Position]:
        async with self.conn.execute(
            "SELECT * FROM positions WHERE mode=? AND status='closed' "
            "ORDER BY closed_at DESC LIMIT ?",
            (mode, limit),
        ) as cur:
            return [self._row_to_position(r) for r in await cur.fetchall()]

    async def realised_pnl(self, mode: str) -> float:
        async with self.conn.execute(
            "SELECT COALESCE(SUM(pnl_usdt), 0) AS s FROM positions WHERE mode=? AND status='closed'",
            (mode,),
        ) as cur:
            row = await cur.fetchone()
        return float(row["s"]) if row else 0.0

    async def pnl_since(self, mode: str, since: datetime) -> float:
        async with self.conn.execute(
            "SELECT COALESCE(SUM(pnl_usdt), 0) AS s FROM positions "
            "WHERE mode=? AND status='closed' AND closed_at >= ?",
            (mode, _iso(since)),
        ) as cur:
            row = await cur.fetchone()
        return float(row["s"]) if row else 0.0

    async def win_loss_counts(self, mode: str | None = None, setup: str | None = None) -> tuple[int, int]:
        sql = "SELECT COUNT(*) AS n, SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END) AS w FROM positions WHERE status='closed'"
        params: tuple[Any, ...] = ()
        if mode:
            sql += " AND mode=?"
            params += (mode,)
        if setup:
            sql += " AND setup=?"
            params += (setup,)
        async with self.conn.execute(sql, params) as cur:
            row = await cur.fetchone()
        return (int(row["n"] or 0), int(row["w"] or 0)) if row else (0, 0)

    # ------------------------------------------------------------------ debates / messages

    async def save_message(self, m: AgentMessage) -> None:
        await self._write(
            "INSERT INTO messages(debate_id, ts, role, agent, text, vote, provider) VALUES(?,?,?,?,?,?,?)",
            (m.debate_id, _iso(m.ts), m.role.value, m.agent_name, m.text,
             m.vote.value if m.vote else None, m.provider),
        )

    async def save_debate(self, d: Decision, snapshot_json: str) -> None:
        await self._write(
            "INSERT OR REPLACE INTO debates(id, ts, approved, reason, decision, snapshot) VALUES(?,?,?,?,?,?)",
            (d.debate_id, _iso(datetime.now(timezone.utc)), int(d.approved), d.reason,
             d.model_dump_json(), snapshot_json),
        )

    async def recent_debates(self, limit: int = 20) -> list[dict[str, Any]]:
        async with self.conn.execute(
            "SELECT id, ts, approved, reason FROM debates ORDER BY ts DESC LIMIT ?", (limit,)
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    # ------------------------------------------------------------------ risk events

    async def log_risk_event(self, kind: str, detail: dict[str, Any], lock_until: datetime | None = None) -> None:
        await self._write(
            "INSERT INTO risk_events(ts, kind, detail, lock_until) VALUES(?,?,?,?)",
            (_iso(datetime.now(timezone.utc)), kind, json.dumps(detail), _iso(lock_until)),
        )

    async def active_lock_until(self) -> datetime | None:
        now = datetime.now(timezone.utc)
        async with self.conn.execute(
            "SELECT MAX(lock_until) AS u FROM risk_events WHERE lock_until IS NOT NULL"
        ) as cur:
            row = await cur.fetchone()
        until = _dt(row["u"]) if row and row["u"] else None
        return until if until and until > now else None

    async def closed_since(self, mode: str, since: datetime) -> list[Position]:
        async with self.conn.execute(
            "SELECT * FROM positions WHERE mode=? AND status='closed' AND closed_at >= ? ORDER BY closed_at",
            (mode, _iso(since)),
        ) as cur:
            return [self._row_to_position(r) for r in await cur.fetchall()]


    # ------------------------------------------------------------------ candle history (Setup League)

    async def last_candle_ts(self, symbol: str, timeframe: str) -> int | None:
        async with self.conn.execute(
            "SELECT MAX(ts) AS t FROM candles WHERE symbol=? AND timeframe=?", (symbol, timeframe)
        ) as cur:
            row = await cur.fetchone()
        return int(row["t"]) if row and row["t"] is not None else None

    async def save_candles(self, symbol: str, timeframe: str, rows: list[list[float]]) -> None:
        async with self._lock:
            await self.conn.executemany(
                "INSERT OR REPLACE INTO candles(symbol, timeframe, ts, open, high, low, close, volume) "
                "VALUES(?,?,?,?,?,?,?,?)",
                [(symbol, timeframe, int(r[0]), *map(float, r[1:6])) for r in rows],
            )
            await self.conn.commit()

    async def prune_candles(self, symbol: str, timeframe: str, before_ms: int) -> None:
        await self._write("DELETE FROM candles WHERE symbol=? AND timeframe=? AND ts < ?",
                          (symbol, timeframe, before_ms))

    async def load_candles(self, symbol: str, timeframe: str) -> list[tuple[Any, ...]]:
        async with self.conn.execute(
            "SELECT ts, open, high, low, close, volume FROM candles WHERE symbol=? AND timeframe=? ORDER BY ts",
            (symbol, timeframe),
        ) as cur:
            return [tuple(r) for r in await cur.fetchall()]


def start_of_utc_day(now: datetime | None = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def hours_from_now(hours: float) -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=hours)
