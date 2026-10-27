from __future__ import annotations

import asyncio
import gzip
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import httpx

from mqmas.config import Settings
from mqmas.ops import Notifier, backup_database, format_event, heartbeat_loop
from mqmas.ws.hub import WebSocketHub


def test_alert_texts():
    opened = {"event": "opened", "position": {"mode": "live", "symbol": "SOL/USDT", "quantity": 0.1,
                                               "entry_price": 100.0, "stop_price": 90.0, "take_profit_price": 118.0}}
    assert "COMPRA SOL/USDT" in format_event("position", opened)
    assert "STOP NÃO COLOCADO" in format_event("risk", {"event": "stop_missing", "symbol": "SOL/USDT", "stop": 90})
    assert format_event("telemetry", {}) is None and format_event("risk", {"event": "unknown"}) is None


def test_hub_listener_sends_to_telegram(tmp_path):
    sent: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        sent.append(json.loads(req.content))
        return httpx.Response(200, json={"ok": True})

    s = Settings(_env_file=None, db_path=tmp_path / "t.db", telegram_bot_token="t", telegram_chat_id="42")
    n = Notifier(s, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    async def go() -> None:
        hub = WebSocketHub()
        hub.add_listener(n.on_event)
        await hub.publish("risk", {"event": "kill_switch", "engaged": True})
        await asyncio.sleep(0)
        stop = asyncio.Event()
        task = asyncio.create_task(n.run(stop))
        await asyncio.sleep(0.1)
        stop.set()
        await task

    asyncio.run(go())
    assert sent and sent[0]["chat_id"] == "42" and "Kill switch ATIVADO" in sent[0]["text"]


def test_backup_is_consistent_and_rotated(tmp_path):
    db = tmp_path / "mqmas.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE t (x INTEGER)")
    con.execute("INSERT INTO t VALUES (7)")
    con.commit()
    con.close()
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    paths = [backup_database(db, keep=3, now=t0 + timedelta(days=i)) for i in range(5)]
    left = sorted((tmp_path / "backups").glob("*.gz"))
    assert left == paths[-3:]
    raw = tmp_path / "restored.db"
    raw.write_bytes(gzip.decompress(left[-1].read_bytes()))
    assert sqlite3.connect(raw).execute("SELECT x FROM t").fetchone() == (7,)


def test_heartbeat_pings_only_while_healthy(tmp_path):
    hits: list[str] = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: hits.append(str(r.url)) or httpx.Response(200)))
    s = Settings(_env_file=None, db_path=tmp_path / "t.db", heartbeat_url="https://hc.test/ping",
                 heartbeat_interval_sec=60)

    async def go(healthy: bool) -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(heartbeat_loop(s, lambda: healthy, stop, client))
        await asyncio.sleep(0.05)
        stop.set()
        await task

    asyncio.run(go(False))
    assert hits == []
    asyncio.run(go(True))
    assert hits == ["https://hc.test/ping"]
