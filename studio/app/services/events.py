"""In-process event bus: agents publish, the office UI listens over /ws.

Each WebSocket client gets its own bounded queue, so a slow browser never
blocks an agent. The latest state of every agent is kept so a new client sees
the office as it is right now.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from app.db import Database
from app.models import AgentEvent

log = logging.getLogger(__name__)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class EventBus:
    def __init__(self, db: Database | None = None) -> None:
        self._db = db
        self._clients: set[asyncio.Queue[dict[str, Any]]] = set()
        self.agent_state: dict[str, dict[str, Any]] = {}
        self.meeting: dict[str, Any] | None = None
        # set by a standalone worker: also publish every event to Redis for the web process
        self.forward: Callable[[dict[str, Any]], Awaitable[None]] | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=200)
        self._clients.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._clients.discard(q)

    def emit(self, type_: str, **payload: Any) -> None:
        msg = {"type": type_, "ts": now_iso(), **payload}
        for q in list(self._clients):
            if q.full():
                try:
                    q.get_nowait()  # drop the oldest; the UI only needs recent state
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(msg)
        if self.forward:
            t = asyncio.get_running_loop().create_task(self.forward(msg))
            self._tasks.add(t)
            t.add_done_callback(self._tasks.discard)

    async def status(self, agent: str, state: str, text: str, **data: Any) -> None:
        """Agent changes what it is doing (shown above its head in the office)."""
        self.agent_state[agent] = {"state": state, "text": text, "ts": now_iso(), **data}
        self.emit("agent_status", agent=agent, state=state, text=text, data=data)

    async def log(self, agent: str, kind: str, text: str, **data: Any) -> None:
        """Persisted activity (feed + sector panels)."""
        log.info("[%s] %s: %s", agent, kind, text)
        if self._db:
            async with self._db.session() as s:
                s.add(AgentEvent(agent=agent, kind=kind, text=text[:4000], data=data))
        self.emit("activity", agent=agent, kind=kind, text=text, data=data)
