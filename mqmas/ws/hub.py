"""Async WebSocket broadcast hub for the pixel-art office.

* One bounded queue per client: a slow browser tab can never block the
  trading loop — when its queue is full, the oldest event is dropped.
* A ring buffer replays recent events to newly connected clients so the
  office is populated instantly (speech bubbles, last telemetry).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from mqmas.models import EventType, WsEvent

log = logging.getLogger(__name__)

CLIENT_QUEUE_SIZE = 256
REPLAY_SIZE = 60


class WebSocketHub:
    def __init__(self) -> None:
        self._clients: dict[WebSocket, asyncio.Queue[str]] = {}
        self._replay: deque[str] = deque(maxlen=REPLAY_SIZE)
        self._last_telemetry: str | None = None
        self._listeners: list[Callable[[str, dict[str, Any]], Awaitable[None]]] = []
        self._pending: set[asyncio.Task[None]] = set()

    def add_listener(self, fn: Callable[[str, dict[str, Any]], Awaitable[None]]) -> None:
        """In-process subscriber (e.g. phone alerts); called without blocking the publisher."""
        self._listeners.append(fn)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def publish(self, event_type: EventType, payload: dict[str, Any]) -> None:
        msg = WsEvent(type=event_type, payload=payload).model_dump_json()
        if event_type == "telemetry":
            self._last_telemetry = msg
        else:
            self._replay.append(msg)
        for q in list(self._clients.values()):
            if q.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()
            q.put_nowait(msg)
        for fn in self._listeners:
            task = asyncio.create_task(self._notify(fn, event_type, payload))
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)

    @staticmethod
    async def _notify(fn: Callable[[str, dict[str, Any]], Awaitable[None]], event_type: str,
                      payload: dict[str, Any]) -> None:
        try:
            await fn(event_type, payload)
        except Exception:
            log.exception("hub listener failed")

    async def serve(self, ws: WebSocket, hello: dict[str, Any]) -> None:
        await ws.accept()
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=CLIENT_QUEUE_SIZE)
        self._clients[ws] = q
        log.info("ws client connected (%d total)", len(self._clients))
        try:
            await ws.send_text(WsEvent(type="hello", payload=hello).model_dump_json())
            for msg in list(self._replay):
                await ws.send_text(msg)
            if self._last_telemetry:
                await ws.send_text(self._last_telemetry)
            sender = asyncio.create_task(self._sender(ws, q))
            receiver = asyncio.create_task(self._receiver(ws))
            done, pending = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            # Retrieve outcomes so disconnect exceptions are not logged as "never retrieved".
            await asyncio.gather(*done, *pending, return_exceptions=True)
        except WebSocketDisconnect:
            pass
        finally:
            self._clients.pop(ws, None)
            log.info("ws client disconnected (%d total)", len(self._clients))

    @staticmethod
    async def _sender(ws: WebSocket, q: asyncio.Queue[str]) -> None:
        while True:
            msg = await q.get()
            await ws.send_text(msg)

    @staticmethod
    async def _receiver(ws: WebSocket) -> None:
        # Clients only send keep-alive pings; anything else is ignored.
        while True:
            await ws.receive_text()
