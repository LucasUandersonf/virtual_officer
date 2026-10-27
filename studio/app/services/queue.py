"""Job queue + scheduler.

* No REDIS_URL (default, cheapest): jobs live in an asyncio queue inside the web process.
* REDIS_URL set: jobs go through a Redis list, so a separate worker (``python -m app.worker``) can run the
  agents while the web service only serves the office; agent events are relayed over Redis pub/sub so the
  office UI stays live either way.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Protocol

log = logging.getLogger(__name__)
JOBS_KEY = "studio:jobs"
EVENTS_CHANNEL = "studio:events"


class Queue(Protocol):
    async def put(self, job: dict[str, Any]) -> None: ...
    async def get(self) -> dict[str, Any]: ...


class MemoryQueue:
    def __init__(self) -> None:
        self._q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def put(self, job: dict[str, Any]) -> None:
        await self._q.put(job)

    async def get(self) -> dict[str, Any]:
        return await self._q.get()


class RedisQueue:
    def __init__(self, url: str) -> None:
        import redis.asyncio as redis  # optional dependency, only imported when used

        self.r = redis.from_url(url, decode_responses=True)

    async def put(self, job: dict[str, Any]) -> None:
        await self.r.lpush(JOBS_KEY, json.dumps(job))

    async def get(self) -> dict[str, Any]:
        while True:
            item = await self.r.brpop([JOBS_KEY], timeout=30)
            if item:
                return dict(json.loads(item[1]))

    async def publish(self, msg: dict[str, Any]) -> None:
        await self.r.publish(EVENTS_CHANNEL, json.dumps(msg, default=str))

    async def relay(self, emit: Any) -> None:
        """Forward events published by a separate worker to this process's WebSocket clients."""
        pubsub = self.r.pubsub()
        await pubsub.subscribe(EVENTS_CHANNEL)
        async for m in pubsub.listen():
            if m.get("type") == "message":
                d = json.loads(m["data"])
                emit(d.pop("type"), **{k: v for k, v in d.items() if k != "ts"})


def make_queue(redis_url: str) -> MemoryQueue | RedisQueue:
    return RedisQueue(redis_url) if redis_url else MemoryQueue()


async def scheduler(queue: Queue, interval_min: int) -> None:
    """Enqueue a pipeline tick now and then every interval."""
    while True:
        await queue.put({"job": "tick"})
        await asyncio.sleep(max(60, interval_min * 60))


async def worker(queue: Queue, office: Any) -> None:
    while True:
        job = await queue.get()
        try:
            kind = job.get("job")
            if kind == "tick":
                await office.tick()
            elif kind == "research":
                async with office._lock:
                    await office.product.research()
            elif kind == "publish":
                await office.sales.publish(await office._product(int(job["product_id"])))
        except Exception:
            log.exception("job failed: %s", job)
