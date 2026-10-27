"""Standalone worker: ``python -m app.worker`` (runs the agents; needs REDIS_URL so the web service sees events).

``python -m app.worker --once`` runs a single pipeline tick and exits — usable as a Railway cron job when
you prefer paying only for the minutes the agents work (webhooks still need the web service).
"""

from __future__ import annotations

import asyncio
import logging
import sys

from app.agents.office import Office
from app.config import get_settings
from app.main import build_context
from app.services.queue import RedisQueue, make_queue, scheduler, worker


async def main(once: bool) -> None:
    s = get_settings()
    logging.basicConfig(level=s.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ctx = build_context(s)
    await ctx.db.init()
    office = Office(ctx)
    queue = make_queue(s.redis_url)
    if isinstance(queue, RedisQueue):
        ctx.bus.forward = queue.publish
    try:
        if once:
            print(await office.tick())
            return
        await asyncio.gather(worker(queue, office), scheduler(queue, s.pipeline_interval_min))
    finally:
        await ctx.llm.aclose()
        await ctx.http.aclose()
        await ctx.db.close()


if __name__ == "__main__":
    asyncio.run(main("--once" in sys.argv))
