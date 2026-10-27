"""``python -m mqmas.runner [--status | --reset-halt]``: one tick of the trend allocator, then exit.

Built for a cron job (Railway cron, a home server's crontab, a free VM): the process lives for about a
minute per run, so hosting costs are close to zero, and nothing is lost when it is not running because
the crash stops rest on Binance and the state lives in SQLite. Run it hourly; the daily rebalance happens
on the first run after 00:00 UTC.

Exit code 0 on success, 1 on any failure (the cron platform shows the run as failed, the heartbeat URL is
not pinged, and Telegram gets the error).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any

import httpx

from mqmas.config import Settings, get_settings
from mqmas.logging_setup import configure_logging
from mqmas.ops import Notifier
from mqmas.persistence.db import Database
from mqmas.research.data import PUBLIC_DATA_URL
from mqmas.trend.live import TrendAllocator

log = logging.getLogger("mqmas.runner")


def build_exchange(s: Settings) -> Any:
    import ccxt.async_support as ccxt_async

    opts: dict[str, Any] = {"enableRateLimit": True, "options": {
        "defaultType": "spot", "fetchMarkets": {"types": ["spot"]}, "adjustForTimeDifference": True,
        "recvWindow": 10_000}}
    if s.exchange_api_key and s.exchange_api_secret:
        opts["apiKey"] = s.exchange_api_key.get_secret_value()
        opts["secret"] = s.exchange_api_secret.get_secret_value()
    ex = getattr(ccxt_async, s.exchange_id)(opts)
    if os.environ.get("MQMAS_PUBLIC_DATA") == "1" and not s.is_live:  # paper from a region api.binance.com refuses
        ex.urls["api"]["public"] = PUBLIC_DATA_URL
    return ex


async def run(argv: list[str], settings: Settings | None = None, exchange: Any = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(prog="python -m mqmas.runner")
    ap.add_argument("--status", action="store_true", help="print the saved state and exit (no orders)")
    ap.add_argument("--reset-halt", action="store_true", help="clear a drawdown halt; trading resumes next run")
    args = ap.parse_args(argv)
    s = settings or get_settings()
    db = Database(s.db_path)
    await db.connect()
    ex = exchange if exchange is not None else build_exchange(s)
    notifier = Notifier(s)
    alloc = TrendAllocator(s, db, ex, notify=notifier.send if notifier.enabled else None)
    try:
        if args.status or args.reset_halt:
            st = await alloc.load()
            if args.reset_halt:
                st.halted, st.peak = None, 0.0
                await alloc.save()
                log.info("halt cleared; the next run trades again")
            return alloc.report(st.cash)
        report = await alloc.run_once()
        if s.heartbeat_url:
            async with httpx.AsyncClient(timeout=10) as c:
                try:
                    await c.get(s.heartbeat_url)
                except httpx.HTTPError as exc:
                    log.warning("heartbeat failed: %s", type(exc).__name__)
        return report
    except Exception as exc:
        if notifier.enabled:
            await notifier.send(f"Alocador de tendência ({alloc.mode}) FALHOU: {type(exc).__name__}: {exc}"[:3500])
        raise
    finally:
        if exchange is None:
            await ex.close()
        await db.close()


def main() -> None:
    s = get_settings()
    configure_logging(s.log_level)
    try:
        report = asyncio.run(run(sys.argv[1:], s))
    except Exception:
        log.exception("trend run failed")
        sys.exit(1)
    print(json.dumps(report, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
