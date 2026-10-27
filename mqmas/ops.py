"""Operations: phone alerts (Telegram), dead-man heartbeat and daily off-site database backups.

* **Alerts** — every position opened/closed and every risk event (circuit breaker, kill switch, stop
  missing on the exchange, order refused, reconciliation failure) is pushed to Telegram. Messages go
  through a queue so a slow Telegram API never blocks trading.
* **Heartbeat** — while the market feed is alive, a ping is sent every few minutes to
  ``HEARTBEAT_URL`` (e.g. healthchecks.io). If the pings stop — crash, Railway outage, frozen loop —
  that service alerts you; a process cannot report its own death.
* **Backup** — once a day the SQLite file is copied with SQLite's online backup API (consistent
  while the app writes), gzipped, kept locally for ``BACKUP_KEEP`` days and sent to Telegram, so a
  copy exists outside Railway.
"""

from __future__ import annotations

import asyncio
import gzip
import logging
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from mqmas.config import Settings

log = logging.getLogger(__name__)

RISK_TEXT = {
    "circuit_breaker": "⛔ Circuit breaker: novas entradas bloqueadas até {lock_until}",
    "kill_switch": "🛑 Kill switch {state}",
    "stop_missing": "🚨 STOP NÃO COLOCADO na Binance para {symbol} (stop {stop}) — stop por software ativo. {detail}",
    "stop_vanished": "🚨 Stop de {symbol} sumiu da Binance ({status}) — recolocando",
    "order_refused": "⚠️ Ordem recusada antes do envio ({symbol}): {detail}",
    "reconcile_failed": "🚨 Falha ao reconciliar posição {position_id} ({symbol}) — verificar na Binance",
    "orphan_pending_entry": "⚠️ Entrada pendente órfã no boot (posição {position_id}) — verificar na Binance",
}


def format_event(event_type: str, payload: dict[str, Any]) -> str | None:
    if event_type == "position":
        p = payload.get("position") or {}
        mode = str(p.get("mode", "")).upper()
        if payload.get("event") == "opened":
            return (f"🟢 {mode} COMPRA {p.get('symbol')}: {p.get('quantity'):.6g} @ {p.get('entry_price'):.6g} · "
                    f"stop {p.get('stop_price'):.6g} · alvo {p.get('take_profit_price'):.6g}")
        if payload.get("event") == "closed":
            pnl = p.get("pnl_usdt") or 0.0
            return (f"{'✅' if pnl > 0 else '🔴'} {mode} VENDA {p.get('symbol')} ({p.get('exit_reason')}) @ "
                    f"{(p.get('exit_price') or 0):.6g} · resultado US$ {pnl:+.2f}")
        return None
    if event_type == "risk":
        tpl = RISK_TEXT.get(str(payload.get("event")))
        if tpl is None:
            return None
        data = {k: v for k, v in payload.items()}
        data["state"] = "ATIVADO" if payload.get("engaged") else "desativado"
        try:
            return tpl.format_map(_Default(data))
        except (ValueError, KeyError):
            return tpl
    return None


class _Default(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "?"


class Notifier:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._s = settings
        self._token = settings.telegram_bot_token.get_secret_value() if settings.telegram_bot_token else None
        self._chat = settings.telegram_chat_id
        self._client = client or httpx.AsyncClient(timeout=20)
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=200)

    @property
    def enabled(self) -> bool:
        return bool(self._token and self._chat)

    async def on_event(self, event_type: str, payload: dict[str, Any]) -> None:
        text = format_event(event_type, payload)
        if text and self.enabled and not self._queue.full():
            self._queue.put_nowait(f"[{self._s.trading_mode.value}] {text}")

    async def send(self, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            r = await self._client.post(f"https://api.telegram.org/bot{self._token}/sendMessage",
                                        json={"chat_id": self._chat, "text": text})
            return r.status_code == 200
        except httpx.HTTPError as exc:
            log.warning("telegram send failed: %s", type(exc).__name__)
            return False

    async def send_document(self, path: Path, caption: str) -> bool:
        if not self.enabled:
            return False
        try:
            with path.open("rb") as fh:
                r = await self._client.post(f"https://api.telegram.org/bot{self._token}/sendDocument",
                                            data={"chat_id": self._chat, "caption": caption},
                                            files={"document": (path.name, fh)})
            return r.status_code == 200
        except (httpx.HTTPError, OSError) as exc:
            log.warning("telegram document failed: %s", type(exc).__name__)
            return False

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                text = await asyncio.wait_for(self._queue.get(), timeout=5)
            except TimeoutError:
                continue
            await self.send(text)
            await asyncio.sleep(1)  # Telegram: ~1 message/second per chat


def trim_memory() -> None:
    """Give freed heap back to the OS after a heavy job (glibc keeps it otherwise; Railway bills RAM)."""
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def backup_database(db_path: Path, keep: int, now: datetime | None = None) -> Path:
    """Consistent online copy of the SQLite file, gzipped; older copies beyond ``keep`` are deleted."""
    now = now or datetime.now(timezone.utc)
    folder = db_path.parent / "backups"
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / f"{db_path.stem}-{now:%Y%m%d-%H%M}.db"
    src, dst = sqlite3.connect(db_path), sqlite3.connect(raw)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    out = raw.with_suffix(".db.gz")
    with raw.open("rb") as fi, gzip.open(out, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    raw.unlink()
    for old in sorted(folder.glob(f"{db_path.stem}-*.db.gz"))[:-keep]:
        old.unlink()
    return out


async def backup_loop(settings: Settings, notifier: Notifier, stop: asyncio.Event) -> None:
    while not stop.is_set():
        now = datetime.now(timezone.utc)
        nxt = now.replace(hour=settings.backup_hour_utc, minute=0, second=0, microsecond=0)
        if nxt <= now:
            nxt += timedelta(days=1)
        try:
            await asyncio.wait_for(stop.wait(), timeout=(nxt - now).total_seconds())
            return
        except TimeoutError:
            pass
        try:
            path = await asyncio.to_thread(backup_database, settings.db_path, settings.backup_keep)
            sent = await notifier.send_document(path, f"Backup diário MQ-MAS {path.name}")
            log.info("database backup %s (%s)", path.name, "sent to telegram" if sent else "local only")
        except Exception:
            log.exception("database backup failed")


async def heartbeat_loop(settings: Settings, healthy: Any, stop: asyncio.Event,
                         client: httpx.AsyncClient | None = None) -> None:
    """Ping HEARTBEAT_URL while ``healthy()`` is true; the external service alerts when pings stop."""
    if not settings.heartbeat_url:
        return
    client = client or httpx.AsyncClient(timeout=10)
    while not stop.is_set():
        if healthy():
            try:
                await client.get(settings.heartbeat_url)
            except httpx.HTTPError as exc:
                log.warning("heartbeat ping failed: %s", type(exc).__name__)
        try:
            await asyncio.wait_for(stop.wait(), timeout=settings.heartbeat_interval_sec)
        except TimeoutError:
            pass
