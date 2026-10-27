"""CEO alerts (Telegram bot / Discord webhook) and buyer emails (Resend). All free tiers, all optional."""

from __future__ import annotations

import logging

import httpx

from app.config import Settings, secret

log = logging.getLogger(__name__)


class Notifier:
    def __init__(self, s: Settings, client: httpx.AsyncClient) -> None:
        self.s, self.client = s, client

    async def ceo(self, text: str) -> None:
        """Short executive alert. Failures are logged, never raised (an alert must not break a sale)."""
        try:
            if self.s.telegram_bot_token and self.s.telegram_chat_id:
                await self.client.post(f"https://api.telegram.org/bot{secret(self.s.telegram_bot_token)}/sendMessage",
                                       json={"chat_id": self.s.telegram_chat_id, "text": text, "disable_web_page_preview": True})
            if self.s.discord_webhook_url:
                await self.client.post(secret(self.s.discord_webhook_url), json={"content": text[:1900]})
        except httpx.HTTPError as exc:
            log.warning("CEO alert failed: %s", exc)

    async def email(self, to: str, subject: str, html: str) -> bool:
        if not (self.s.resend_api_key and to):
            log.info("email to %s skipped (no RESEND_API_KEY): %s", to, subject)
            return False
        try:
            r = await self.client.post("https://api.resend.com/emails",
                                       headers={"Authorization": f"Bearer {secret(self.s.resend_api_key)}"},
                                       json={"from": self.s.email_from, "to": [to], "subject": subject, "html": html})
            return r.status_code < 300
        except httpx.HTTPError as exc:
            log.warning("email to %s failed: %s", to, exc)
            return False
