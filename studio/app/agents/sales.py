"""Agent 4 — Leo, Account Executive & Ops: publish listings, record sales, deliver the download, onboard the buyer."""

from __future__ import annotations

import html
import secrets
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.agents.base import Agent
from app.integrations.payments import GumroadClient, ListingResult, SaleEvent, StripeClient, WhopClient, estimate_fee
from app.models import Download, Listing, MarketingKit, Product, Sale, utcnow


def listing_description(p: Product, kit: MarketingKit | None) -> str:
    if not kit:
        return f"{p.tagline}\n\n{p.idea.mvp}"
    lp = kit.landing
    bullets = "\n".join(f"• {b}" for b in lp.get("benefits", []))
    return f"{lp.get('subheadline', '')}\n\n{bullets}\n\nYou get: a .zip with the solution, an English install guide, .env.example and scripts."


class Sales(Agent):
    key = "sales"

    def _clients(self) -> dict[str, StripeClient | GumroadClient | WhopClient]:
        s, h = self.ctx.settings, self.ctx.http
        all_ = {"stripe": StripeClient(s, h), "gumroad": GumroadClient(s, h), "whop": WhopClient(s, h)}
        return {k: v for k, v in all_.items() if k in s.platforms()}

    async def publish(self, product: Product) -> list[Listing]:
        clients = self._clients()
        await self.status("working", f"Publishing {product.name} on {', '.join(clients) or 'the studio page'}")
        async with self.ctx.db.session() as ses:
            kit = await ses.scalar(select(MarketingKit).where(MarketingKit.product_id == product.id))
        desc = listing_description(product, kit)
        out: list[Listing] = []
        for name, client in clients.items():
            try:
                res = await client.publish(product.id, product.name, desc, product.price_cents)
            except Exception as exc:  # one platform down must not block the others
                res = ListingResult(name, "", "", "error", str(exc)[:500])
            async with self.ctx.db.session() as ses:
                row = await ses.scalar(select(Listing).where(Listing.product_id == product.id, Listing.platform == name))
                if not row:
                    row = Listing(product_id=product.id, platform=name)
                    ses.add(row)
                row.external_id, row.url, row.status, row.note = res.external_id, res.url, res.status, res.note
            out.append(row)
        live = [r for r in out if r.status == "live"]
        page = f"{self.ctx.settings.public_base_url}/p/{product.slug}"
        async with self.ctx.db.session() as ses:
            p = await ses.get(Product, product.id)
            if p and live:
                p.status, p.published_at = "published", utcnow()
        if live:
            await self.log("publish", f"{product.name} live on {', '.join(r.platform for r in live)} at ${product.price_cents / 100:.0f}",
                           product_id=product.id, page=page)
            await self.ctx.notify.ceo(f"🚀 New product published: {product.name} (${product.price_cents / 100:.0f})\n"
                                      f"Platforms: {', '.join(r.platform for r in live)}\nLanding: {page}")
            await self.status("idle", f"{product.name} is live")
        else:
            why = "; ".join(f"{r.platform}: {r.note}" for r in out) or "no sales platform configured (set STRIPE_SECRET_KEY)"
            await self.log("publish_blocked", f"{product.name} not live yet: {why}", product_id=product.id)
            await self.status("blocked", "Waiting for a sales platform key")
        return out

    async def record_sale(self, ev: SaleEvent) -> Sale | None:
        """Idempotent per (platform, external_id). Refund events flip an existing sale."""
        s = self.ctx.settings
        async with self.ctx.db.session() as ses:
            existing = await ses.scalar(select(Sale).where(Sale.platform == ev.platform, Sale.external_id == ev.external_id))
            if ev.refunded:
                if existing:
                    existing.refunded = True
                    await self.log("refund", f"Refund on {ev.platform}: ${existing.gross_cents / 100:.2f}", sale_id=existing.id)
                return existing
            if existing:
                return existing
            listing = None
            if ev.listing_ref:
                listing = await ses.scalar(select(Listing).where(Listing.platform == ev.platform, Listing.external_id == ev.listing_ref))
            fee = ev.fee_cents if ev.fee_cents is not None else estimate_fee(s, ev.platform, ev.gross_cents)
            sale = Sale(platform=ev.platform, external_id=ev.external_id, product_id=listing.product_id if listing else None,
                        buyer_email=ev.buyer_email, gross_cents=ev.gross_cents, fee_cents=fee, fee_estimated=ev.fee_cents is None)
            ses.add(sale)
            try:
                await ses.flush()
            except IntegrityError:
                return None
            product = await ses.get(Product, sale.product_id) if sale.product_id else None
        name = product.name if product else "unknown product"
        await self.log("sale", f"💰 Sale: {name} ${ev.gross_cents / 100:.2f} on {ev.platform}", sale_id=sale.id,
                       product_id=sale.product_id, gross_usd=ev.gross_cents / 100)
        self.ctx.bus.emit("sale", product=name, gross_usd=ev.gross_cents / 100, platform=ev.platform)
        await self.ctx.notify.ceo(f"💰 New sale: {name} — ${ev.gross_cents / 100:.2f} on {ev.platform}")
        if product:
            await self.deliver(sale, product)
        return sale

    async def deliver(self, sale: Sale, product: Product) -> str:
        s = self.ctx.settings
        token = secrets.token_urlsafe(32)
        async with self.ctx.db.session() as ses:
            ses.add(Download(token=token, sale_id=sale.id, product_id=product.id, max_count=s.download_max,
                             expires_at=utcnow() + timedelta(days=s.download_ttl_days)))
            kit = await ses.scalar(select(MarketingKit).where(MarketingKit.product_id == product.id))
        link = f"{s.public_base_url}/d/{token}"
        first = (kit.emails[0] if kit and kit.emails else {"subject": f"Your {product.name} is ready", "body": ""})
        body = (f"<p>Thanks for buying <b>{html.escape(product.name)}</b>!</p><p><a href=\"{link}\">Download your package</a> "
                f"(valid {s.download_ttl_days} days, {s.download_max} downloads).</p><p>{html.escape(first['body'])}</p>"
                "<p>Start with README.md: setup takes about 10 minutes. Reply to this email if you get stuck.</p>")
        sent = await self.ctx.notify.email(sale.buyer_email, first["subject"], body)
        async with self.ctx.db.session() as ses:
            row = await ses.get(Sale, sale.id)
            if row:
                row.delivered = sent
        await self.log("delivery", f"Download link {'emailed to buyer' if sent else 'created (email not configured)'} for {product.name}",
                       sale_id=sale.id)
        return link
