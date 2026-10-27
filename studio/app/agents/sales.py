"""Agent 4 — Leo, Account Executive & Ops: publish listings, record sales, deliver the download, onboard the buyer."""

from __future__ import annotations

import html
import secrets
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.agents.base import Agent
from app.integrations.payments import (
    CheckoutSignal,
    GumroadClient,
    ListingResult,
    SaleEvent,
    StripeClient,
    WhopClient,
    estimate_fee,
)
from app.models import Download, Listing, MarketingKit, Product, Sale, utcnow

YOU_GET = "You get: a .zip with the solution, a 5-step English install guide, .env.example and scripts. Instant download."
LIMITS = {"stripe": 500, "whop": 1000, "gumroad": 5000}


def listing_description(p: Product, kit: MarketingKit | None, platform: str = "stripe") -> str:
    """Formatted for each platform: Stripe shows short plain text, Whop plain text with bullets, Gumroad renders HTML."""
    lp = (kit.landing if kit else None) or {"subheadline": p.tagline, "benefits": [p.idea.mvp] if p.idea else []}
    benefits = lp.get("benefits", [])[:5]
    anchor = lp.get("value_anchor", "")
    if platform == "gumroad":
        e = html.escape
        pas = lp.get("pas") or {}
        body = (f"<p><strong>{e(lp.get('subheadline', ''))}</strong></p>"
                + (f"<p>{e(pas.get('problem', ''))} {e(pas.get('agitate', ''))}</p><p>{e(pas.get('solve', ''))}</p>" if pas else "")
                + "<ul>" + "".join(f"<li>{e(b)}</li>" for b in benefits) + "</ul>"
                + (f"<p>{e(anchor)}</p>" if anchor else "") + f"<p>{e(YOU_GET)}</p>")
    elif platform == "whop":
        body = f"{lp.get('subheadline', '')}\n\n" + "\n".join(f"• {b}" for b in benefits) + (f"\n\n{anchor}" if anchor else "") + f"\n\n{YOU_GET}"
    else:
        body = f"{lp.get('subheadline', '')} {anchor} {YOU_GET}".strip()
    return body[: LIMITS.get(platform, 1000)]


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
        out: list[Listing] = []
        for name, client in clients.items():
            try:
                res = await client.publish(product.id, product.name, listing_description(product, kit, name), product.price_cents)
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

    async def pause(self, pid: int) -> None:
        """Stop selling a product: deactivate its Stripe payment link and mark it paused (reversible: approve again)."""
        async with self.ctx.db.session() as ses:
            p = await ses.get(Product, pid)
            listings = (await ses.scalars(select(Listing).where(Listing.product_id == pid))).all()
        clients = self._clients()
        for x in listings:
            if x.platform == "stripe" and x.status == "live" and isinstance(clients.get("stripe"), StripeClient):
                try:
                    await clients["stripe"].deactivate(x.external_id)  # type: ignore[union-attr]
                except Exception as exc:
                    await self.log("pause_error", f"Could not deactivate {x.external_id}: {exc}")
                    continue
            async with self.ctx.db.session() as ses:
                row = await ses.get(Listing, x.id)
                if row:
                    row.status = "paused"
        async with self.ctx.db.session() as ses:
            row = await ses.get(Product, pid)
            if row:
                row.status = "paused"
        await self.log("pause", f"Paused {p.name if p else pid}")

    async def checkout_signal(self, sig: CheckoutSignal) -> str:
        """Abandoned checkout -> one recovery email, only with marketing consent. Failed payment -> logged for the CEO."""
        stats = await self.ctx.db.get_state("checkout")
        stats[sig.kind] = int(stats.get(sig.kind, 0)) + 1
        action = "logged"
        if sig.kind == "abandoned" and sig.email and sig.consent:
            sent_to = set(stats.get("recovered_to", []))
            key = f"{sig.email.lower()}|{sig.listing_ref}"
            async with self.ctx.db.session() as ses:
                listing = await ses.scalar(select(Listing).where(Listing.platform == "stripe", Listing.external_id == sig.listing_ref))
                product = await ses.get(Product, listing.product_id) if listing else None
            if listing and product and key not in sent_to:
                body = (f"<p>You were one step away from <b>{html.escape(product.name)}</b>.</p>"
                        f"<p>{html.escape(product.tagline)}</p><p><a href=\"{listing.url}\">Finish your order</a> "
                        "— instant download, 5-minute setup.</p><p>Questions before buying? Just reply.</p>")
                if await self.ctx.notify.email(sig.email, f"Still want {product.name}?", body):
                    sent_to.add(key)
                    stats["recovery_emails"] = int(stats.get("recovery_emails", 0)) + 1
                    action = "recovery email sent"
            stats["recovered_to"] = sorted(sent_to)[-2000:]
        elif sig.kind == "abandoned":
            action = "no consent: not contacted"
        await self.ctx.db.set_state("checkout", stats)
        await self.log(sig.kind, f"Checkout {sig.kind}{' (' + sig.reason + ')' if sig.reason else ''}: {action}")
        if sig.kind == "failed":
            await self.ctx.notify.ceo(f"⚠️ Payment failed ({sig.reason}). Failed so far: {stats['failed']}")
        return action

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
