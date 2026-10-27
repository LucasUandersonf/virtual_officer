"""Agent 4 — Leo, Account Executive & Ops: publish listings, record sales, deliver the download, onboard the buyer."""

from __future__ import annotations

import html
import secrets
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.agents.base import Agent
from app.agents.frameworks import PRICE_MAX, PRICE_MIN, charm_price
from app.config import secret
from app.integrations.payments import (
    CheckoutSignal,
    GumroadClient,
    ListingResult,
    SaleEvent,
    StripeClient,
    WhopClient,
    estimate_fee,
)
from app.models import Download, FunnelDay, Listing, MarketingKit, Product, Sale, utcnow

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
        promo_note = ""
        stripe_client = clients.get("stripe")
        if any(r.platform == "stripe" for r in live) and isinstance(stripe_client, StripeClient) and self.ctx.settings.launch_discount_pct > 0:
            try:
                promo = await stripe_client.launch_promo(product.slug, self.ctx.settings.launch_discount_pct)
                promos = await self.ctx.db.get_state("promos")
                promos[str(product.id)] = promo
                await self.ctx.db.set_state("promos", promos)
                promo_note = f" · launch code {promo['code']} (-{promo['percent']}% for 48 h)"
            except Exception as exc:
                await self.log("promo_error", f"Launch coupon not created: {str(exc)[:200]}", product_id=product.id)
        if live:
            await self.log("publish", f"{product.name} live on {', '.join(r.platform for r in live)} at ${product.price_cents / 100:.0f}"
                           f"{promo_note}",
                           product_id=product.id, page=page)
            await self.ctx.notify.ceo(f"🚀 New product published: {product.name} (${product.price_cents / 100:.0f})\n"
                                      f"Platforms: {', '.join(r.platform for r in live)}\nLanding: {page}")
            await self.status("idle", f"{product.name} is live")
        else:
            why = "; ".join(f"{r.platform}: {r.note}" for r in out) or "no sales platform configured (set STRIPE_SECRET_KEY)"
            await self.log("publish_blocked", f"{product.name} not live yet: {why}", product_id=product.id)
            await self.status("blocked", "Waiting for a sales platform key")
        return out

    STRIPE_EVENTS = ["checkout.session.completed", "checkout.session.expired", "charge.refunded", "charge.dispute.created",
                     "payment_intent.payment_failed"]

    async def self_configure(self) -> dict[str, Any]:
        """Leo sets up his own plumbing: Stripe webhook (all events he listens to) + wallets, Gumroad sale pings."""
        s = self.ctx.settings
        clients = self._clients()
        report: dict[str, Any] = {}
        public = s.public_base_url.startswith("https://") and "localhost" not in s.public_base_url
        stripe = clients.get("stripe")
        if isinstance(stripe, StripeClient):
            if not public:
                report["stripe_webhook"] = "waiting for a public https PUBLIC_BASE_URL (Railway domain)"
            else:
                stored = await self.ctx.db.get_state("stripe_webhook")
                known = secret(s.stripe_webhook_secret) or stored.get("secret", "")
                try:
                    res = await stripe.ensure_webhook(f"{s.public_base_url}/webhooks/stripe", self.STRIPE_EVENTS, known)
                    if res.get("secret"):
                        await self.ctx.db.set_state("stripe_webhook", {"id": res["id"], "secret": res["secret"]})
                    report["stripe_webhook"] = f"{res['action']} ({len(self.STRIPE_EVENTS)} events)"
                except Exception as exc:
                    report["stripe_webhook"] = f"error: {str(exc)[:160]}"
            try:
                w = await stripe.enable_wallets()
                report["wallets"] = f"Apple Pay / Google Pay: {w['action']}"
            except Exception as exc:
                report["wallets"] = f"error: {str(exc)[:160]} (enable them in Stripe -> Settings -> Payment methods)"
        gum = clients.get("gumroad")
        if isinstance(gum, GumroadClient) and public:
            st = await self.ctx.db.get_state("gumroad_webhook")
            key = secret(s.gumroad_webhook_secret) or st.get("key") or secrets.token_urlsafe(24)
            if not st.get("subscribed"):
                try:
                    await gum.subscribe_sales(key)
                    await self.ctx.db.set_state("gumroad_webhook", {"key": key, "subscribed": True})
                    report["gumroad_ping"] = "subscribed to sale pings"
                except Exception as exc:
                    report["gumroad_ping"] = f"error: {str(exc)[:160]}"
            else:
                report["gumroad_ping"] = "already subscribed"
        await self.ctx.db.set_state("autoconfig_sales", report)
        if report:
            await self.log("setup", "Self-configuration: " + "; ".join(f"{k}: {v}" for k, v in report.items()))
        return report

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
        """Idempotent per (platform, external_id). Refunds and disputes flip the sale and revoke its downloads."""
        if ev.refunded:
            return await self._reverse(ev)
        s = self.ctx.settings
        async with self.ctx.db.session() as ses:
            existing = await ses.scalar(select(Sale).where(Sale.platform == ev.platform, Sale.external_id == ev.external_id))
            if existing:
                return existing
            listing = None
            if ev.listing_ref:
                listing = await ses.scalar(select(Listing).where(Listing.platform == ev.platform, Listing.external_id == ev.listing_ref))
            fee = ev.fee_cents if ev.fee_cents is not None else estimate_fee(s, ev.platform, ev.gross_cents)
            sale = Sale(platform=ev.platform, external_id=ev.external_id, product_id=listing.product_id if listing else None,
                        buyer_email=ev.buyer_email, gross_cents=ev.gross_cents, fee_cents=fee, fee_estimated=ev.fee_cents is None,
                        payment_ref=ev.payment_ref or None)
            ses.add(sale)
            try:
                await ses.flush()
            except IntegrityError:
                return None
            product = await ses.get(Product, sale.product_id) if sale.product_id else None
            recent = await ses.scalar(select(func.count(Sale.id)).where(
                Sale.buyer_email == ev.buyer_email, Sale.created_at >= utcnow() - timedelta(hours=1))) if ev.buyer_email else 0
        name = product.name if product else "unknown product"
        await self.log("sale", f"💰 Sale: {name} ${ev.gross_cents / 100:.2f} on {ev.platform}", sale_id=sale.id,
                       product_id=sale.product_id, gross_usd=ev.gross_cents / 100)
        self.ctx.bus.emit("sale", product=name, gross_usd=ev.gross_cents / 100, platform=ev.platform)
        await self.ctx.notify.ceo(f"💰 New sale: {name} — ${ev.gross_cents / 100:.2f} on {ev.platform}")
        if (recent or 0) >= 3:  # velocity check: flag, don't block (Stripe Radar scores the card itself)
            await self.log("fraud_flag", f"{recent} purchases from {ev.buyer_email} in 1 hour — review before honoring support requests",
                           sale_id=sale.id)
            await self.ctx.notify.ceo(f"🚩 Possible card testing: {recent} purchases from the same email in 1 hour ({name}).")
        if product:
            await self.deliver(sale, product)
        if ev.platform == "stripe" and ev.payment_ref:
            await self._audit_fee(sale.id, ev.payment_ref)
        return sale

    async def _reverse(self, ev: SaleEvent) -> Sale | None:
        async with self.ctx.db.session() as ses:
            sale = await ses.scalar(select(Sale).where(Sale.platform == ev.platform,
                                                       (Sale.payment_ref == ev.payment_ref) | (Sale.external_id == ev.external_id)))
            if not sale:
                return None
            sale.refunded = True
            sale.disputed = sale.disputed or ev.disputed
            for d in (await ses.scalars(select(Download).where(Download.sale_id == sale.id))).all():
                d.count = d.max_count  # access revoked
        what = "Dispute (chargeback)" if ev.disputed else "Refund"
        await self.log("dispute" if ev.disputed else "refund", f"{what} on {ev.platform}: ${sale.gross_cents / 100:.2f}; download revoked",
                       sale_id=sale.id)
        if ev.disputed:
            await self.ctx.notify.ceo(f"⚠️ Chargeback opened: ${sale.gross_cents / 100:.2f} ({sale.buyer_email}). Download access revoked; "
                                      "answer it in the Stripe dashboard with the delivery log.")
        return sale

    async def _audit_fee(self, sale_id: int, payment_intent: str) -> None:
        """Replace the estimated Stripe fee by the real one (and record what settles, in which currency)."""
        client = self._clients().get("stripe")
        if not isinstance(client, StripeClient):
            return
        try:
            st = await client.settlement(payment_intent)
        except Exception:
            st = None
        if not st:
            return
        async with self.ctx.db.session() as ses:
            sale = await ses.get(Sale, sale_id)
            if sale:
                sale.fee_cents, sale.fee_estimated = st["fee_usd_cents"], False
                sale.settle_currency, sale.settle_net = st["currency"], st["net"]

    async def related(self, product: Product, limit: int = 2) -> list[Product]:
        """Cross-sell: other live products for the same kind of buyer (same ecosystem/type), best sellers first."""
        async with self.ctx.db.session() as ses:
            rows = (await ses.scalars(select(Product).where(Product.status == "published", Product.id != product.id))).unique().all()
            sold = dict((await ses.execute(select(Sale.product_id, func.count(Sale.id)).where(Sale.refunded.is_(False))
                                           .group_by(Sale.product_id))).all())
        same = [p for p in rows if p.solution_type == product.solution_type]
        return sorted(same, key=lambda p: -sold.get(p.id, 0))[:limit]

    async def deliver(self, sale: Sale, product: Product) -> str:
        s = self.ctx.settings
        token = secrets.token_urlsafe(32)
        async with self.ctx.db.session() as ses:
            ses.add(Download(token=token, sale_id=sale.id, product_id=product.id, max_count=s.download_max,
                             expires_at=utcnow() + timedelta(days=s.download_ttl_days)))
            kit = await ses.scalar(select(MarketingKit).where(MarketingKit.product_id == product.id))
        link = f"{s.public_base_url}/d/{token}"
        first = (kit.emails[0] if kit and kit.emails else {"subject": f"Your {product.name} is ready", "body": ""})
        cross = await self.related(product, 1)
        extra = (f"<p>Buyers of {html.escape(product.name)} often add <a href=\"{s.public_base_url}/p/{cross[0].slug}\">"
                 f"{html.escape(cross[0].name)}</a> — {html.escape(cross[0].tagline)}</p>" if cross else "")
        body = (f"<p>Thanks for buying <b>{html.escape(product.name)}</b>!</p><p><a href=\"{link}\">Download your package</a> "
                f"(valid {s.download_ttl_days} days, {s.download_max} downloads).</p><p>{html.escape(first['body'])}</p>"
                "<p>Start with the 3-Step Quickstart at the top of README.md. Reply to this email if you get stuck.</p>" + extra)
        sent = await self.ctx.notify.email(sale.buyer_email, first["subject"], body)
        async with self.ctx.db.session() as ses:
            row = await ses.get(Sale, sale.id)
            if row:
                row.delivered = sent
        await self.log("delivery", f"Download link {'emailed to buyer' if sent else 'created (email not configured)'} for {product.name}",
                       sale_id=sale.id)
        return link

    # ------------------------------------------------------------------ funnel + elasticity
    async def track(self, product_id: int, field: str) -> None:
        today = utcnow().date()
        async with self.ctx.db.session() as ses:
            row = await ses.scalar(select(FunnelDay).where(FunnelDay.product_id == product_id, FunnelDay.day == today))
            if not row:
                row = FunnelDay(product_id=product_id, day=today, views=0, clicks=0)
                ses.add(row)
            setattr(row, field, (getattr(row, field) or 0) + 1)

    async def funnel(self, days: int = 14) -> list[dict[str, Any]]:
        since = utcnow() - timedelta(days=days)
        async with self.ctx.db.session() as ses:
            prods = (await ses.scalars(select(Product).where(Product.status == "published"))).unique().all()
            fun = {pid: (int(v or 0), int(c or 0)) for pid, v, c in (await ses.execute(
                select(FunnelDay.product_id, func.sum(FunnelDay.views), func.sum(FunnelDay.clicks))
                .where(FunnelDay.day >= since.date()).group_by(FunnelDay.product_id))).all()}
            sold = dict((await ses.execute(select(Sale.product_id, func.count(Sale.id)).where(
                Sale.refunded.is_(False), Sale.created_at >= since).group_by(Sale.product_id))).all())
        out = []
        for p in prods:
            v, c = fun.get(p.id, (0, 0))
            n = int(sold.get(p.id, 0))
            out.append({"id": p.id, "name": p.name, "topic": p.idea.topic if p.idea else "", "price_usd": p.price_cents / 100,
                        "views": v, "clicks": c, "sales": n, "ctr": round(c / v, 4) if v else None,
                        "checkout_cr": round(n / c, 4) if c else None, "visitor_to_buyer": round(n / v, 4) if v else None})
        return out

    async def funnel_review(self, growth: Any = None) -> list[dict[str, Any]]:
        """Daily: diagnose each live product's funnel and act — copy problem -> Maya; price problem -> reprice."""
        state = await self.ctx.db.get_state("funnel")
        last_reprice = state.get("last_reprice", {})
        rewritten = set(state.get("rewritten", []))
        actions = []
        for f in await self.funnel():
            if f["views"] < 100:
                continue  # not enough traffic to judge
            diagnosis, act = "", ""
            if (f["ctr"] or 0) < 0.03 and growth is not None and f["id"] not in rewritten:
                diagnosis, act = f"landing CTR {f['ctr']:.1%} < 3%: the copy is not converting", "Maya rewrites the landing"
                async with self.ctx.db.session() as ses:
                    prod = await ses.get(Product, f["id"])
                if prod:
                    await growth.create_kit(prod, angle=f"Funnel data: {f['views']} views but only {f['clicks']} buy clicks. "
                                                         "Rewrite the headline and first screen around the sharpest loss; keep the numbers.")
                rewritten.add(f["id"])
            elif f["clicks"] >= 20 and (f["checkout_cr"] or 0) < 0.10:
                diagnosis, act = f"checkout conversion {f['checkout_cr']:.0%} < 10% with {f['clicks']} clicks: price friction", "lower price"
                act = await self._elastic(f, -1, last_reprice)
            elif f["sales"] >= 5 and (f["visitor_to_buyer"] or 0) > 0.05:
                diagnosis, act = f"visitor-to-buyer {f['visitor_to_buyer']:.1%} > 5%: demand is strong", "raise price"
                act = await self._elastic(f, +1, last_reprice)
            if diagnosis:
                await self.log("funnel", f"{f['name']}: {diagnosis} → {act}", product_id=f["id"])
                actions.append({**f, "diagnosis": diagnosis, "action": act})
        topic_cr: dict[str, list[float]] = {}
        for f in await self.funnel(30):
            if f["views"] >= 100 and f["topic"]:
                topic_cr.setdefault(f["topic"], []).append(f["visitor_to_buyer"] or 0.0)
        await self.ctx.db.set_state("funnel", {"last_reprice": last_reprice, "rewritten": sorted(rewritten),
                                               "topic_cr": {t: sum(v) / len(v) for t, v in topic_cr.items()}})
        return actions

    async def _elastic(self, f: dict[str, Any], direction: int, last_reprice: dict[str, str]) -> str:
        key = str(f["id"])
        if key in last_reprice and utcnow() - datetime.fromisoformat(last_reprice[key]) < timedelta(days=7):
            return "price change skipped (one change per 7 days)"
        ladder = [p for p in range(PRICE_MIN, PRICE_MAX + 1) if p % 10 in (7, 9)]
        cur = charm_price(f["price_usd"])
        i = ladder.index(cur) + direction
        if not 0 <= i < len(ladder):
            return "already at the price limit"
        new = ladder[i]
        done = await self.reprice(f["id"], new)
        last_reprice[key] = utcnow().isoformat()
        return f"${cur} → ${new} ({done})"

    async def reprice(self, pid: int, new_usd: int) -> str:
        async with self.ctx.db.session() as ses:
            listings = (await ses.scalars(select(Listing).where(Listing.product_id == pid, Listing.status == "live"))).all()
        client = self._clients().get("stripe")
        done = []
        for x in listings:
            if x.platform == "stripe" and isinstance(client, StripeClient) and "product " in x.note:
                ref = x.note.split("product ", 1)[1].split()[0]
                try:
                    res = await client.reprice(ref, x.external_id, pid, new_usd * 100)
                except Exception as exc:
                    done.append(f"stripe failed: {str(exc)[:80]}")
                    continue
                async with self.ctx.db.session() as ses:
                    row = await ses.get(Listing, x.id)
                    if row:
                        row.external_id, row.url, row.note = res.external_id, res.url, res.note
                done.append("stripe updated")
            else:
                done.append(f"{x.platform}: update the price manually")
        async with self.ctx.db.session() as ses:
            p = await ses.get(Product, pid)
            if p:
                p.price_cents = new_usd * 100
        return ", ".join(done) or "studio price updated"
