"""Public pages: payment webhooks, product landing pages (/p/{slug}), the buyer's download link (/d/{token})."""

from __future__ import annotations

import hmac
import html
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from sqlalchemy import select

from app.config import secret
from app.integrations.payments import (
    WebhookError,
    parse_gumroad_ping,
    parse_stripe_checkout_signal,
    parse_stripe_event,
    parse_whop_event,
    verify_hmac_hex,
    verify_stripe_signature,
)
from app.models import Download, Listing, MarketingKit, Product, utcnow

router = APIRouter()


# ------------------------------------------------------------------ webhooks
@router.post("/webhooks/stripe")
async def stripe_webhook(request: Request) -> dict[str, bool]:
    s = request.app.state.settings
    body = await request.body()
    if not s.stripe_webhook_secret:
        raise HTTPException(503, "STRIPE_WEBHOOK_SECRET not set")
    try:
        verify_stripe_signature(body, request.headers.get("stripe-signature", ""), secret(s.stripe_webhook_secret))
    except WebhookError as exc:
        raise HTTPException(400, str(exc)) from exc
    event = json.loads(body)
    ev = parse_stripe_event(event)
    if ev:
        await request.app.state.office.sales.record_sale(ev)
    elif sig := parse_stripe_checkout_signal(event):
        await request.app.state.office.sales.checkout_signal(sig)
    return {"ok": True}


@router.post("/webhooks/gumroad")
async def gumroad_webhook(request: Request, key: str = "") -> dict[str, bool]:
    expected = secret(request.app.state.settings.gumroad_webhook_secret)
    if not expected or not hmac.compare_digest(expected, key):
        raise HTTPException(401, "bad key")
    form = {k: str(v) for k, v in (await request.form()).items()}
    await request.app.state.office.sales.record_sale(parse_gumroad_ping(form))
    return {"ok": True}


@router.post("/webhooks/whop")
async def whop_webhook(request: Request) -> dict[str, bool]:
    s = request.app.state.settings
    body = await request.body()
    if not s.whop_webhook_secret:
        raise HTTPException(503, "WHOP_WEBHOOK_SECRET not set")
    try:
        verify_hmac_hex(body, request.headers.get("x-whop-signature", ""), secret(s.whop_webhook_secret))
    except WebhookError as exc:
        raise HTTPException(400, str(exc)) from exc
    ev = parse_whop_event(json.loads(body))
    if ev:
        await request.app.state.office.sales.record_sale(ev)
    return {"ok": True}


# ------------------------------------------------------------------ delivery
@router.get("/d/{token}")
async def download(request: Request, token: str) -> FileResponse:
    async with request.app.state.db.session() as ses:
        d = await ses.get(Download, token)
        if not d or d.expires_at < utcnow() or d.count >= d.max_count:
            raise HTTPException(410, "This download link has expired. Reply to your purchase email for a new one.")
        p = await ses.get(Product, d.product_id)
        d.count += 1
    if not p or not os.path.exists(p.artifact_path):
        raise HTTPException(404, "file missing")
    return FileResponse(p.artifact_path, media_type="application/zip", filename=f"{p.slug}.zip")


# ------------------------------------------------------------------ landing pages
PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><meta name="description" content="{sub}">
<style>
:root{{--bg:#fff;--ink:#141826;--mut:#5B6478;--acc:#5B4BE0;--tile:#F4F5FA;--warn:#8A5A00;--warnbg:#FFF4DB}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0A0E19;--ink:#E6E9F2;--mut:#98A2BD;--acc:#9C8FFF;--tile:#141B2E;--warn:#FFD58A;--warnbg:#2A2210}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,sans-serif}}
main{{max-width:760px;margin:0 auto;padding:40px 16px}}h1{{font-size:clamp(28px,5vw,44px);line-height:1.15;margin:0 0 12px}}
.sub{{color:var(--mut);font-size:19px}}.cta{{display:inline-block;margin:20px 0 6px;padding:15px 24px;border-radius:12px;background:var(--acc);
color:#fff;text-decoration:none;font-weight:800;font-size:17px}}.cta.off{{opacity:.5;pointer-events:none}}ul{{padding-left:20px}}
.box{{background:var(--tile);border-radius:14px;padding:16px 20px;margin:20px 0}}h2{{margin-top:36px}}.q{{font-weight:700;margin-top:14px}}
.promo{{background:var(--warnbg);color:var(--warn);border-radius:12px;padding:10px 14px;font-weight:700;margin:0 0 18px}}
.loss{{border-left:4px solid #E5484D;padding:4px 0 4px 14px;font-weight:600}}.trust{{list-style:none;padding:0;display:flex;flex-wrap:wrap;gap:8px 18px;
font-size:14px;color:var(--mut)}}.trust li::before{{content:"✓ ";color:#22B07D;font-weight:800}}.small{{font-size:14px;color:var(--mut)}}
.rel a{{color:var(--acc)}}
</style></head><body><main>
{promo}<h1>{title}</h1><p class="sub">{sub}</p>
<a class="cta{off}" href="{buy}">{cta}</a><ul class="trust">{trust}</ul>
{pas}{loss}
<div class="box"><ul>{benefits}</ul></div>
<h2>How it works</h2><ol>{how}</ol>
<h2>FAQ</h2>{faq}
<a class="cta{off}" href="{buy}">{cta}</a>
<p class="small">{blurb}</p>{related}
</main></body></html>"""

_BOLD = re.compile(r"\*\*(.+?)\*\*")
BOT_UA = re.compile(r"bot|crawl|spider|preview|slurp|facebookexternalhit|curl|python-httpx|headless", re.IGNORECASE)


def _md(text: str) -> str:
    """Escape, then render **bold** (skimmable bullets)."""
    return _BOLD.sub(r"<strong>\1</strong>", html.escape(text))


async def _live_listing(ses: Any, pid: int) -> Listing | None:
    return await ses.scalar(select(Listing).where(Listing.product_id == pid, Listing.status == "live")
                            .order_by(Listing.platform.desc()))  # stripe before gumroad/whop when several


@router.get("/p/{slug}", response_class=HTMLResponse)
async def landing(request: Request, slug: str) -> HTMLResponse:
    office = request.app.state.office
    async with request.app.state.db.session() as ses:
        p = await ses.scalar(select(Product).where(Product.slug == slug))
        if not p:
            raise HTTPException(404)
        kit = await ses.scalar(select(MarketingKit).where(MarketingKit.product_id == p.id))
        listing = await _live_listing(ses, p.id)
    if listing and not BOT_UA.search(request.headers.get("user-agent", "")):
        await office.sales.track(p.id, "views")
    lp = (kit.landing if kit else {}) or {}
    e = html.escape
    promo = (await request.app.state.db.get_state("promos")).get(str(p.id))
    promo_html = ""
    if promo and promo.get("expires_at", 0) > time.time():
        until = datetime.fromtimestamp(promo["expires_at"], timezone.utc).strftime("%b %d, %H:%M UTC")
        promo_html = f"<p class='promo'>Launch offer: {promo['percent']}% off with code <code>{e(promo['code'])}</code> until {until}.</p>"
    related = await office.sales.related(p, 2)
    rel_html = ("<div class='box rel'><strong>Complete your stack</strong><ul>" + "".join(
        f"<li><a href='/p/{e(r.slug)}'>{e(r.name)}</a> — {e(r.tagline)}</li>" for r in related) + "</ul></div>") if related else ""
    pas = lp.get("pas") or {}
    return HTMLResponse(PAGE.format(
        title=e(lp.get("headline") or p.name), sub=e(lp.get("subheadline") or p.tagline), buy=f"/buy/{e(p.slug)}",
        off="" if listing else " off", cta=e(lp.get("cta") or f"Get Instant Access — ${p.price_cents / 100:.0f}"),
        trust="".join(f"<li>{e(t)}</li>" for t in lp.get("trust", [])), promo=promo_html,
        benefits="".join(f"<li>{_md(b)}</li>" for b in lp.get("benefits", [])),
        how="".join(f"<li>{_md(b)}</li>" for b in lp.get("how_it_works", [])),
        faq="".join(f"<p class='q'>{e(f['q'])}</p><p>{e(f['a'])}</p>" for f in lp.get("faq", [])),
        blurb=e(lp.get("checkout_blurb", "")), related=rel_html,
        loss=(f"<p class='loss'>{e(lp['cost_of_inaction'])}</p>" if lp.get("cost_of_inaction") else "")
        + (f"<div class='box'><p>{e(lp['value_anchor'])}</p></div>" if lp.get("value_anchor") else ""),
        pas=(f"<h2>{e(pas.get('problem', ''))}</h2><p>{e(pas.get('agitate', ''))}</p><p><strong>{e(pas.get('solve', ''))}</strong></p>"
             if pas else "")))


@router.get("/buy/{slug}")
async def buy(request: Request, slug: str) -> RedirectResponse:
    """Counts the buy click (Leo's funnel), then sends the buyer to the platform checkout."""
    async with request.app.state.db.session() as ses:
        p = await ses.scalar(select(Product).where(Product.slug == slug))
        listing = await _live_listing(ses, p.id) if p else None
    if not p or not listing or not listing.url:
        raise HTTPException(404, "not on sale")
    if not BOT_UA.search(request.headers.get("user-agent", "")):
        await request.app.state.office.sales.track(p.id, "clicks")
    return RedirectResponse(listing.url, status_code=302)


@router.get("/thanks", response_class=HTMLResponse)
async def thanks() -> HTMLResponse:
    return HTMLResponse("<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
                        "<title>Thank you</title><body style='font:18px system-ui;max-width:640px;margin:60px auto;padding:0 16px'>"
                        "<h1>Thank you!</h1><p>Your download link is on its way to your inbox (check spam just in case). "
                        "It usually arrives within a minute.</p></body>")
