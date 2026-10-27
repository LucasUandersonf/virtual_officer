"""End-to-end tests of the office, offline (no LLM key, no network)."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import zipfile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.agents.office import Office
from app.config import Settings
from app.integrations.payments import SaleEvent, WebhookError, parse_stripe_event, verify_stripe_signature
from app.main import build_context, create_app
from app.models import CostEntry, Download, Idea, Listing, Product, Sale
from app.services import finance, n8n, packaging


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, database_url=f"sqlite+aiosqlite:///{tmp_path}/t.db", artifacts_dir=str(tmp_path / "a"),
                    research_offline=True, theatre_delay_sec=0, max_builds_per_day=3, admin_token="secret-token",
                    stripe_webhook_secret="whsec_test")


@pytest.fixture
async def office(settings):
    ctx = build_context(settings)
    await ctx.db.init()
    yield Office(ctx)
    await ctx.http.aclose()
    await ctx.db.close()


async def test_pipeline_runs_offline(office):
    done = await office.tick()
    assert done["ideas"] == 3 and done.get("weekly")
    async with office.ctx.db.session() as s:
        ideas = (await s.scalars(select(Idea))).all()
        products = (await s.scalars(select(Product))).unique().all()
        costs = (await s.scalars(select(CostEntry))).all()
    kept = [i for i in ideas if i.provider == "playbook"]
    discarded = [i for i in ideas if i.provider == "frameworks"]
    assert len(kept) == 3 and discarded and all(i.status == "rejected" and "Discarded by Iris" in i.rationale for i in discarded)
    assert products and all(p.qa["ok"] for p in products)
    assert all(p.status == "ready" for p in products)  # AUTOPUBLISH off: waits for the CEO
    assert {c.category for c in costs} == {"railway", "other", "llm"}
    with zipfile.ZipFile(products[0].artifact_path) as z:
        names = z.namelist()
    assert any(n.endswith("README.md") for n in names) and any(n.endswith(".env.example") for n in names)


async def test_cost_accrual_is_idempotent(office, settings):
    first = await finance.accrue_costs(office.ctx.db, settings)
    assert first > 0
    assert await finance.accrue_costs(office.ctx.db, settings) == 0
    p = await finance.pnl(office.ctx.db, settings)
    assert p["profit_usd"] < 0 and p["costs_usd"] == pytest.approx(first / 100)


async def test_sale_is_recorded_once_and_delivered(office):
    await office.tick()
    async with office.ctx.db.session() as s:
        p = (await s.scalars(select(Product))).unique().first()
        s.add(Listing(product_id=p.id, platform="stripe", external_id="plink_1", url="https://buy.stripe.com/x"))
    ev = SaleEvent("stripe", "cs_1", 4900, "buyer@example.com", "plink_1")
    sale = await office.sales.record_sale(ev)
    again = await office.sales.record_sale(ev)
    assert sale and again and sale.id == again.id
    async with office.ctx.db.session() as s:
        assert len((await s.scalars(select(Sale))).all()) == 1
        assert (await s.scalars(select(Download))).first() is not None
    assert sale.fee_cents == round(4900 * (0.029 + 0.7 * 0.015) + 30)  # domestic + expected share of foreign cards
    await office.sales.record_sale(SaleEvent("stripe", "cs_1", 4900, "", "", refunded=True))
    assert (await finance.pnl(office.ctx.db, office.ctx.settings))["sales"] == 0


def test_stripe_signature():
    body = b'{"type":"checkout.session.completed"}'
    ts = int(time.time())
    sig = hmac.new(b"whsec_x", f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    verify_stripe_signature(body, f"t={ts},v1={sig}", "whsec_x")
    with pytest.raises(WebhookError):
        verify_stripe_signature(body, f"t={ts},v1={'0' * 64}", "whsec_x")
    with pytest.raises(WebhookError):
        verify_stripe_signature(body, f"t={ts - 3600},v1={sig}", "whsec_x")


def test_parse_stripe_event():
    ev = parse_stripe_event({"type": "checkout.session.completed", "data": {"object": {
        "id": "cs_9", "payment_status": "paid", "amount_total": 3900, "customer_details": {"email": "a@b.co"}, "payment_link": "plink_9"}}})
    assert ev and ev.gross_cents == 3900 and ev.listing_ref == "plink_9"
    assert parse_stripe_event({"type": "invoice.paid", "data": {"object": {}}}) is None


def test_workflow_builder_and_validator():
    wf = n8n.build_workflow("Demo", "demo", [{"kind": "webhook", "name": "In"}, {"kind": "set", "name": "Map", "params": {"fields": ["email"]}},
                                             {"kind": "email", "name": "Mail"}, {"kind": "bogus", "name": "X"}], "guide")
    report = n8n.validate_workflow(wf)
    assert report["ok"], report
    names = [n["name"] for n in wf["nodes"]]
    assert "X" not in names and wf["connections"]["In"]["main"][0][0]["node"] == n8n.GUARD_NAME
    assert wf["connections"][n8n.GUARD_NAME]["main"][0][0]["node"] == "Map"
    mail = next(n for n in wf["nodes"] if n["name"] == "Mail")
    assert "$('Map').item.json.email" in mail["parameters"]["toEmail"]
    assert "SMTP_FROM" in n8n.env_vars(wf)


def test_python_gate_blocks_dangerous_code():
    assert packaging.check_python("def process(payload):\n    return {}\n")["ok"]
    assert not packaging.check_python("import subprocess\ndef process(p):\n    return {}\n")["ok"]
    assert not packaging.check_python("def process(p):\n    return eval(p['x'])\n")["ok"]
    assert not packaging.check_python("def handler(p):\n    return {}\n")["ok"]
    assert not packaging.check_python("def process(:\n")["ok"]


def test_api_and_webhooks(settings):
    app = create_app(settings, run_agents=False)
    with TestClient(app) as c:
        assert c.get("/health").json()["ok"]
        s = c.get("/api/dashboard/summary").json()
        assert set(s["agents"]) == {"product", "builder", "growth", "sales", "cfo"}
        assert c.get("/api/sectors").status_code == 200
        assert c.get("/api/dashboard/sales/daily?days=7").json()[-1]["day"]
        assert c.post("/api/run/tick").status_code == 401
        assert c.post("/api/run/tick", headers={"X-Admin-Token": "secret-token"}).json() == {"queued": "tick"}
        assert c.post("/webhooks/stripe", content=b"{}", headers={"stripe-signature": "t=1,v1=bad"}).status_code == 400
        body = json.dumps({"type": "checkout.session.completed", "data": {"object": {"id": "cs_w", "payment_status": "paid",
                                                                                     "amount_total": 2900}}}).encode()
        ts = int(time.time())
        sig = hmac.new(b"whsec_test", f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
        assert c.post("/webhooks/stripe", content=body, headers={"stripe-signature": f"t={ts},v1={sig}"}).json() == {"ok": True}
        assert c.get("/api/dashboard/summary").json()["all_time"]["sales"] == 1
        assert c.get("/d/nope").status_code == 410
        assert "Studio Floor" in c.get("/").text


async def test_llm_router_validates_and_falls_back(settings):
    import httpx
    from pydantic import BaseModel

    from app.llm.router import LLMRouter, Provider

    class Out(BaseModel):
        answer: str

    replies = iter(['```json\n{"answer": "from groq"}\n```', "not json at all"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": next(replies)}}], "usage": {"total_tokens": 12}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    router = LLMRouter(settings, providers=[Provider("groq", "https://x", "k", {"smart": "m", "fast": "m"})], client=client)
    obj, provider = await router.json("t", "sys", "user", Out, lambda: Out(answer="fallback"))
    assert (obj.answer, provider) == ("from groq", "groq")
    obj, provider = await router.json("t", "sys", "user", Out, lambda: Out(answer="fallback"))
    assert (obj.answer, provider) == ("fallback", "playbook")
    await client.aclose()


async def test_cfo_audit_flags_stale_products(office):
    from datetime import timedelta

    from app.models import utcnow

    await office.tick()
    async with office.ctx.db.session() as s:
        p = (await s.scalars(select(Product))).unique().first()
        p.status, p.published_at = "published", utcnow() - timedelta(days=25)
    flagged = await office.cfo.audit(office.growth, office.sales)
    assert [f["health"] for f in flagged] == ["reposition"]
    assert (await office.ctx.db.get_state("portfolio"))["repositioned"] == [p.id]
    econ = await office.cfo.economics(Idea(score=70, price_cents=4700, assessment={"urgency": "hair_on_fire"}))
    assert econ.contribution_margin > 0.9 and econ.ok


async def test_abandoned_checkout_gets_one_recovery_email_only_with_consent(office, monkeypatch):
    from app.integrations.payments import CheckoutSignal

    sent = []

    async def fake_email(to, subject, html):
        sent.append(to)
        return True

    monkeypatch.setattr(office.ctx.notify, "email", fake_email)
    await office.tick()
    async with office.ctx.db.session() as s:
        p = (await s.scalars(select(Product))).unique().first()
        s.add(Listing(product_id=p.id, platform="stripe", external_id="plink_r", url="https://buy.stripe.com/r"))
    assert await office.sales.checkout_signal(CheckoutSignal("abandoned", "cs_a", "x@y.co", "plink_r", False)) == "no consent: not contacted"
    assert await office.sales.checkout_signal(CheckoutSignal("abandoned", "cs_b", "x@y.co", "plink_r", True)) == "recovery email sent"
    assert await office.sales.checkout_signal(CheckoutSignal("abandoned", "cs_c", "x@y.co", "plink_r", True)) == "logged"
    assert sent == ["x@y.co"]


async def test_refund_and_dispute_revoke_downloads(office):
    await office.tick()
    async with office.ctx.db.session() as s:
        p = (await s.scalars(select(Product))).unique().first()
        s.add(Listing(product_id=p.id, platform="stripe", external_id="plink_d", url="https://buy.stripe.com/d"))
    sale = await office.sales.record_sale(SaleEvent("stripe", "cs_d", 4700, "b@c.co", "plink_d", payment_ref="pi_d"))
    await office.sales.record_sale(SaleEvent("stripe", "pi_d", 4700, "", "", refunded=True, payment_ref="pi_d", disputed=True))
    async with office.ctx.db.session() as s:
        row = await s.get(Sale, sale.id)
        d = (await s.scalars(select(Download).where(Download.sale_id == sale.id))).first()
    assert row.refunded and row.disputed and d.count == d.max_count


async def test_funnel_tracking_and_elasticity(office):
    from app.models import FunnelDay

    await office.tick()
    async with office.ctx.db.session() as s:
        p = (await s.scalars(select(Product))).unique().first()
        p.status = "published"
        s.add(Listing(product_id=p.id, platform="stripe", external_id="plink_f", url="https://buy.stripe.com/f"))
        from app.models import utcnow
        s.add(FunnelDay(product_id=p.id, day=utcnow().date(), views=300, clicks=40))
    start_price = p.price_cents // 100
    actions = await office.sales.funnel_review(office.growth)
    assert actions and "price friction" in actions[0]["diagnosis"]
    async with office.ctx.db.session() as s:
        assert (await s.get(Product, p.id)).price_cents // 100 < start_price
    again = await office.sales.funnel_review(office.growth)
    assert "skipped" in again[0]["action"]  # one price change per 7 days


async def test_cfo_runway_break_even_and_ltv(office, settings):
    from app.services import finance

    await office.tick()
    g = await finance.goals(office.ctx.db, settings)
    assert g["break_even"]["sales_needed"] >= 1 and not g["break_even"]["from_real_sales"]
    assert g["runway"]["monthly_burn_usd"] > 0
    lc = await finance.ltv_cac(office.ctx.db, settings)
    assert lc["organic_only"] and lc["ltv_cac"] is None
    st = await office.cfo.infra_control()
    assert not st["measured"] and not st["austerity"] and office.ctx.llm.budget_factor == 1.0


async def test_agents_self_configure_from_stripe(tmp_path):
    """Leo creates the webhook (and keeps its secret) + turns on wallets; Nina reads country, currency, cash, tax."""
    import httpx

    from app.agents.office import Office
    from app.services import finance

    calls = []

    def stripe_api(req: httpx.Request) -> httpx.Response:
        calls.append((req.method, req.url.path))
        path = req.url.path
        if req.url.host == "open.er-api.com":
            return httpx.Response(200, json={"rates": {"BRL": 5.0}})
        if path == "/v1/account":
            return httpx.Response(200, json={"country": "BR", "default_currency": "brl"})
        if path == "/v1/balance":
            return httpx.Response(200, json={"available": [{"amount": 50000, "currency": "brl"}], "pending": [{"amount": 10000, "currency": "brl"}]})
        if path == "/v1/webhook_endpoints" and req.method == "GET":
            return httpx.Response(200, json={"data": []})
        if path == "/v1/webhook_endpoints" and req.method == "POST":
            return httpx.Response(200, json={"id": "we_1", "secret": "whsec_created_by_leo"})
        if path == "/v1/payment_method_configurations" and req.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "pmc_1", "is_default": True, "apple_pay": {"display_preference": {"value": "off"}}}]})
        if path.startswith("/v1/payment_method_configurations/"):
            return httpx.Response(200, json={"id": "pmc_1"})
        return httpx.Response(404, json={})

    s = Settings(_env_file=None, database_url=f"sqlite+aiosqlite:///{tmp_path}/a.db", artifacts_dir=str(tmp_path / "a"),
                 research_offline=True, theatre_delay_sec=0, stripe_secret_key="sk_test_x", public_base_url="https://studio.example.com")
    ctx = build_context(s)
    await ctx.http.aclose()
    ctx.http = httpx.AsyncClient(transport=httpx.MockTransport(stripe_api))
    await ctx.db.init()
    o = Office(ctx)
    rep = await o.sales.self_configure()
    assert rep["stripe_webhook"].startswith("created") and "turned on" in rep["wallets"]
    assert (await ctx.db.get_state("stripe_webhook"))["secret"] == "whsec_created_by_leo"
    await o.cfo.self_configure()
    fp = await finance.fin_params(ctx.db, s)
    assert fp["payout_currency"] == "brl" and fp["country"] == "BR"
    assert fp["cash_usd"] == 120.0 and fp["cash_source"].startswith("Stripe")  # R$ 600 / 5
    assert fp["tax_rate"] == 0.06 and "provision" in fp["tax_source"]
    explicit = Settings(_env_file=None, database_url=s.database_url, tax_rate_pct=0.11, payout_currency="usd")
    fp2 = await finance.fin_params(ctx.db, explicit)
    assert fp2["tax_rate"] == 0.11 and fp2["payout_currency"] == "usd"  # explicit settings always win
    await ctx.http.aclose()
    await ctx.db.close()


def test_railway_domain_becomes_public_url(monkeypatch):
    monkeypatch.setenv("RAILWAY_PUBLIC_DOMAIN", "studio-production.up.railway.app")
    assert Settings(_env_file=None).public_base_url == "https://studio-production.up.railway.app"
    assert Settings(_env_file=None, public_base_url="https://my.shop").public_base_url == "https://my.shop"
