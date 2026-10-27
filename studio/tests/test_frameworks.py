"""The frameworks behind each agent's vote."""

from __future__ import annotations

import json

import pytest

from app.agents import frameworks as fw
from app.agents.growth import FAQ, KitSpec, ensure_objections, fallback_kit, honesty_violations
from app.agents.sales import listing_description
from app.integrations.payments import parse_stripe_checkout_signal
from app.models import Idea, MarketingKit, Product
from app.services import n8n, packaging

NICE = "It would be nice to automate my Notion notes."
FIRE = "I'm losing $500/month because I forget to renew client contracts or bill them."
WORKAROUND = ("Currently I download the Stripe spreadsheet, upload it to Google Sheets, run a local Python script and then "
              "email each client manually one by one. Takes me 6 hours a week.")


# ------------------------------------------------------------------ Iris
def test_hair_on_fire_vs_nice_to_have():
    assert fw.assess_post(NICE).urgency == "nice_to_have"
    fire = fw.assess_post(FIRE)
    assert fire.urgency == "hair_on_fire" and fire.money_loss_usd == 500
    assert fw.assess_post("I spend 8 hours a week copying orders").urgency == "hair_on_fire"
    assert fw.assess_post("We keep losing clients because nobody follows up").churn_risk


def test_workarounds_raise_willingness_to_pay():
    w = fw.assess_post(WORKAROUND)
    assert w.workaround_steps >= 4 and w.hours_per_week == 6
    assert w.wtp > fw.assess_post("Is there a tool for invoices?").wtp


def test_opportunity_score_formula():
    a = fw.assess_post(WORKAROUND, engagement=30, intent_hits=1)
    assert a.opportunity == pytest.approx(a.importance + max(a.importance - a.satisfaction, 0))
    assert a.opportunity > fw.assess_post(NICE).opportunity


def _idea(**kw):
    base = dict(ttv_minutes=5, integrations=("Stripe", "Google Sheets", "Gmail"), search_queries=("stripe to sheets n8n",),
                ecosystems=("n8n", "stripe"), hours_saved_month=4)
    return fw.assess_idea([fw.assess_post(WORKAROUND, 40, 1).as_dict()], **{**base, **kw})


def test_idea_gates_ttv_moat_commodity():
    assert _idea().approve
    slow = _idea(ttv_minutes=20)
    assert not slow.approve and any("time-to-value" in r for r in slow.reasons)
    thin = _idea(integrations=("OpenAI",))
    assert not thin.approve and any("moat" in r for r in thin.reasons)
    wrapper = _idea(commodity=True)
    assert not wrapper.approve and wrapper.moat == 0


def test_value_based_charm_price():
    assert fw.value_price(4) in (27, 29)  # 4 h x $50 = $200/month of value -> ~$29 once
    assert all(p % 10 in (7, 9) and 19 <= p <= 149 for p in (fw.charm_price(x) for x in range(0, 400, 7)))
    v = _idea()
    assert v.no_brainer_ratio >= 3 and v.value_month_low == 200


# ------------------------------------------------------------------ Theo
def test_guard_idempotency_and_retries():
    wf = n8n.build_workflow("Demo", "demo", [{"kind": "webhook", "name": "In"}, {"kind": "http", "name": "Call"},
                                             {"kind": "respond", "name": "Ok"}], "guide")
    guard = next(n for n in wf["nodes"] if n["name"] == n8n.GUARD_NAME)
    js = guard["parameters"]["jsCode"]
    assert "$getWorkflowStaticData" in js and "WEBHOOK_SECRET" in js and "REQUIRED" in js
    assert next(n for n in wf["nodes"] if n["name"] == "Call")["retryOnFail"]
    assert "Ok" not in [n["name"] for n in wf["nodes"]]  # acknowledged on receipt instead
    assert n8n.validate_workflow(wf)["ok"]


def test_secret_scanner():
    wf = n8n.build_workflow("Leak", "leak", [{"kind": "schedule", "name": "S"},
                                             {"kind": "http", "name": "H", "params": {"url": "https://x?k=sk_live_abcdefghijklmnop"}}], "g")
    report = n8n.validate_workflow(wf)
    assert not report["ok"] and any(c["check"] == "no hard-coded secrets" and not c["ok"] for c in report["checks"])
    assert not packaging.check_python('TOKEN = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"\ndef process(p):\n    return {}\n')["ok"]


def test_service_template_is_defensive():
    files = packaging.python_service_files("Svc", "def process(payload):\n    return {}\n", "{}")
    main = files["src/main.py"]
    for needle in ("compare_digest", "MAX_BODY", "idempotency_key", "_rate_limited", "status_code=422"):
        assert needle in main
    assert "src/test_api.py" in files and "install.sh" in files


# ------------------------------------------------------------------ Maya
def _product():
    idea = Idea(topic="no_show", title="No-Show Autopilot", problem="Clients book and don't show up; revenue lost.",
                niche="Coaches (solo)", solution_type="n8n", mvp="Webhook -> reminders", price_cents=4700,
                assessment={"hours_saved_month": 6, "moat": 3, "ttv_minutes": 5})
    p = Product(id=1, name="No-Show Autopilot", slug="nsa", tagline="Cut no-shows", solution_type="n8n", price_cents=4700)
    p.idea = idea
    return p


def test_maya_kit_uses_only_real_numbers_and_answers_objections():
    p = _product()
    kit = fallback_kit(p)
    assert "$300" in kit.landing.value_anchor and "$1,200" in kit.landing.value_anchor and "$47" in kit.landing.value_anchor
    qs = " ".join(f.q for f in kit.landing.faq)
    assert "code" in qs and "How long" in qs and "n8n" in qs
    assert [e.day for e in kit.emails] == [0, 1, 3, 7]
    assert not honesty_violations(kit)
    faq = ensure_objections([FAQ(q="Is it good?", a="Yes")], p, {"ttv": 5})
    assert len(faq) >= 4


def test_honesty_gate_catches_fake_social_proof():
    data = json.loads(fallback_kit(_product()).model_dump_json())
    data["landing"]["subheadline"] = "Trusted by 2,000+ happy customers. ★★★★★"
    assert honesty_violations(KitSpec.model_validate(data))


# ------------------------------------------------------------------ Leo
def test_listing_formatted_per_platform():
    p = _product()
    kit = MarketingKit(product_id=1, landing=fallback_kit(p).landing.model_dump())
    stripe, whop, gum = (listing_description(p, kit, x) for x in ("stripe", "whop", "gumroad"))
    assert len(stripe) <= 500 and "<" not in stripe
    assert "•" in whop and gum.startswith("<p>") and "<ul>" in gum


def test_abandoned_checkout_requires_consent_flag():
    sig = parse_stripe_checkout_signal({"type": "checkout.session.expired", "data": {"object": {
        "id": "cs_1", "payment_link": "plink_1", "customer_details": {"email": "a@b.co"}, "consent": {"promotions": "opt_in"}}}})
    assert sig and sig.kind == "abandoned" and sig.consent
    no = parse_stripe_checkout_signal({"type": "checkout.session.expired", "data": {"object": {"id": "cs_2"}}})
    assert no and not no.consent
    failed = parse_stripe_checkout_signal({"type": "payment_intent.payment_failed", "data": {"object": {
        "id": "pi_1", "last_payment_error": {"code": "card_declined"}}}})
    assert failed and failed.kind == "failed" and failed.reason == "card_declined"


# ------------------------------------------------------------------ level 2
def test_iris_multi_tool_workaround_and_payer_habit():
    a = fw.assess_post("I export Stripe payouts to a CSV, paste into Google Sheets, then copy totals into QuickBooks and email my "
                       "accountant. I pay for Zapier already.")
    assert a.multi_tool_workaround and a.wtp == 10 and a.paying_signal > 0
    assert fw.assess_post("Looking for a free alternative, can't afford another tool").paying_signal < 0


def test_iris_scope_friction_and_payer_gates():
    assert not _idea(scope_steps=15).approve
    heavy = _idea(env_vars=6, credentials=3, needs_server=True, technical_audience=False)
    assert heavy.friction > heavy.friction_tolerance and any("entry barrier" in r for r in heavy.reasons)
    stingy = fw.assess_idea([{**fw.assess_post("free alternative please, can't afford it").as_dict()}], ttv_minutes=5,
                            integrations=("A", "B", "C"), search_queries=("q",), ecosystems=("n8n",), hours_saved_month=6,
                            payer_prior=-0.8)
    assert any("avoids paying" in r for r in stingy.reasons)
    assert fw.value_price(4) == 29 and fw.value_price(20) == 149


def test_theo_self_healing_branch_and_layout():
    wf = n8n.build_workflow("Demo", "demo", [{"kind": "webhook", "name": "In"}, {"kind": "http", "name": "Call"},
                                             {"kind": "email", "name": "Mail"}], "guide")
    assert wf["connections"]["Call"]["main"][1][0]["node"] == n8n.ERROR_SET
    assert next(n for n in wf["nodes"] if n["name"] == "Call")["onError"] == "continueErrorOutput"
    assert any(n["type"] == "n8n-nodes-base.stopAndError" for n in wf["nodes"])
    assert n8n.validate_workflow(wf)["ok"]
    files = packaging.python_service_files("Svc", 'import os\nKEY = os.environ.get("MY_KEY", "")\ndef process(p):\n    return {}\n', "{}", "svc")
    assert set(files) >= {"src/main.py", "src/logic.py", "src/Dockerfile", "install.sh"}
    assert "--limit-concurrency" in files["src/Dockerfile"] and "MALLOC_ARENA_MAX" in files["src/Dockerfile"]
    assert packaging.python_env_vars(files["src/main.py"], files["src/logic.py"]) == ["MY_KEY", "RATE_LIMIT_PER_MIN", "WEBHOOK_SECRET"]
    assert "last_error" in files["src/main.py"]


def test_theo_readme_quickstart_gate():
    from app.agents.builder import check_ttv, readme

    text = readme("X", "t", "p", "n8n", "g", [("A", "a"), ("B", "b")])
    assert "## 3-Step Quickstart" in text and check_ttv(text, [("A", "a")])["ok"]
    assert not check_ttv(text, [(str(i), "") for i in range(7)])["ok"]


def test_maya_polish_loss_aversion_cta_trust_us_english():
    from app.agents.growth import polish_kit, us_english

    p = _product()
    kit = fallback_kit(p)
    kit.landing.cta = "Buy now"
    kit.landing.benefits[0] = "Optimise your colour scheme: and more"
    out = polish_kit(kit, p, {"hours": 6, "value_month_low": 300, "value_month_high": 900, "price": 47, "ttv": 5}, ["stripe"], 0)
    lp = out.landing
    assert lp.cta == "Get Instant Access — $47"
    from app.agents.growth import benefit_cta
    assert benefit_cta("Get it for $49", "$49") == benefit_cta("Buy now - $49", "$49") == "Get Instant Access — $49"
    assert benefit_cta("Automate My Workflow in 2 Mins", "$49") == "Automate My Workflow in 2 Mins"
    assert "$300" in lp.cost_of_inaction and "every single day" in lp.cost_of_inaction
    assert lp.benefits[0].startswith("**Optimize your colour scheme".replace("colour", "color"))
    assert any("Stripe" in t for t in lp.trust) and not any("guarantee" in t for t in lp.trust)
    assert any(f.q == "Is there a guarantee?" for f in lp.faq)
    assert not honesty_violations(out)
    with_refund = polish_kit(fallback_kit(p), p, {"hours": 6, "value_month_low": 300, "value_month_high": 900, "price": 47, "ttv": 5},
                             ["stripe"], 14)
    assert any("14-day money-back" in t for t in with_refund.landing.trust)
    assert us_english("We realised the behaviour") == "We realized the behavior"


def test_leo_dispute_parsing():
    from app.integrations.payments import parse_stripe_event

    ev = parse_stripe_event({"type": "charge.dispute.created", "data": {"object": {"id": "dp_1", "charge": "ch_1",
                                                                                   "payment_intent": "pi_9", "amount": 4700}}})
    assert ev and ev.refunded and ev.disputed and ev.payment_ref == "pi_9"
    paid = parse_stripe_event({"type": "checkout.session.completed", "data": {"object": {
        "id": "cs_9", "payment_status": "paid", "amount_total": 4700, "payment_intent": "pi_9"}}})
    assert paid and paid.payment_ref == "pi_9"
