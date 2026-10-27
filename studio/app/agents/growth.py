"""Agent 3 — Maya, Growth Hacker & Copywriter: direct-response copy in US English.

* PAS (Problem-Agitate-Solve) built on the pain Iris measured.
* Value anchoring with real numbers only: hours saved x $50-150/h, and what a freelancer would charge to
  build the integration, against the one-time price.
* Objection removal at checkout: the FAQ always answers "Do I need to code?", "How long does setup take?",
  "Does it work on my n8n plan?" (n8n products) and "What if it doesn't work for me?".
* A 4-email onboarding sequence (day 0, 1, 3, 7) aimed at the first win and low refunds.
* Honesty gate: copy with invented testimonials, customer counts, ratings or guarantees is thrown away
  and replaced by the deterministic kit — a fake claim costs more in refunds and bans than it earns.
* Loss aversion (Kahneman): the cost of waiting, per month and per day, from the measured hours.
* Skimmable copy: short bullets with a bold lead, headline <= 80 chars, for mobile readers.
* Trust without fake proof: real signals only (secure checkout, instant download, runs on the buyer's
  own stack, built-in auth/idempotency) and a guarantee only when REFUND_DAYS > 0.
* Benefit-driven CTA ("Get Instant Access — $47", never a bare "Buy now"); native US English.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select

from app.agents import frameworks, playbook
from app.agents.base import Agent
from app.models import MarketingKit, Product


class FAQ(BaseModel):
    q: str
    a: str


class PAS(BaseModel):
    problem: str
    agitate: str
    solve: str


class Landing(BaseModel):
    headline: str = Field(max_length=120)
    subheadline: str
    pas: PAS
    value_anchor: str
    benefits: list[str] = Field(min_length=3, max_length=6)
    how_it_works: list[str] = Field(min_length=3, max_length=5)
    faq: list[FAQ] = Field(min_length=3, max_length=8)
    cta: str = Field(max_length=40)
    checkout_blurb: str
    cost_of_inaction: str = ""  # loss aversion; recomputed from real numbers in polish_kit
    trust: list[str] = Field(default_factory=list)  # real trust signals only; set in polish_kit


class Email(BaseModel):
    day: int
    subject: str
    body: str


class RedditPost(BaseModel):
    subreddit: str
    title: str
    body: str


class Social(BaseModel):
    x: list[str] = Field(min_length=2, max_length=5)
    linkedin: list[str] = Field(min_length=1, max_length=3)
    reddit: list[RedditPost] = Field(min_length=1, max_length=3)


class KitSpec(BaseModel):
    landing: Landing
    emails: list[Email] = Field(min_length=4, max_length=6)
    social: Social


# Claims nobody at the studio can back up: fake social proof, ratings, guarantees, "#1".
FAKE_CLAIMS = re.compile(r"(testimonial|trusted by \d|\b\d[\d,]*\+?\s+(happy\s+)?(customers|users|businesses|founders|clients)\b|"
                         r"★|5[- ]star|rated \d|#1\b|best[- ]selling|as seen on|"
                         # affirmative promises only: asking "Is there a guarantee?" and answering honestly is fine
                         r"(money[- ]back|satisfaction|100\s?%|ironclad|risk[- ]free)\s+guarantee|guaranteed\s+(results?|roi|income|sales|profits?|to))",
                         re.IGNORECASE)


def honesty_violations(kit: KitSpec, allow_guarantee: bool = False) -> list[str]:
    hits = {m.group(0) for m in FAKE_CLAIMS.finditer(kit.model_dump_json())}
    if allow_guarantee:
        hits = {h for h in hits if "money" not in h.lower() and "back" not in h.lower()}
    return sorted(hits)


# ------------------------------------------------------------------ polish: deterministic conversion rules
BRITISH = {"optimise": "optimize", "organise": "organize", "colour": "color", "behaviour": "behavior", "favour": "favor",
           "analyse": "analyze", "cancelled": "canceled", "centre": "center", "whilst": "while", "programme": "program",
           "enquiry": "inquiry", "catalogue": "catalog", "prioritise": "prioritize", "realise": "realize", "recognise": "recognize",
           "minimise": "minimize", "maximise": "maximize", "customise": "customize", "utilise": "utilize", "personalise": "personalize",
           "automatise": "automate", "learnt": "learned", "travelling": "traveling", "labour": "labor"}
_BRIT = re.compile(r"\b(" + "|".join(BRITISH) + r")(s|d|ing|r)?\b", re.IGNORECASE)
GENERIC_CTA = {"buy", "buy now", "purchase", "order", "order now", "checkout", "add to cart", "get it", "submit", "pay", "pay now"}


def us_english(text: str) -> str:
    def fix(m: re.Match[str]) -> str:
        base = BRITISH[m.group(1).lower()]
        out = base + (m.group(2) or "")
        return out[0].upper() + out[1:] if m.group(1)[0].isupper() else out
    return _BRIT.sub(fix, text)


def skim(bullet: str, max_words: int = 16) -> str:
    """Bold lead + short line: '**Get back 6 hours a month** — ...'."""
    b = bullet.strip()
    words = b.split()
    if len(words) > max_words:
        b = " ".join(words[:max_words]).rstrip(",;:") + "…"
    if "**" in b:
        return b
    for sep in (" — ", ": ", " - "):
        if sep in b:
            lead, rest = b.split(sep, 1)
            return f"**{lead.strip()}** — {rest.strip()}"
    w = b.split()
    return f"**{' '.join(w[:4])}** {' '.join(w[4:])}".strip()


def benefit_cta(cta: str, price: str) -> str:
    core = re.sub(r"\s*(?:for|at|only|—|–|-)?\s*\$\d+.*$", "", cta).strip().lower()
    if not core or core in GENERIC_CTA or len(cta) > 40:
        return f"Get Instant Access — {price}"
    return cta


def polish_kit(kit: KitSpec, p: Product, n: dict[str, float], platforms: list[str], refund_days: int) -> KitSpec:
    lp = kit.landing
    price = f"${n['price']:.0f}"
    lp.headline = lp.headline if len(lp.headline) <= 80 else lp.headline[:80].rsplit(" ", 1)[0] + "…"
    lp.benefits = [skim(b) for b in lp.benefits]
    lp.cta = benefit_cta(lp.cta, price)
    lp.cost_of_inaction = (f"Every month you wait costs about {n['hours']:.0f} hours — ${n['value_month_low']:,.0f} to "
                           f"${n['value_month_high']:,.0f} of your time, roughly ${n['value_month_low'] / 30:,.0f} every single day.")
    checkout = {"stripe": "Secure checkout by Stripe (cards, Apple Pay, Google Pay)", "gumroad": "Secure checkout by Gumroad",
                "whop": "Secure checkout by Whop"}
    trust = [checkout[x] for x in platforms if x in checkout][:1] or ["Secure checkout"]
    trust += ["Instant download right after payment",
              "Runs on your own " + ("n8n" if p.solution_type == "n8n" else "server") + ": your data never passes through us",
              "Webhook authentication and duplicate protection built in"]
    if refund_days > 0:
        trust.append(f"{refund_days}-day money-back guarantee: reply to your receipt")
    lp.trust = trust
    g = (f"Yes: {refund_days}-day money-back guarantee. Reply to your receipt and you get a full refund." if refund_days > 0 else
         "We don't sell promises: if it doesn't run in your setup, reply to your receipt and we'll get it working with you.")
    if not any("guarantee" in f.q.lower() or "refund" in f.q.lower() for f in lp.faq):
        lp.faq = (lp.faq + [FAQ(q="Is there a guarantee?", a=g)])[-8:] if len(lp.faq) >= 8 else lp.faq + [FAQ(q="Is there a guarantee?", a=g)]
    data = json.loads(kit.model_dump_json())

    def walk(v: Any) -> Any:
        if isinstance(v, str):
            return us_english(v)
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        return v
    return KitSpec.model_validate(walk(data))


def numbers(p: Product) -> dict[str, float]:
    """Value-anchor numbers from Iris's assessment (falls back to the playbook for older ideas)."""
    a = (p.idea.assessment if p.idea else None) or {}
    t = playbook.BY_KEY.get(p.idea.topic if p.idea else "")
    hours = float(a.get("hours_saved_month") or (t.hours_saved_month if t else 4))
    anchor = frameworks.value_anchor(hours, int(a.get("moat") or (len(t.integrations) if t else 2)))
    return {"hours": hours, **anchor, "price": p.price_cents / 100, "ttv": float(a.get("ttv_minutes") or 5)}


def required_faq(p: Product, n: dict[str, float]) -> list[FAQ]:
    n8n = p.solution_type == "n8n"
    out = [
        FAQ(q="Do I need to code?", a="No. Import the workflow and fill in a few variables; every step is in the README." if n8n else
            "No coding: set two environment variables and run one Docker command (or deploy the folder on Railway/Render)."),
        FAQ(q="How long does setup take?", a=f"About {n['ttv']:.0f} minutes, following the 5-step install guide."),
        FAQ(q="What if it doesn't work for me?", a="Reply to your purchase email with what you see; we help you get it running."),
    ]
    if n8n:
        out.insert(2, FAQ(q="Does it work on the n8n free / self-hosted plan?",
                          a="Yes: it uses only built-in n8n nodes and runs on n8n Cloud or any self-hosted n8n v1.30+."))
    return out


def ensure_objections(faq: list[FAQ], p: Product, n: dict[str, float]) -> list[FAQ]:
    """The checkout objections are always answered, whatever the LLM wrote."""
    have = " ".join(f.q.lower() for f in faq)
    keys = {"Do I need to code?": "code", "How long does setup take?": "how long", "What if it doesn't work for me?": "doesn't work",
            "Does it work on the n8n free / self-hosted plan?": "plan"}
    extra = [f for f in required_faq(p, n) if keys[f.q] not in have]
    return (extra + faq)[:8]


def fallback_kit(p: Product) -> KitSpec:
    i = p.idea
    n = numbers(p)
    price = f"${n['price']:.0f}"
    kind = "n8n workflow" if p.solution_type == "n8n" else "ready-to-deploy FastAPI service"
    a_kind = f"an {kind}" if p.solution_type == "n8n" else f"a {kind}"
    pain = i.problem.split(";")[0].rstrip(".")
    who = i.niche.split("(")[0].strip()
    anchor = (f"Doing this by hand costs you about {n['hours']:.0f} hours a month — ${n['value_month_low']:,.0f} to "
              f"${n['value_month_high']:,.0f} of your time. Hiring a developer to build the same integration runs about "
              f"${n['dev_build_cost']:,.0f}. This one is {price}, once.")
    return KitSpec(
        landing=Landing(
            headline=p.tagline or p.name,
            subheadline=f"{a_kind[0].upper()}{a_kind[1:]} for {who.lower()}. Running in about {n['ttv']:.0f} minutes, then it works on its own.",
            pas=PAS(problem=f"{pain}.",
                    agitate=f"Every week it repeats: about {n['hours']:.0f} hours a month of manual work, and the money or clients "
                            "that slip through while you are busy doing it.",
                    solve=f"{p.name} handles it automatically: {i.mvp}"),
            value_anchor=anchor,
            benefits=[f"Get back ~{n['hours']:.0f} hours every month.", "Runs 24/7 on your own stack: no new SaaS subscription.",
                      "Duplicate or fake webhook calls are filtered out, so nothing is sent twice.",
                      "Every setting is a plain variable, documented step by step.", "One-time price, lifetime use for your business."],
            how_it_works=["Download the package and open the README.", f"Import the {kind} and fill in .env.example.",
                          "Run the test, switch it on and forget about it."],
            faq=required_faq(p, n) + [FAQ(q="What do I receive?",
                                          a="A .zip with the solution, an English install guide, .env.example and scripts.")],
            cta=f"Get Instant Access — {price}", checkout_blurb=f"{p.name}: instant download, 5-step install guide included."),
        emails=[
            Email(day=0, subject=f"Your {p.name} is ready (5-minute setup inside)",
                  body="Here is your download. Open README.md and follow the 5 steps: import, credentials, variables, test, activate."),
            Email(day=1, subject="Your first win in 5 minutes",
                  body="Run the included test once. When you see the first result, switch it on: from here it works without you."),
            Email(day=3, subject="Stuck? The 2 settings that fix most setups",
                  body="Check that every variable from .env.example is set and that the credentials in each node are selected. "
                       "Still stuck? Reply with a screenshot and we'll help."),
            Email(day=7, subject=f"Your first week: ~{n['hours'] / 4:.0f} hours back",
                  body="That's roughly the time this saves each week. Tell us the next manual task you'd like gone; we build what buyers ask for."),
        ],
        social=Social(
            x=[f"Technical breakdown 🧵 {pain}. Here's the exact flow that fixes it:\n\n{i.mvp}\n\nNo new SaaS, runs on your own stack.",
               f"Solopreneur math: {n['hours']:.0f} h/month × $50/h = ${n['value_month_low']:,.0f}/month of your time. "
               "Automate the boring money tasks first.",
               f"We packaged the flow above as {a_kind}: {p.name}. {p.tagline}"],
            linkedin=[f"Mini case study: {who.lower()} and one recurring problem.\n\nThe problem: {i.problem}\n"
                      f"The cost: ~{n['hours']:.0f} hours a month, ${n['value_month_low']:,.0f}-{n['value_month_high']:,.0f} of billable time.\n"
                      f"The fix: {i.mvp}\n\nIf you want the ready-made version, it's {p.name}."],
            reddit=[RedditPost(subreddit="r/n8n" if p.solution_type == "n8n" else "r/SaaS",
                               title=f"How I automated: {i.title}",
                               body=f"{i.problem}\n\nThe flow, step by step: {i.mvp}\n\nHappy to answer questions in the comments.")]),
    )


class Growth(Agent):
    key = "growth"

    async def create_kit(self, product: Product, angle: str = "") -> MarketingKit:
        await self.status("working", f"Writing copy for {product.name}" + (" (new angle)" if angle else ""))
        i = product.idea
        n = numbers(product)
        spec, provider = await self.ctx.llm.json(
            self.key, self.persona.system,
            f"Product: {product.name}\nTagline: {product.tagline}\nType: {product.solution_type}\nPrice: ${n['price']:.0f} one-time\n"
            f"Buyer: {i.niche}\nProblem (from Iris's research): {i.problem}\nWhat it does: {i.mvp}\n"
            f"Numbers you may use (and no others): saves ~{n['hours']:.0f} h/month = ${n['value_month_low']:,.0f}-"
            f"${n['value_month_high']:,.0f} of the buyer's time at $50-150/h; a freelancer would charge ~${n['dev_build_cost']:,.0f} "
            f"to build it; setup takes ~{n['ttv']:.0f} minutes.\n{angle}\n"
            "Write: landing page with PAS and a value anchor + checkout blurb; FAQ that kills checkout objections; a 4-email "
            "onboarding sequence (day 0 delivery + 5-minute quick start, day 1 first win, day 3 troubleshooting, day 7 value "
            "recap) built to prevent refunds; value-first native posts for X, LinkedIn and Reddit (Reddit teaches the method, "
            "no hard sell). No testimonials, user counts, ratings or guarantees.",
            KitSpec, lambda: fallback_kit(product), max_tokens=3500, temperature=0.7)
        refund_days = self.ctx.settings.refund_days
        bad = honesty_violations(spec, allow_guarantee=refund_days > 0)
        if bad and provider != "playbook":
            await self.log("honesty", f"Rejected LLM copy for {product.name}: unbacked claims {bad}", product_id=product.id)
            spec, provider = fallback_kit(product), "playbook"
        spec.landing.faq = ensure_objections(spec.landing.faq, product, n)
        spec = polish_kit(spec, product, n, self.ctx.settings.platforms(), refund_days)
        async with self.ctx.db.session() as ses:
            kit = await ses.scalar(select(MarketingKit).where(MarketingKit.product_id == product.id))
            if not kit:
                kit = MarketingKit(product_id=product.id)
                ses.add(kit)
            kit.landing, kit.emails = spec.landing.model_dump(), [e.model_dump() for e in spec.emails]
            kit.social, kit.provider = spec.social.model_dump(), provider
            db_p = await ses.get(Product, product.id)
            if db_p and db_p.status in ("built", "marketing"):
                db_p.status = "ready"
        await self.log("copy", f"Kit for {product.name}: PAS landing, value anchor, {len(spec.emails)} emails, "
                       f"{len(spec.social.x) + len(spec.social.linkedin) + len(spec.social.reddit)} posts", product_id=product.id,
                       provider=provider)
        await self.status("idle", f"Copy ready: “{spec.landing.headline[:60]}”")
        return kit
