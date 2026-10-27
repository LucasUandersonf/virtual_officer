"""Agent 3 — Maya, Growth Hacker & Copywriter: direct-response copy in US English.

* PAS (Problem-Agitate-Solve) built on the pain Iris measured.
* Value anchoring with real numbers only: hours saved x $50-150/h, and what a freelancer would charge to
  build the integration, against the one-time price.
* Objection removal at checkout: the FAQ always answers "Do I need to code?", "How long does setup take?",
  "Does it work on my n8n plan?" (n8n products) and "What if it doesn't work for me?".
* A 4-email onboarding sequence (day 0, 1, 3, 7) aimed at the first win and low refunds.
* Honesty gate: copy with invented testimonials, customer counts, ratings or guarantees is thrown away
  and replaced by the deterministic kit — a fake claim costs more in refunds and bans than it earns.
"""

from __future__ import annotations

import re

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
                         r"★|5[- ]star|rated \d|guarantee[ds]?\b|#1\b|best[- ]selling|as seen on)", re.IGNORECASE)


def honesty_violations(kit: KitSpec) -> list[str]:
    return sorted({m.group(0) for m in FAKE_CLAIMS.finditer(kit.model_dump_json())})


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
            cta=f"Get it for {price}", checkout_blurb=f"{p.name}: instant download, 5-step install guide included."),
        emails=[
            Email(day=0, subject=f"Your {p.name} is ready (5-minute setup inside)",
                  body="Here is your download. Open README.md and follow the 5 steps: import, credentials, variables, test, activate."),
            Email(day=1, subject="Your first win in 5 minutes",
                  body="Run the included test once. When you see the first result, switch it on: from here it works without you."),
            Email(day=3, subject="Stuck? The 2 settings that fix 90% of setups",
                  body="Check that every variable from .env.example is set and that the credentials in each node are selected. "
                       "Still stuck? Reply with a screenshot and we'll help."),
            Email(day=7, subject=f"Your first week: ~{n['hours'] / 4:.0f} hours back",
                  body="That's roughly the time this saves each week. Tell us the next manual task you'd like gone; we build what buyers ask for."),
        ],
        social=Social(
            x=[f"{pain}. That's ~{n['hours']:.0f} h/month gone. We packaged the fix: {p.name}. {p.tagline}",
               f"Solopreneur math: {n['hours']:.0f} h/month x $50/h = ${n['value_month_low']:,.0f}/month. Automate the boring money tasks first."],
            linkedin=[f"Most {who.lower()} lose hours every week to this: {i.problem}\n\nHere's the flow we use to fix it: {i.mvp}\n\n"
                      f"We packaged it as {a_kind}: {p.name}."],
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
        bad = honesty_violations(spec)
        if bad and provider != "playbook":
            await self.log("honesty", f"Rejected LLM copy for {product.name}: unbacked claims {bad}", product_id=product.id)
            spec, provider = fallback_kit(product), "playbook"
        spec.landing.faq = ensure_objections(spec.landing.faq, product, n)
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
