"""Agent 3 — Maya, Growth Hacker & Copywriter: landing page, checkout copy, nurture emails, organic posts (English)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.agents.base import Agent
from app.models import MarketingKit, Product


class FAQ(BaseModel):
    q: str
    a: str


class Landing(BaseModel):
    headline: str = Field(max_length=120)
    subheadline: str
    benefits: list[str] = Field(min_length=3, max_length=6)
    how_it_works: list[str] = Field(min_length=3, max_length=5)
    faq: list[FAQ] = Field(min_length=3, max_length=6)
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
    emails: list[Email] = Field(min_length=3, max_length=6)
    social: Social


def fallback_kit(p: Product) -> KitSpec:
    i = p.idea
    price = f"${p.price_cents // 100}"
    kind = "n8n workflow" if p.solution_type == "n8n" else "ready-to-deploy FastAPI service"
    return KitSpec(
        landing=Landing(
            headline=p.tagline or p.name,
            subheadline=f"A {kind} for {i.niche.lower()}. Install in 10 minutes, then it runs on its own.",
            benefits=[f"Stops this for good: {i.problem.split(';')[0].rstrip('.')}.",
                      "Runs 24/7 on your own stack: no new SaaS subscription.",
                      "Every setting is a plain variable, documented step by step.",
                      "One-time price, lifetime use for your business."],
            how_it_works=["Download the package and open the README.", f"Import the {kind} and fill in .env.example.",
                          "Test with the sample data, switch it on and forget about it."],
            faq=[FAQ(q="Do I need to code?", a="No for n8n workflows: import and fill in the variables. "
                     "The Python services deploy with one command and come with a Dockerfile."),
                 FAQ(q="What do I receive?", a="A .zip with the solution, an English install guide, .env.example and start scripts."),
                 FAQ(q="Is it a subscription?", a=f"No. {price} once, for use in your business and your clients' projects."),
                 FAQ(q="What if it doesn't fit my setup?", a="Reply to the purchase email; we help you adapt it.")],
            cta=f"Get it for {price}", checkout_blurb=f"{p.name}: instant download, install guide included."),
        emails=[Email(day=0, subject=f"Your {p.name} is ready", body="Here is your download link and the 10-minute setup guide."),
                Email(day=2, subject="The one setting most people miss", body="Open .env.example and set every variable before activating."),
                Email(day=5, subject="How is it running?", body="Reply with a screenshot of your first run, we'll review it for free."),
                Email(day=10, subject="Your next automation", body="Tell us the next manual task you'd like gone; we build what buyers ask for.")],
        social=Social(
            x=[f"{i.problem.split('.')[0]}. So we built a fix: {p.name}. {p.tagline}",
               f"Solopreneur tip: automate the boring money tasks first. {p.name} does it for {i.niche.lower().split('(')[0].strip()}."],
            linkedin=[f"Most {i.niche.split('(')[0].strip().lower()} lose hours every week to this: {i.problem}\n\n"
                      f"We packaged the fix as {kind}: {p.name}. {p.tagline}"],
            reddit=[RedditPost(subreddit="r/n8n" if p.solution_type == "n8n" else "r/SaaS",
                               title=f"How I automated: {i.title}",
                               body=f"{i.problem}\n\nThe flow: {i.mvp}\n\nHappy to share details in the comments.")]),
    )


class Growth(Agent):
    key = "growth"

    async def create_kit(self, product: Product) -> MarketingKit:
        await self.status("working", f"Writing copy for {product.name}")
        i = product.idea
        spec, provider = await self.ctx.llm.json(
            self.key, self.persona.system,
            f"Product: {product.name}\nTagline: {product.tagline}\nType: {product.solution_type}\nPrice: ${product.price_cents / 100:.0f} one-time\n"
            f"Buyer: {i.niche}\nProblem: {i.problem}\nWhat it does: {i.mvp}\n\n"
            "Write: landing page + checkout blurb, a 4-email onboarding/nurture sequence (day 0 delivery, then value), "
            "and organic posts for X, LinkedIn and Reddit (value-first, no spam, Reddit posts teach the method).",
            KitSpec, lambda: fallback_kit(product), max_tokens=3000, temperature=0.7)
        async with self.ctx.db.session() as ses:
            kit = MarketingKit(product_id=product.id, landing=spec.landing.model_dump(), emails=[e.model_dump() for e in spec.emails],
                               social=spec.social.model_dump(), provider=provider)
            ses.add(kit)
            db_p = await ses.get(Product, product.id)
            if db_p:
                db_p.status = "ready"
        await self.log("copy", f"Kit for {product.name}: landing, {len(spec.emails)} emails, "
                       f"{len(spec.social.x) + len(spec.social.linkedin) + len(spec.social.reddit)} posts", product_id=product.id,
                       provider=provider)
        await self.status("idle", f"Copy ready: “{spec.landing.headline[:60]}”")
        return kit
