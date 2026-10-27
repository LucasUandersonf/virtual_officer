"""Agent 1 — Iris, Head of Product: mine pains, score them, propose MVPs with a USD price."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import timedelta
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.agents import playbook
from app.agents.base import Agent
from app.integrations.research import RawPost, mine
from app.models import Idea, Product, Sale, Signal, utcnow


class IdeaSpec(BaseModel):
    title: str = Field(max_length=120)
    problem: str
    niche: str
    solution_type: Literal["n8n", "python"]
    mvp: str
    price_usd: int = Field(ge=9, le=299)
    rationale: str


def catalogue_posts() -> list[RawPost]:
    """Offline seed: one synthetic signal per known pain, clearly labelled as source 'playbook'."""
    return [RawPost("playbook", t.key, f"Known pain: {t.problem[:180]}", "", t.problem, 0) for t in playbook.TOPICS]


class HeadOfProduct(Agent):
    key = "product"

    async def research(self) -> list[Idea]:
        s = self.ctx.settings
        await self.status("working", "Mining Reddit, Hacker News and Product Hunt for pains")
        status: dict[str, str] = {}
        posts: list[RawPost] = []
        if not s.research_offline:
            posts, status = await mine(s, self.ctx.http)
        if not posts:
            posts = catalogue_posts()
            status["playbook"] = "used built-in pain catalogue (no live source answered)"
        new = await self._store(posts)
        await self.log("research", f"Mined {len(posts)} posts, {new} new pain signals", sources=status)
        await self.status("thinking", "Scoring pains and drafting MVPs")
        ideas = await self._propose()
        await self.status("idle", f"{len(ideas)} new ideas on the board" if ideas else "No new pain worth a product today")
        return ideas

    async def _store(self, posts: list[RawPost]) -> int:
        new = 0
        async with self.ctx.db.session() as ses:
            for p in posts:
                text = f"{p.title}\n{p.text}"
                topic = p.external_id if p.source == "playbook" else playbook.classify(text)
                # curated catalogue entries count as two intent hits: known, recurring pains
                intent = 2.0 if p.source == "playbook" else playbook.intent_score(text)
                if not topic or (intent == 0 and p.source != "playbook"):
                    continue
                exists = await ses.scalar(select(Signal.id).where(Signal.source == p.source, Signal.external_id == p.external_id))
                if exists:
                    continue
                ses.add(Signal(source=p.source, external_id=p.external_id, title=p.title[:500], url=p.url, text=p.text[:4000],
                               topic=topic, engagement=p.engagement, intent=intent))
                new += 1
        return new

    async def topic_scores(self) -> dict[str, dict[str, float]]:
        """Demand evidence over the last 30 days + what the market already taught us (sales per topic)."""
        since = utcnow() - timedelta(days=30)
        focus = set((await self.ctx.db.get_state("strategy")).get("focus_topics", []))
        async with self.ctx.db.session() as ses:
            sigs = (await ses.execute(select(Signal.topic, Signal.intent, Signal.engagement).where(Signal.created_at >= since))).all()
            sold = dict((await ses.execute(select(Idea.topic, func.count(Sale.id)).join(Product, Product.idea_id == Idea.id)
                                           .join(Sale, Sale.product_id == Product.id).group_by(Idea.topic))).all())
            flops = dict((await ses.execute(select(Idea.topic, func.count(Product.id)).join(Product, Product.idea_id == Idea.id)
                                            .where(Product.status == "published",
                                                   Product.published_at < utcnow() - timedelta(days=21)).group_by(Idea.topic))).all())
        agg: dict[str, dict[str, float]] = defaultdict(lambda: {"signals": 0.0, "score": 0.0})
        for topic, intent, eng in sigs:
            agg[topic]["signals"] += 1
            agg[topic]["score"] += 2 * intent + math.log1p(max(eng, 0))
        for topic, a in agg.items():
            a["score"] += 3 * sold.get(topic, 0) - 2 * max(0, flops.get(topic, 0) - sold.get(topic, 0))
            a["score"] += 2.0 if topic in focus else 0.0
        return dict(agg)

    async def _propose(self) -> list[Idea]:
        s = self.ctx.settings
        scores = await self.topic_scores()
        async with self.ctx.db.session() as ses:
            busy = set((await ses.scalars(select(Idea.topic).where(
                (Idea.status.notin_(["rejected", "failed"])) | (Idea.created_at >= utcnow() - timedelta(days=14))))).all())
        ranked = sorted(((t, a) for t, a in scores.items() if t not in busy and t in playbook.BY_KEY),
                        key=lambda x: -x[1]["score"])[: s.ideas_per_research]
        ideas: list[Idea] = []
        for topic, agg in ranked:
            ideas.append(await self._draft(topic, agg))
        return ideas

    async def _draft(self, topic: str, agg: dict[str, float]) -> Idea:
        t = playbook.BY_KEY[topic]
        async with self.ctx.db.session() as ses:
            ev = (await ses.execute(select(Signal.title, Signal.url, Signal.source).where(Signal.topic == topic)
                                    .order_by(Signal.engagement.desc()).limit(6))).all()
        quotes = "\n".join(f"- [{src}] {title}" for title, _url, src in ev)

        def fallback() -> IdeaSpec:
            return IdeaSpec(title=t.product, problem=t.problem, niche=t.niche, solution_type=t.solution_type,  # type: ignore[arg-type]
                            mvp=t.mvp, price_usd=t.price_usd,
                            rationale=f"{int(agg['signals'])} recent signals on this pain; proven category for {t.niche.lower()}.")

        spec, provider = await self.ctx.llm.json(
            self.key, self.persona.system,
            f"Pain category: {topic}\nKnown framing: {t.problem}\nTypical buyer: {t.niche}\n"
            f"Evidence from forums ({int(agg['signals'])} signals, score {agg['score']:.1f}):\n{quotes}\n\n"
            "Propose ONE narrow MVP that can ship as an n8n workflow or a small FastAPI service within a day. "
            "Price it in USD as a one-time purchase (typical 19-99). Be specific about the niche.",
            IdeaSpec, fallback)
        async with self.ctx.db.session() as ses:
            idea = Idea(topic=topic, title=spec.title[:200], problem=spec.problem, niche=spec.niche[:200], solution_type=spec.solution_type,
                        mvp=spec.mvp, price_cents=spec.price_usd * 100, score=round(agg["score"], 2),
                        evidence=[{"title": a, "url": b, "source": c} for a, b, c in ev], rationale=spec.rationale, provider=provider)
            ses.add(idea)
        await self.log("idea", f"New idea: {idea.title} (${spec.price_usd}, score {idea.score})", idea_id=idea.id, provider=provider)
        return idea
