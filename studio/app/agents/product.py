"""Agent 1 — Iris, Head of Product: find hair-on-fire pains, measure them, propose MVPs that sell.

Every mined post is scored with the frameworks in ``app.agents.frameworks`` (urgency, money/time loss,
workaround steps -> WTP, Opportunity Score). Topics are ranked by that evidence plus what the market
already taught us (sales and flops per topic). For each candidate Iris computes the idea verdict first
(TTV, distribution, moat, value-based price, MVP score): ideas that fail are discarded with their
reasons before anyone spends an LLM call or a meeting on them. Only the words come from the LLM.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.agents import frameworks, playbook
from app.agents.base import Agent
from app.integrations.research import RawPost, mine
from app.models import Idea, Product, Sale, Signal, utcnow

URGENCY_WEIGHT = {"hair_on_fire": 1.5, "important": 1.0, "nice_to_have": 0.3}


class IdeaSpec(BaseModel):
    title: str = Field(max_length=120)
    job_to_be_done: str = Field(description="When <situation>, I want to <motivation>, so I can <outcome>.")
    emotional_job: str = Field(description="The feeling the buyer wants to get rid of or gain (the hidden motivation)")
    social_job: str = Field(description="How the buyer wants to be seen by clients/peers")
    problem: str
    niche: str
    solution_type: Literal["n8n", "python"]
    mvp: str = Field(description="What the buyer imports/configures and what then runs on its own")
    rationale: str


def catalogue_posts() -> list[RawPost]:
    """Offline seed: one synthetic signal per known pain, clearly labelled as source 'playbook'."""
    return [RawPost("playbook", t.key, f"Known pain: {t.problem[:180]}", "", t.problem, 0) for t in playbook.TOPICS]


CREDENTIAL_KINDS = {"email": "SMTP", "sheets": "Google", "slack": "Slack"}


def assess_topic(t: playbook.Topic, signals: list[dict[str, Any]]) -> frameworks.IdeaAssessment:
    creds = {CREDENTIAL_KINDS[s.kind] for s in t.steps if s.kind in CREDENTIAL_KINDS}
    return frameworks.assess_idea(
        signals, ttv_minutes=t.setup_minutes, integrations=t.integrations, search_queries=t.search_queries,
        ecosystems=t.ecosystems, hours_saved_month=t.hours_saved_month, commodity=t.commodity,
        scope_steps=len(t.steps) if t.steps else 4, env_vars=len(t.env) + 1,  # +1: WEBHOOK_SECRET
        credentials=len(creds), needs_server=t.solution_type == "python", technical_audience=t.technical_audience,
        payer_prior=t.payer_prior)


class HeadOfProduct(Agent):
    key = "product"

    async def research(self) -> list[Idea]:
        s = self.ctx.settings
        await self.status("working", "Mining Reddit, Hacker News and Product Hunt for hair-on-fire pains")
        status: dict[str, str] = {}
        posts: list[RawPost] = []
        if not s.research_offline:
            posts, status = await mine(s, self.ctx.http)
        if not posts:
            posts = catalogue_posts()
            status["playbook"] = "used built-in pain catalogue (no live source answered)"
        new, hot = await self._store(posts)
        await self.log("research", f"Mined {len(posts)} posts: {new} new pain signals, {hot} hair-on-fire", sources=status)
        await self.status("thinking", "Scoring opportunity, workarounds, TTV, distribution and moat")
        ideas = await self._propose()
        kept = [i for i in ideas if i.status == "proposed"]
        await self.status("idle", f"{len(kept)} ideas to the board, {len(ideas) - len(kept)} discarded" if ideas
                          else "No new pain worth a product today")
        return kept

    async def _store(self, posts: list[RawPost]) -> tuple[int, int]:
        new = hot = 0
        async with self.ctx.db.session() as ses:
            for p in posts:
                text = f"{p.title}\n{p.text}"
                topic = p.external_id if p.source == "playbook" else playbook.classify(text)
                # curated catalogue entries count as two intent hits: known, recurring pains
                intent = 2.0 if p.source == "playbook" else playbook.intent_score(text)
                if not topic:
                    continue
                a = frameworks.assess_post(text, p.engagement, intent)
                # ingestion rule: keep explicit pains; a nice-to-have post with no ask is noise
                if p.source != "playbook" and intent == 0 and a.urgency == "nice_to_have":
                    continue
                exists = await ses.scalar(select(Signal.id).where(Signal.source == p.source, Signal.external_id == p.external_id))
                if exists:
                    continue
                ses.add(Signal(source=p.source, external_id=p.external_id, title=p.title[:500], url=p.url, text=p.text[:4000],
                               topic=topic, engagement=p.engagement, intent=intent, assessment=a.as_dict()))
                new += 1
                hot += a.urgency == "hair_on_fire"
        return new, hot

    async def _signals(self, since_days: int = 30) -> dict[str, list[dict[str, Any]]]:
        since = utcnow() - timedelta(days=since_days)
        async with self.ctx.db.session() as ses:
            rows = (await ses.execute(select(Signal.topic, Signal.assessment, Signal.engagement, Signal.intent)
                                      .where(Signal.created_at >= since))).all()
        out: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for topic, a, eng, intent in rows:
            if not a:  # signals stored before the frameworks existed
                a = {"urgency": "important", "opportunity": 8.0 + intent, "wtp": 0.0, "workaround_steps": 0}
            out[topic].append({**a, "engagement": eng})
        return out

    async def topic_scores(self) -> dict[str, dict[str, float]]:
        """Evidence (opportunity x urgency + WTP + reach) plus what the market already taught us."""
        strategy = await self.ctx.db.get_state("strategy")
        focus = set(strategy.get("focus_topics", []))
        roi = strategy.get("margins", {})  # Nina's 30-day ROI per topic (cohort of real sales)
        topic_cr = (await self.ctx.db.get_state("funnel")).get("topic_cr", {})  # Leo's visitor-to-buyer per topic
        async with self.ctx.db.session() as ses:
            sold = dict((await ses.execute(select(Idea.topic, func.count(Sale.id)).join(Product, Product.idea_id == Idea.id)
                                           .join(Sale, Sale.product_id == Product.id).where(Sale.refunded.is_(False))
                                           .group_by(Idea.topic))).all())
            flops = dict((await ses.execute(select(Idea.topic, func.count(Product.id)).join(Product, Product.idea_id == Idea.id)
                                            .where(Product.status == "published",
                                                   Product.published_at < utcnow() - timedelta(days=21)).group_by(Idea.topic))).all())
        agg: dict[str, dict[str, float]] = {}
        for topic, sigs in (await self._signals()).items():
            score = sum(URGENCY_WEIGHT.get(a.get("urgency", ""), 1.0) * a.get("opportunity", 0) / 2 + a.get("wtp", 0) / 2
                        + math.log1p(max(a.get("engagement", 0), 0)) for a in sigs)
            score += 3 * sold.get(topic, 0) - 2 * max(0, flops.get(topic, 0) - sold.get(topic, 0))
            score += 2.0 if topic in focus else 0.0
            score += min(5.0, 2.0 * float(roi.get(topic, 0)))  # topics that made real money rank higher
            cr = topic_cr.get(topic)
            if cr is not None:  # the market's verdict on our landing pages
                score += 2.0 if cr > 0.03 else -2.0 if cr < 0.005 else 0.0
            hof = sum(a.get("urgency") == "hair_on_fire" for a in sigs)
            agg[topic] = {"signals": float(len(sigs)), "score": round(score, 2), "hair_on_fire": float(hof),
                          "sales": float(sold.get(topic, 0))}
        return agg

    async def verdicts(self) -> dict[str, frameworks.IdeaAssessment]:
        signals = await self._signals()
        return {t: assess_topic(playbook.BY_KEY[t], sigs) for t, sigs in signals.items() if t in playbook.BY_KEY}

    async def _propose(self) -> list[Idea]:
        s = self.ctx.settings
        scores = await self.topic_scores()
        signals = await self._signals()
        async with self.ctx.db.session() as ses:
            busy = set((await ses.scalars(select(Idea.topic).where(
                (Idea.status.notin_(["rejected", "failed"])) | (Idea.created_at >= utcnow() - timedelta(days=14))))).all())
        candidates = [(t, a) for t, a in scores.items() if t not in busy and t in playbook.BY_KEY]
        verdicts = {t: assess_topic(playbook.BY_KEY[t], signals.get(t, [])) for t, _ in candidates}
        ranked = sorted(candidates, key=lambda x: (-verdicts[x[0]].approve, -verdicts[x[0]].mvp_score, -x[1]["score"]))
        ideas: list[Idea] = []
        kept = 0
        for topic, agg in ranked:
            v = verdicts[topic]
            if v.approve and kept < s.ideas_per_research:
                ideas.append(await self._draft(topic, agg, v))
                kept += 1
            elif not v.approve:
                ideas.append(await self._discard(topic, agg, v))
        return ideas

    async def _evidence(self, topic: str) -> list[tuple[str, str, str]]:
        async with self.ctx.db.session() as ses:
            return [tuple(r) for r in (await ses.execute(select(Signal.title, Signal.url, Signal.source).where(Signal.topic == topic)
                                                          .order_by(Signal.engagement.desc()).limit(6))).all()]  # type: ignore[misc]

    async def _discard(self, topic: str, agg: dict[str, float], v: frameworks.IdeaAssessment) -> Idea:
        t = playbook.BY_KEY[topic]
        ev = await self._evidence(topic)
        async with self.ctx.db.session() as ses:
            idea = Idea(topic=topic, title=t.product, problem=t.problem, niche=t.niche, solution_type=t.solution_type, mvp=t.mvp,
                        price_cents=v.price_usd * 100, score=v.mvp_score, evidence=[{"title": a, "url": b, "source": c} for a, b, c in ev],
                        rationale="Discarded by Iris: " + "; ".join(v.reasons), provider="frameworks", status="rejected",
                        assessment=v.as_dict())
            ses.add(idea)
        await self.log("discard", f"Discarded {t.product}: {'; '.join(v.reasons)}", idea_id=idea.id)
        return idea

    async def _draft(self, topic: str, agg: dict[str, float], v: frameworks.IdeaAssessment) -> Idea:
        t = playbook.BY_KEY[topic]
        ev = await self._evidence(topic)
        quotes = "\n".join(f"- [{src}] {title}" for title, _url, src in ev)

        def fallback() -> IdeaSpec:
            return IdeaSpec(title=t.product, problem=t.problem, niche=t.niche, solution_type=t.solution_type,  # type: ignore[arg-type]
                            job_to_be_done=f"When {t.problem.split(';')[0].lower().rstrip('.')}, I want it handled automatically, "
                                           f"so I can stop losing time and money.",
                            emotional_job=t.emotional_job or "Stop the stress of doing it by hand.",
                            social_job=t.social_job or "Look more professional to clients.",
                            mvp=t.mvp, rationale=self._why(v, agg))

        spec, provider = await self.ctx.llm.json(
            self.key, self.persona.system,
            f"Pain category: {topic}\nKnown framing: {t.problem}\nTypical buyer: {t.niche}\n"
            f"Integrations available: {', '.join(t.integrations)}\nEcosystems: {', '.join(t.ecosystems)}\n"
            f"Evidence ({int(agg['signals'])} signals, {int(agg['hair_on_fire'])} hair-on-fire):\n{quotes}\n\n"
            f"Measured: urgency {v.urgency}, opportunity {v.opportunity:.1f}/20, WTP {v.wtp:.1f}/10 (max {v.max_workaround_steps} "
            f"manual steps described), saves ~{v.hours_saved_month:.0f} h/month (${v.value_month_low:.0f}-{v.value_month_high:.0f} of "
            f"the buyer's time), price ${v.price_usd} one-time.\n\n"
            f"Friction {v.friction:.1f}/10 (tolerance {v.friction_tolerance:.1f}); payer habit {v.payer_habit:+.2f}; "
            f"MVP scope {v.scope_steps} steps.\nKnown hidden motivation: {t.emotional_job} / {t.social_job}\n\n"
            "Write ONE narrow MVP the buyer can set up in under 5 minutes (import, fill 2-3 keys, activate), as an n8n workflow or "
            "a small FastAPI service. It must solve 100% of ONE job in a single use: no extra features. State the functional "
            "Job-To-Be-Done, the emotional job (the feeling behind the request) and the social job (how they want to be seen). "
            "Be specific about the niche. Do not change the numbers.",
            IdeaSpec, fallback)
        async with self.ctx.db.session() as ses:
            idea = Idea(topic=topic, title=spec.title[:200], problem=spec.problem, niche=spec.niche[:200], solution_type=spec.solution_type,
                        mvp=spec.mvp, price_cents=v.price_usd * 100, score=v.mvp_score,
                        evidence=[{"title": a, "url": b, "source": c} for a, b, c in ev],
                        rationale=f"JTBD: {spec.job_to_be_done} · {spec.rationale}", provider=provider,
                        assessment={**v.as_dict(), "job_to_be_done": spec.job_to_be_done,
                                    "emotional_job": spec.emotional_job, "social_job": spec.social_job})
            ses.add(idea)
        await self.log("idea", f"New idea: {idea.title} — {v.urgency.replace('_', ' ')}, opportunity {v.opportunity:.1f}, "
                       f"MVP {v.mvp_score:.0f}, ${v.price_usd}", idea_id=idea.id, provider=provider)
        return idea

    @staticmethod
    def _why(v: frameworks.IdeaAssessment, agg: dict[str, float]) -> str:
        return (f"{int(agg['signals'])} signals ({int(agg['hair_on_fire'])} hair-on-fire); saves ~{v.hours_saved_month:.0f} h/month "
                f"= ${v.value_month_low:.0f}-{v.value_month_high:.0f} of the buyer's time vs ${v.price_usd} once "
                f"(×{v.no_brainer_ratio:.0f}); {v.moat} integrated systems; setup {v.ttv_minutes} min.")
