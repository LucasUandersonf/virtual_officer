"""The office: runs the pipeline and the meetings.

Pipeline per tick (every PIPELINE_INTERVAL_MIN):
  daily close -> research (when due) -> weekly goals meeting (when due) -> idea review meetings
  -> build -> marketing kit -> publish (AUTOPUBLISH or CEO approval).

Meetings follow one rule taken from the quant desk in this repo: **votes come from numbers in code,
LLMs only write the words.** Each agent's vote is a deterministic gate (effort, evidence, price band,
payback); one LLM call writes everybody's lines, so a meeting costs a single free-tier request.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select

from app.agents import frameworks
from app.agents.base import PERSONAS, Context
from app.agents.builder import Builder, effort_hours
from app.agents.cfo import CFO
from app.agents.frameworks import charm_price
from app.agents.growth import Growth
from app.agents.product import HeadOfProduct
from app.agents.sales import Sales
from app.models import Idea, Meeting, MeetingMessage, Product, utcnow
from app.services import finance

log = logging.getLogger(__name__)
ORDER = ("product", "builder", "growth", "sales", "cfo")


class Lines(BaseModel):
    product: str = Field(max_length=420)
    builder: str = Field(max_length=420)
    growth: str = Field(max_length=420)
    sales: str = Field(max_length=420)
    cfo: str = Field(max_length=420)


class Office:
    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.product = HeadOfProduct(ctx)
        self.builder = Builder(ctx)
        self.growth = Growth(ctx)
        self.sales = Sales(ctx)
        self.cfo = CFO(ctx)
        self._lock = asyncio.Lock()
        self.last_tick: str = ""

    # ------------------------------------------------------------------ helpers
    async def _product(self, pid: int) -> Product:
        async with self.ctx.db.session() as ses:
            p = await ses.get(Product, pid)
            assert p is not None
            return p

    async def _pause(self, factor: float = 1.0) -> None:
        if self.ctx.settings.theatre_delay_sec > 0:
            await asyncio.sleep(self.ctx.settings.theatre_delay_sec * factor)

    async def _lines(self, subject: str, facts: str, fallback: Lines) -> tuple[Lines, str]:
        roles = "\n".join(f"- {k}: {p.name}, {p.role}" for k, p in PERSONAS.items())
        return await self.ctx.llm.json(
            "office", "You write the minutes of a short meeting in a small autonomous product studio. Team:\n" + roles +
            "\nEach person speaks once (1-3 sentences, first person, specific, uses the numbers given, no invented facts). "
            "Their vote is already decided by the numbers: the line must be consistent with it.",
            f"Meeting: {subject}\nFacts and votes:\n{facts}", Lines, lambda: fallback, tier="fast", max_tokens=900)

    async def _meeting(self, kind: str, subject: str, idea_id: int | None, lines: Lines, votes: dict[str, str],
                       data: dict[str, dict[str, Any]], decision: str, summary: str) -> Meeting:
        bus = self.ctx.bus
        async with self.ctx.db.session() as ses:
            m = Meeting(kind=kind, subject=subject[:300], idea_id=idea_id)
            ses.add(m)
        bus.meeting = {"id": m.id, "kind": kind, "subject": subject, "votes": {}}
        bus.emit("meeting_start", id=m.id, kind=kind, subject=subject, participants=list(ORDER))
        await self._pause(1.2)  # walking to the board room
        for role in ORDER:
            bus.emit("typing", agent=role)
            await self._pause()
            text = getattr(lines, role)
            async with self.ctx.db.session() as ses:
                ses.add(MeetingMessage(meeting_id=m.id, agent=role, text=text, vote=votes.get(role, ""), data=data.get(role, {})))
            bus.meeting["votes"][role] = votes.get(role, "")
            bus.emit("meeting_message", meeting_id=m.id, agent=role, text=text, vote=votes.get(role, ""), data=data.get(role, {}))
        await self._pause()
        async with self.ctx.db.session() as ses:
            row = await ses.get(Meeting, m.id)
            assert row is not None
            row.decision, row.summary, row.ended_at = decision[:300], summary, utcnow()
        bus.emit("meeting_end", id=m.id, decision=decision, summary=summary)
        bus.meeting = None
        await self.ctx.bus.log("office", "meeting", f"{subject} → {decision}", meeting_id=m.id)
        return m

    # ------------------------------------------------------------------ idea review
    async def review_idea(self, idea: Idea) -> str:
        """Each agent votes with its own framework (numbers in code); one LLM call writes the minutes."""
        a = dict(idea.assessment or {})
        eco = await self.cfo.economics(idea)
        hours = effort_hours(idea)
        n_ev = len(idea.evidence or [])
        price = idea.price_cents // 100
        new_price = charm_price(price)
        ttv = int(a.get("ttv_minutes", 5))
        dist = float(a.get("distribution", 0))
        value_low = float(a.get("value_month_low", 0))
        ratio = value_low / new_price if new_price else 0.0
        votes = {
            # Iris: hair-on-fire/important pain, TTV <= 5 min, moat >= 2, not a commodity, MVP score >= 45
            "product": "APPROVE" if a.get("approve", idea.score >= 2) else "REJECT",
            # Theo: buildable in a day and installable in <= 5 minutes
            "builder": "APPROVE" if idea.solution_type in ("n8n", "python") and hours <= 8 and ttv <= frameworks.TTV_LIMIT_MIN else "REJECT",
            # Maya: zero-CAC distribution (search intent + ecosystem) and not saturated
            "growth": "APPROVE" if (dist >= 5 or (not a and n_ev >= 2)) and not a.get("commodity") else "REJECT",
            # Leo: charm price in $19-149 and a no-brainer (one month of value >= 3x the price)
            "sales": "APPROVE" if ratio >= 3 or not a else "REJECT",
            # Nina: payback >= 1, contribution margin >= 70 %, build slot free
            "cfo": "APPROVE" if eco.ok else "REJECT",
        }
        approvals = sum(v == "APPROVE" for v in votes.values())
        decision = "approved" if votes["cfo"] == "APPROVE" and approvals >= 4 else "rejected"
        data = {"product": {k: a.get(k) for k in ("urgency", "opportunity", "wtp", "moat", "mvp_score")} | {"evidence": n_ev},
                "builder": {"effort_h": hours, "ttv_min": ttv},
                "growth": {"distribution": dist, "commodity": bool(a.get("commodity"))},
                "sales": {"price_usd": new_price, "was": price, "value_month_low": value_low, "no_brainer": round(ratio, 1)},
                "cfo": {"est_sales_month": eco.est_monthly_sales, "est_net_usd": eco.est_monthly_net_usd,
                        "margin": eco.contribution_margin, "cost_share_usd": eco.cost_share_usd, "payback": eco.payback_ratio,
                        "builds_left": eco.builds_left_today}}
        cfo_why = "; ".join(eco.reasons()) or (f"${eco.est_monthly_net_usd:.0f}/month expected vs ${eco.cost_share_usd:.2f} share "
                                               f"(×{eco.payback_ratio:.1f}), margin {eco.contribution_margin:.0%}")
        urg = str(a.get("urgency", "unknown")).replace("_", " ")
        fallback = Lines(
            product=(f"Pitch: {idea.title}. Urgency: {urg}; opportunity {float(a.get('opportunity', 0)):.1f}/20, WTP "
                     f"{float(a.get('wtp', 0)):.1f}/10, moat {a.get('moat', '?')} systems, MVP score {idea.score:.0f}. "
                     f"Buyer: {idea.niche}." if a else f"Pitch: {idea.title}. {idea.problem}"),
            builder=(f"{'Feasible' if votes['builder'] == 'APPROVE' else 'No'}: {idea.solution_type}, ~{hours:.1f} h to build, buyer "
                     f"setup ~{ttv} min. Guard node for auth + idempotency, retries on every external call."),
            growth=(f"Distribution {dist:.0f}/10: clear search intent and an ecosystem to ride; I'll lead with PAS on the measured pain."
                    if votes["growth"] == "APPROVE" else f"Distribution {dist:.0f}/10 or saturated: no zero-CAC channel, I can't sell it."),
            sales=(f"${new_price} one-time" + (f" (from ${price})" if new_price != price else "") +
                   (f": one month of value is ${value_low:.0f}, ×{ratio:.0f} the price — a no-brainer." if a else ".") +
                   f" Channels: {', '.join(self.ctx.settings.platforms()) or 'studio landing page until a platform key is set'}."),
            cfo=f"{'Approve' if votes['cfo'] == 'APPROVE' else 'Veto'}: {cfo_why}.")
        facts = "\n".join(f"{k}: vote {votes[k]}; {data[k]}" for k in ORDER) + f"\nIdea: {idea.title} — {idea.problem}\nMVP: {idea.mvp}"
        lines, _ = await self._lines(f"Idea review: {idea.title}", facts, fallback)
        await self._meeting("idea_review", f"Idea review: {idea.title}", idea.id, lines, votes, data, decision,
                            f"{approvals}/5 approve · CFO: {cfo_why}")
        async with self.ctx.db.session() as ses:
            row = await ses.get(Idea, idea.id)
            if row:
                row.status = decision
                row.price_cents = new_price * 100
        return decision

    # ------------------------------------------------------------------ weekly goals
    async def weekly_goals(self) -> None:
        s = self.ctx.settings
        g = await finance.goals(self.ctx.db, s)
        scores = await self.product.topic_scores()
        margins = await self.cfo.topic_margins(30)
        # Nina's rule: real contribution margin of the last 30 days first, then Iris's evidence
        verdicts = await self.product.verdicts()
        # topics Iris would discard (TTV, commodity, thin moat) only stay in focus if they already make money
        eligible = [t for t in scores if margins.get(t, 0.0) > 0 or (t in verdicts and verdicts[t].approve)]
        ranked = sorted(eligible, key=lambda t: (-margins.get(t, 0.0), -scores[t]["score"]))
        focus = ranked[:3]
        await self.ctx.db.set_state("strategy", {"focus_topics": focus, "set_at": date.today().isoformat(), "margins": margins})
        portfolio = await self.cfo.portfolio()
        stale = [p["name"] for p in portfolio if p["health"] in ("reposition", "shutdown")]
        rev = g["revenue"]
        behind = rev["run_rate"] < rev["target"]
        votes = {k: "" for k in ORDER}
        by_margin = ", ".join(f"{t} ${m:.0f}" for t, m in sorted(margins.items(), key=lambda x: -x[1])[:3]) or "no sales yet"
        fallback = Lines(
            cfo=(f"Month to date: ${rev['current']:.2f} net, run-rate ${rev['run_rate']:.2f} vs goal ${rev['target']:.0f}; profit "
                 f"${g['profit_mtd']:.2f} after fees and ${g['break_even']['monthly_cost_usd']:.2f}/month of infrastructure. "
                 f"30-day margin by topic: {by_margin}." + (f" Stale (21+ days, no sale): {', '.join(stale)}." if stale else "")),
            product=f"Focus this week: {', '.join(focus) or 'broad research'} — real margin first, then the strongest hair-on-fire evidence.",
            builder=f"{g['products_week']['current']}/{g['products_week']['target']} products this week. Small, guarded builds; setup under 5 minutes.",
            growth=("We're behind: one value-first Reddit/LinkedIn post per live product, and new angles for the stale ones." if behind
                    else "On track: double down on the channels that brought sales."),
            sales=f"{g['live_products']} products live. Download in seconds after the signed webhook; consented abandoned checkouts get one follow-up.")
        facts = f"Goals snapshot: {g}\n30-day contribution by topic: {margins}\nFocus chosen: {focus}\nStale products: {stale}"
        lines, _ = await self._lines("Weekly goals review", facts, fallback)
        await self._meeting("weekly_goals", "Weekly goals review", None, lines, votes, {"cfo": g}, "focus: " + ", ".join(focus),
                            f"Run-rate ${rev['run_rate']:.2f} vs ${rev['target']:.0f} goal; focus {', '.join(focus)}")
        await self.ctx.db.set_state("weekly", {"last": date.today().isoformat()})

    # ------------------------------------------------------------------ pipeline
    async def tick(self) -> dict[str, Any]:
        if self._lock.locked():
            return {"skipped": "busy"}
        async with self._lock:
            return await self._tick()

    async def _due(self, key: str, minutes: int) -> bool:
        st = await self.ctx.db.get_state(key)
        last = st.get("last_at")
        return not last or datetime.fromisoformat(last) + timedelta(minutes=minutes) <= utcnow()

    async def _tick(self) -> dict[str, Any]:
        s = self.ctx.settings
        done: dict[str, Any] = {}
        await self.cfo.daily_close(self.growth, self.sales)
        if await self._due("research", s.research_interval_min):
            ideas = await self.product.research()
            await self.ctx.db.set_state("research", {"last_at": utcnow().isoformat()})
            done["ideas"] = len(ideas)
        weekly = await self.ctx.db.get_state("weekly")
        if not weekly.get("last") or date.fromisoformat(weekly["last"]) <= date.today() - timedelta(days=7):
            await self.weekly_goals()
            done["weekly"] = True
        async with self.ctx.db.session() as ses:
            proposed = (await ses.scalars(select(Idea).where(Idea.status == "proposed").order_by(Idea.score.desc()).limit(2))).all()
        done["reviewed"] = [await self.review_idea(i) for i in proposed]

        async with self.ctx.db.session() as ses:
            approved = (await ses.scalars(select(Idea).where(Idea.status == "approved").order_by(Idea.score.desc()).limit(1))).all()
        for idea in approved:
            if (await self.cfo.economics(idea)).builds_left_today <= 0:
                break
            async with self.ctx.db.session() as ses:
                row = await ses.get(Idea, idea.id)
                if row:
                    row.status = "building"
            try:
                p = await self.builder.build(idea)
                done["built"] = p.name
            except Exception:
                log.exception("build failed for idea %s", idea.id)
                async with self.ctx.db.session() as ses:
                    row = await ses.get(Idea, idea.id)
                    if row:
                        row.status = "failed"
                await self.builder.status("idle", "Build crashed; logged for review")

        async with self.ctx.db.session() as ses:
            built = (await ses.scalars(select(Product.id).where(Product.status == "built"))).all()
        for pid in built:
            await self.growth.create_kit(await self._product(pid))
            done.setdefault("kits", 0)
            done["kits"] += 1

        async with self.ctx.db.session() as ses:
            ready = (await ses.scalars(select(Product.id).where(Product.status == "ready"))).all()
        notified = set((await self.ctx.db.get_state("approval_requests")).get("ids", []))
        for pid in ready:
            p = await self._product(pid)
            if s.autopublish:
                await self.sales.publish(p)
            elif pid not in notified:
                await self.sales.log("approval", f"{p.name} is ready for your approval (${p.price_cents / 100:.0f})", product_id=pid)
                await self.ctx.notify.ceo(f"✅ {p.name} is ready (${p.price_cents / 100:.0f}). Approve it in the office: "
                                          f"{s.public_base_url}/#products")
                notified.add(pid)
                await self.sales.status("waiting", f"{p.name} awaits CEO approval")
        await self.ctx.db.set_state("approval_requests", {"ids": sorted(notified)})
        self.last_tick = utcnow().isoformat(timespec="seconds")
        return done
