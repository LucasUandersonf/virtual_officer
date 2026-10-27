"""Agent 5 — Nina, CFO & Corporate Strategy: unit economics, veto, daily close, portfolio audit, weekly focus.

    Net profit = gross revenue (USD) - checkout fees (Stripe/Gumroad/Whop) - infrastructure (Railway + LLM usage)

* Veto: an idea must pay at least its share of the office's fixed cost (payback >= 1) with a contribution
  margin >= 70 % after platform fees, and a build slot must be free today.
* Portfolio audit (daily): a product live > 21 days with no sale is flagged for repositioning (Maya writes a
  new angle, once); > 42 days with no sale is recommended for shutdown (auto-paused only with AUTOPAUSE_STALE).
* Weekly focus: topics ranked by real contribution margin of the last 30 days, then by Iris's evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from sqlalchemy import func, select

from app.agents.base import Agent
from app.integrations.payments import estimate_fee
from app.models import Idea, Product, Sale, utcnow
from app.services import finance

MIN_MARGIN = 0.70
REPOSITION_DAYS, SHUTDOWN_DAYS = 21, 42


@dataclass(frozen=True)
class Economics:
    est_monthly_sales: float
    est_monthly_net_usd: float
    contribution_margin: float  # after the worst platform fee among those configured
    cost_share_usd: float
    payback_ratio: float  # expected monthly contribution / cost share; >= 1 pays its share of the office
    builds_left_today: int

    @property
    def ok(self) -> bool:
        return self.payback_ratio >= 1.0 and self.contribution_margin >= MIN_MARGIN and self.builds_left_today > 0

    def reasons(self) -> list[str]:
        out = []
        if self.builds_left_today <= 0:
            out.append("no build slot left today")
        if self.contribution_margin < MIN_MARGIN:
            out.append(f"contribution margin {self.contribution_margin:.0%} < {MIN_MARGIN:.0%}")
        if self.payback_ratio < 1.0:
            out.append(f"expected ${self.est_monthly_net_usd:.0f}/month does not cover its ${self.cost_share_usd:.2f} share")
        return out


class CFO(Agent):
    key = "cfo"

    async def economics(self, idea: Idea) -> Economics:
        s = self.ctx.settings
        start = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        async with self.ctx.db.session() as ses:
            live = await ses.scalar(select(func.count(Product.id)).where(Product.status.in_(["published", "ready", "built"]))) or 0
            built_today = await ses.scalar(select(func.count(Product.id)).where(Product.created_at >= start)) or 0
        a = idea.assessment or {}
        # Conservative demand model until real sales exist: MVP score 0-100 -> 0.1-2.6 sales/month, x1.5 when the pain
        # is hair-on-fire, capped at 10. Real sales then move the topic priors in Iris's scoring.
        est = min(10.0, (0.1 + idea.score / 40) * (1.5 if a.get("urgency") == "hair_on_fire" else 1.0))
        platforms = s.platforms() or ["stripe"]
        fee = max(estimate_fee(s, pl, idea.price_cents) for pl in platforms)
        margin = (idea.price_cents - fee) / idea.price_cents if idea.price_cents else 0.0
        net = est * (idea.price_cents - fee) / 100
        share = s.monthly_fixed_costs_usd / (live + 1) if s.monthly_fixed_costs_usd else 0.0
        ratio = net / share if share else 99.0
        return Economics(round(est, 2), round(net, 2), round(margin, 3), round(share, 2), round(ratio, 2),
                         max(0, s.max_builds_per_day - built_today))

    async def topic_margins(self, days: int = 30) -> dict[str, float]:
        """Contribution (gross - fees) per topic over the window, in USD."""
        since = utcnow() - timedelta(days=days)
        async with self.ctx.db.session() as ses:
            rows = (await ses.execute(select(Idea.topic, func.sum(Sale.gross_cents - Sale.fee_cents))
                                      .join(Product, Product.idea_id == Idea.id).join(Sale, Sale.product_id == Product.id)
                                      .where(Sale.refunded.is_(False), Sale.created_at >= since).group_by(Idea.topic))).all()
        return {t: int(v or 0) / 100 for t, v in rows}

    async def portfolio(self) -> list[dict[str, Any]]:
        """Per published product: revenue, fees, contribution margin, days live, health."""
        now = utcnow()
        async with self.ctx.db.session() as ses:
            prods = (await ses.scalars(select(Product).where(Product.status.in_(["published", "paused"])))).unique().all()
            agg = {pid: (n, g, f) for pid, n, g, f in (await ses.execute(
                select(Sale.product_id, func.count(Sale.id), func.coalesce(func.sum(Sale.gross_cents), 0),
                       func.coalesce(func.sum(Sale.fee_cents), 0)).where(Sale.refunded.is_(False)).group_by(Sale.product_id))).all()}
        out = []
        for p in prods:
            n, g, f = (int(x) for x in agg.get(p.id, (0, 0, 0)))
            days = (now - p.published_at).days if p.published_at else 0
            health = ("paused" if p.status == "paused" else "selling" if n else "shutdown" if days > SHUTDOWN_DAYS
                      else "reposition" if days > REPOSITION_DAYS else "new")
            out.append({"id": p.id, "name": p.name, "days_live": days, "sales": n, "gross_usd": g / 100, "fees_usd": f / 100,
                        "contribution_usd": (g - f) / 100, "margin": round((g - f) / g, 3) if g else None, "health": health})
        return out

    async def audit(self, growth: Any = None, sales: Any = None) -> list[dict[str, Any]]:
        """Flag stale products; reposition once via Maya; optionally pause after SHUTDOWN_DAYS."""
        s = self.ctx.settings
        state = await self.ctx.db.get_state("portfolio")
        repositioned = set(state.get("repositioned", []))
        flagged = [p for p in await self.portfolio() if p["health"] in ("reposition", "shutdown")]
        for p in flagged:
            if p["health"] == "reposition" and p["id"] not in repositioned and growth is not None:
                async with self.ctx.db.session() as ses:
                    prod = await ses.get(Product, p["id"])
                if prod:
                    await self.log("audit", f"{p['name']}: {p['days_live']} days live, 0 sales → reposition (new copy angle)")
                    await growth.create_kit(prod, angle="Reposition: the first copy did not sell in 21 days. Lead with a different "
                                                        "pain and a sharper outcome; keep every number given.")
                    repositioned.add(p["id"])
            elif p["health"] == "shutdown":
                if s.autopause_stale and sales is not None:
                    await sales.pause(p["id"])
                    await self.log("audit", f"{p['name']}: {p['days_live']} days, 0 sales → paused")
                else:
                    await self.log("audit", f"{p['name']}: {p['days_live']} days, 0 sales → recommend shutdown")
        await self.ctx.db.set_state("portfolio", {"repositioned": sorted(repositioned)})
        return flagged

    async def daily_close(self, growth: Any = None, sales: Any = None) -> dict[str, object]:
        s = self.ctx.settings
        await finance.accrue_costs(self.ctx.db, s)
        state = await self.ctx.db.get_state("cfo")
        today = date.today().isoformat()
        if state.get("last_close") == today:
            return state
        total = await finance.pnl(self.ctx.db, s)
        mtd = await finance.pnl(self.ctx.db, s, date.today().replace(day=1))
        margin = f"{mtd['net_revenue_usd'] / mtd['gross_usd']:.0%}" if mtd["gross_usd"] else "n/a, no sales yet"
        text = (f"Daily close — MTD gross ${mtd['gross_usd']:.2f} − fees ${mtd['fees_usd']:.2f} − infra ${mtd['costs_usd']:.2f} "
                f"= profit ${mtd['profit_usd']:.2f} (contribution margin {margin}) · all-time ${total['profit_usd']:.2f} · "
                f"Railway ≈ ${s.railway_monthly_usd:.2f}/month")
        await self.log("close", text, mtd=mtd)
        flagged = await self.audit(growth, sales)
        if state.get("last_close"):  # skip the alert on the very first boot
            extra = f"\n{len(flagged)} product(s) need attention (21+ days without a sale)." if flagged else ""
            await self.ctx.notify.ceo(f"📊 {text}{extra}")
        state["last_close"] = today
        await self.ctx.db.set_state("cfo", state)
        return state
