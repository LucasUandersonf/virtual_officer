"""Agent 5 — Nina, CFO & Goals: unit economics per idea, daily close, weekly goals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy import func, select

from app.agents.base import Agent
from app.models import Idea, Product, utcnow
from app.services import finance


@dataclass(frozen=True)
class Economics:
    est_monthly_sales: float
    est_monthly_net_usd: float
    cost_share_usd: float
    payback_ratio: float  # expected net / cost share; >= 1 pays its share of the office
    builds_left_today: int

    @property
    def ok(self) -> bool:
        return self.payback_ratio >= 1.0 and self.builds_left_today > 0


class CFO(Agent):
    key = "cfo"

    async def economics(self, idea: Idea) -> Economics:
        s = self.ctx.settings
        start = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        async with self.ctx.db.session() as ses:
            live = await ses.scalar(select(func.count(Product.id)).where(Product.status.in_(["published", "ready", "built"]))) or 0
            built_today = await ses.scalar(select(func.count(Product.id)).where(Product.created_at >= start)) or 0
        # Conservative demand model: half a sale a month baseline plus ~0.25 per point of evidence score, capped at 10.
        # The real numbers replace this as soon as sales come in (topic priors in the Head of Product's scoring).
        est = min(10.0, 0.5 + 0.25 * max(idea.score, 0))
        net = est * idea.price_cents / 100 * 0.9
        share = s.monthly_fixed_costs_usd / (live + 1) if s.monthly_fixed_costs_usd else 0.0
        ratio = net / share if share else 99.0
        return Economics(round(est, 2), round(net, 2), round(share, 2), round(ratio, 2), max(0, s.max_builds_per_day - built_today))

    async def daily_close(self) -> dict[str, object]:
        s = self.ctx.settings
        await finance.accrue_costs(self.ctx.db, s)
        state = await self.ctx.db.get_state("cfo")
        today = date.today().isoformat()
        if state.get("last_close") == today:
            return state
        total = await finance.pnl(self.ctx.db, s)
        mtd = await finance.pnl(self.ctx.db, s, date.today().replace(day=1))
        text = (f"Daily close — MTD net ${mtd['net_revenue_usd']:.2f}, costs ${mtd['costs_usd']:.2f}, profit ${mtd['profit_usd']:.2f} · "
                f"all-time profit ${total['profit_usd']:.2f} · Railway ≈ ${s.railway_monthly_usd:.2f}/month")
        await self.log("close", text, mtd=mtd)
        if state.get("last_close"):  # skip the alert on the very first boot
            await self.ctx.notify.ceo(f"📊 {text}")
        state["last_close"] = today
        await self.ctx.db.set_state("cfo", state)
        return state
