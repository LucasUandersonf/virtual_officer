"""P&L in USD: revenue from real sales minus platform fees minus the cost of running the office.

Fixed costs (Railway services + other subscriptions) accrue one row per day (monthly / days in month),
so the dashboard shows the true daily burn even before the first sale. LLM cost is zero (free tiers),
and is still counted as a line so the CEO sees it.
"""

from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import func, select

from app.config import Settings
from app.db import Database
from app.models import CostEntry, Idea, LLMCall, Product, Sale


def _cents(usd: float) -> int:
    return round(usd * 100)


async def accrue_costs(db: Database, s: Settings, day: date | None = None) -> int:
    """Idempotent: one row per category and day. Returns cents added."""
    day = day or date.today()
    days = calendar.monthrange(day.year, day.month)[1]
    rows = [("railway", s.railway_monthly_usd, "Railway: web + Postgres + Redis + n8n (estimate)"),
            ("other", s.other_monthly_usd, "Other fixed costs"), ("llm", 0.0, "LLMs on free tiers")]
    added = 0
    async with db.session() as ses:
        for cat, monthly, note in rows:
            exists = await ses.scalar(select(CostEntry.id).where(CostEntry.category == cat, CostEntry.day == day))
            if exists:
                continue
            amount = _cents(monthly / days)
            ses.add(CostEntry(category=cat, day=day, amount_cents=amount, note=note))
            added += amount
    return added


async def pnl(db: Database, s: Settings, since: date | None = None) -> dict[str, Any]:
    async with db.session() as ses:
        q = select(func.coalesce(func.sum(Sale.gross_cents), 0), func.coalesce(func.sum(Sale.fee_cents), 0), func.count(Sale.id)
                   ).where(Sale.refunded.is_(False))
        c = select(CostEntry.category, func.sum(CostEntry.amount_cents)).group_by(CostEntry.category)
        r = select(func.coalesce(func.sum(Sale.gross_cents), 0)).where(Sale.refunded.is_(True))
        if since:
            start = datetime.combine(since, datetime.min.time())
            q, r = q.where(Sale.created_at >= start), r.where(Sale.created_at >= start)
            c = c.where(CostEntry.day >= since)
        gross, fees, n = (int(v) for v in (await ses.execute(q)).one())  # Postgres SUM() returns Decimal
        refunds = int(await ses.scalar(r) or 0)
        costs = {k: int(v) for k, v in (await ses.execute(c)).all()}
    total_costs = sum(costs.values())
    net = int(gross) - int(fees)
    return {"gross_usd": gross / 100, "fees_usd": fees / 100, "net_revenue_usd": net / 100, "sales": int(n),
            "refunds_usd": refunds / 100, "costs_usd": total_costs / 100, "costs_by_category_usd": {k: v / 100 for k, v in costs.items()},
            "profit_usd": (net - total_costs) / 100}


async def daily_series(db: Database, days: int = 30) -> list[dict[str, Any]]:
    start = date.today() - timedelta(days=days - 1)
    async with db.session() as ses:
        sales = (await ses.execute(select(Sale.created_at, Sale.gross_cents, Sale.fee_cents).where(
            Sale.refunded.is_(False), Sale.created_at >= datetime.combine(start, datetime.min.time())))).all()
        costs = (await ses.execute(select(CostEntry.day, func.sum(CostEntry.amount_cents)).where(
            CostEntry.day >= start).group_by(CostEntry.day))).all()
    by_day = {start + timedelta(days=i): {"revenue": 0, "fees": 0, "sales": 0, "costs": 0} for i in range(days)}
    for ts, g, f in sales:
        d = by_day.get(ts.date())
        if d:
            d["revenue"] += g
            d["fees"] += f
            d["sales"] += 1
    for d0, amt in costs:
        if d0 in by_day:
            by_day[d0]["costs"] += int(amt)
    out = []
    for d0, v in sorted(by_day.items()):
        out.append({"day": d0.isoformat(), "sales": v["sales"], "revenue_usd": v["revenue"] / 100,
                    "costs_usd": (v["costs"] + v["fees"]) / 100, "profit_usd": (v["revenue"] - v["fees"] - v["costs"]) / 100})
    return out


async def goals(db: Database, s: Settings) -> dict[str, Any]:
    today = date.today()
    month_start = today.replace(day=1)
    week_start = today - timedelta(days=today.weekday())
    days_in_month = calendar.monthrange(today.year, today.month)[1]
    mtd = await pnl(db, s, month_start)
    async with db.session() as ses:
        wk = datetime.combine(week_start, datetime.min.time())
        products_week = await ses.scalar(select(func.count(Product.id)).where(Product.created_at >= wk)) or 0
        ideas_week = await ses.scalar(select(func.count(Idea.id)).where(Idea.created_at >= wk)) or 0
        live = await ses.scalar(select(func.count(Product.id)).where(Product.status == "published")) or 0
        llm_today = await ses.scalar(select(func.count(LLMCall.id)).where(
            LLMCall.created_at >= datetime.combine(today, datetime.min.time()))) or 0
    run_rate = mtd["net_revenue_usd"] / today.day * days_in_month
    avg_price = 45.0
    monthly_cost = s.monthly_fixed_costs_usd
    return {
        "revenue": {"label": "Monthly net revenue", "current": mtd["net_revenue_usd"], "target": s.goal_monthly_revenue_usd,
                    "run_rate": round(run_rate, 2), "pct": min(1.0, mtd["net_revenue_usd"] / max(s.goal_monthly_revenue_usd, 1))},
        "profit_mtd": mtd["profit_usd"],
        "break_even": {"monthly_cost_usd": monthly_cost, "sales_needed": int(-(-monthly_cost // (avg_price * 0.9))) if monthly_cost else 0,
                       "covered": mtd["net_revenue_usd"] >= monthly_cost * today.day / days_in_month},
        "products_week": {"current": products_week, "target": s.goal_products_per_week},
        "ideas_week": {"current": ideas_week, "target": s.goal_ideas_per_week},
        "live_products": live, "llm_calls_today": llm_today, "llm_budget": s.llm_daily_call_budget,
    }
