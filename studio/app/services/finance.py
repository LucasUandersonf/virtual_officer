"""P&L in USD: revenue from real sales minus platform fees minus the cost of running the office.

Fixed costs (Railway services + other subscriptions) accrue one row per day (monthly / days in month),
so the dashboard shows the true daily burn even before the first sale. LLM cost is zero (free tiers),
and is still counted as a line so the CEO sees it.
"""

from __future__ import annotations

import calendar
import math
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import func, select

from app.config import Settings
from app.db import Database
from app.models import CostEntry, Idea, LLMCall, Product, Sale


def _cents(usd: float) -> int:
    return round(usd * 100)


async def accrue_costs(db: Database, s: Settings, day: date | None = None, railway_month_usd: float | None = None) -> int:
    """Idempotent: one row per category and day. Returns cents added. ``railway_month_usd`` = measured usage when known."""
    day = day or date.today()
    days = calendar.monthrange(day.year, day.month)[1]
    measured = railway_month_usd is not None
    rows = [("railway", railway_month_usd if measured else s.railway_monthly_usd,
             "Railway usage (measured via API)" if measured else "Railway: web + Postgres + Redis + n8n (estimate)"),
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


# Conservative tax provisions by Stripe account country, used only when TAX_RATE_PCT is not set. A reserve, not tax
# advice: BR = Simples Nacional, first bracket of Anexo III (6 % of revenue); confirm the real rate with an accountant.
TAX_PROVISION = {"BR": 0.06}


ZERO_DECIMAL = {"bif", "clp", "djf", "gnf", "jpy", "kmf", "krw", "mga", "pyg", "rwf", "ugx", "vnd", "vuv", "xaf", "xof", "xpf"}


def from_minor(amount: int, currency: str) -> float:
    return float(amount) if currency.lower() in ZERO_DECIMAL else amount / 100


async def usd_rate(client: Any, currency: str) -> float | None:
    """Units of ``currency`` per 1 USD (free open.er-api.com, no key). None when unavailable."""
    if currency.lower() == "usd":
        return 1.0
    try:
        r = await client.get("https://open.er-api.com/v6/latest/USD")
        return float(r.json()["rates"][currency.upper()])
    except Exception:
        return None


async def fin_params(db: Database, s: Settings) -> dict[str, Any]:
    """Effective payout currency, tax rate and cash: explicit settings win, else what Nina discovered."""
    auto = await db.get_state("autoconfig_finance")
    payout = (s.payout_currency or auto.get("payout_currency") or "usd").lower()
    if s.tax_rate_pct is not None:
        tax, tax_src = s.tax_rate_pct, "TAX_RATE_PCT"
    elif auto.get("country") in TAX_PROVISION:
        tax, tax_src = TAX_PROVISION[auto["country"]], f"provision for {auto['country']} (Nina)"
    else:
        tax, tax_src = 0.0, "not modeled (set TAX_RATE_PCT)"
    if s.cash_balance_usd is not None:
        cash, cash_src = s.cash_balance_usd, "CASH_BALANCE_USD"
    elif auto.get("cash_usd") is not None:
        cash, cash_src = float(auto["cash_usd"]), "Stripe balance (Nina)"
    else:
        cash, cash_src = None, "unknown"
    return {"payout_currency": payout, "payout_source": "PAYOUT_CURRENCY" if s.payout_currency else
            ("Stripe account (Nina)" if auto.get("payout_currency") else "default"), "tax_rate": tax, "tax_source": tax_src,
            "cash_usd": cash, "cash_source": cash_src, "country": auto.get("country", "")}


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
    fp = await fin_params(db, s)
    fx = round(net * s.fx_conversion_pct) if fp["payout_currency"] != "usd" else 0
    tax = round(gross * fp["tax_rate"])
    return {"gross_usd": gross / 100, "fees_usd": fees / 100, "net_revenue_usd": net / 100, "sales": int(n),
            "refunds_usd": refunds / 100, "costs_usd": total_costs / 100, "costs_by_category_usd": {k: v / 100 for k, v in costs.items()},
            "fx_usd": fx / 100, "tax_usd": tax / 100, "payout_usd": (net - fx) / 100,
            # Net profit = gross - checkout fees - infrastructure (Railway + LLM); after FX and tax when modeled
            "profit_usd": (net - total_costs) / 100, "profit_after_tax_usd": (net - fx - tax - total_costs) / 100}


async def avg_net_per_sale(db: Database, s: Settings) -> tuple[float, bool]:
    """Real average net per sale (after fees); falls back to a $49 list price when nothing sold yet."""
    async with db.session() as ses:
        n, g, f = (await ses.execute(select(func.count(Sale.id), func.coalesce(func.sum(Sale.gross_cents), 0),
                                            func.coalesce(func.sum(Sale.fee_cents), 0)).where(Sale.refunded.is_(False)))).one()
    if int(n):
        return (int(g) - int(f)) / 100 / int(n), True
    return 49.0 * (1 - s.stripe_fee_pct) - s.stripe_fee_fixed_usd, False


async def ltv_cac(db: Database, s: Settings) -> dict[str, Any]:
    """LTV = average order value x orders per buyer; CAC = paid acquisition / new buyers (30 days)."""
    since = datetime.now() - timedelta(days=30)
    async with db.session() as ses:
        rows = (await ses.execute(select(Sale.buyer_email, func.count(Sale.id), func.sum(Sale.gross_cents - Sale.fee_cents))
                                  .where(Sale.refunded.is_(False), Sale.buyer_email != "").group_by(Sale.buyer_email))).all()
        new_buyers = await ses.scalar(select(func.count(func.distinct(Sale.buyer_email))).where(
            Sale.refunded.is_(False), Sale.created_at >= since, Sale.buyer_email != "")) or 0
    buyers = len(rows)
    orders = sum(int(r[1]) for r in rows)
    net = sum(int(r[2] or 0) for r in rows) / 100
    aov = net / orders if orders else 0.0
    ltv = net / buyers if buyers else 0.0
    cac = s.ad_spend_monthly_usd / new_buyers if new_buyers and s.ad_spend_monthly_usd else 0.0
    return {"buyers": buyers, "orders": orders, "repeat_rate": round((orders - buyers) / buyers, 3) if buyers else 0.0,
            "aov_usd": round(aov, 2), "ltv_usd": round(ltv, 2), "cac_usd": round(cac, 2),
            "ltv_cac": round(ltv / cac, 1) if cac else None, "organic_only": s.ad_spend_monthly_usd == 0}


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
    avg_net, real = await avg_net_per_sale(db, s)
    monthly_cost = s.monthly_fixed_costs_usd
    burn = monthly_cost - run_rate
    fp = await fin_params(db, s)
    # explicit cash is a starting amount (add the studio's profit since); the Stripe balance already includes sales
    cash_known = fp["cash_usd"] is not None
    cash = (fp["cash_usd"] or 0.0) + ((await pnl(db, s))["profit_usd"] if fp["cash_source"] == "CASH_BALANCE_USD" else 0.0)
    return {
        "revenue": {"label": "Monthly net revenue", "current": mtd["net_revenue_usd"], "target": s.goal_monthly_revenue_usd,
                    "run_rate": round(run_rate, 2), "pct": min(1.0, mtd["net_revenue_usd"] / max(s.goal_monthly_revenue_usd, 1))},
        "profit_mtd": mtd["profit_usd"],
        "break_even": {"monthly_cost_usd": monthly_cost, "avg_net_per_sale_usd": round(avg_net, 2), "from_real_sales": real,
                       "sales_needed": math.ceil(monthly_cost / avg_net) if monthly_cost and avg_net > 0 else 0,
                       "covered": mtd["net_revenue_usd"] >= monthly_cost * today.day / days_in_month},
        "runway": {"cash_usd": round(cash, 2), "monthly_burn_usd": round(max(burn, 0.0), 2), "cash_set": cash_known,
                   "cash_source": fp["cash_source"], "payout_currency": fp["payout_currency"], "tax_rate": fp["tax_rate"],
                   "tax_source": fp["tax_source"],
                   "months": None if burn <= 0 else round(max(cash, 0.0) / burn, 1)},
        "products_week": {"current": products_week, "target": s.goal_products_per_week},
        "ideas_week": {"current": ideas_week, "target": s.goal_ideas_per_week},
        "live_products": live, "llm_calls_today": llm_today, "llm_budget": s.llm_daily_call_budget,
    }
