"""Executive dashboard API (USD): revenue, P&L incl. Railway, sales per day, products, ideas, meetings, sectors."""

from __future__ import annotations

import hmac
from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlalchemy import func, select

from app.agents.base import PERSONAS
from app.config import secret
from app.models import AgentEvent, Idea, Listing, MarketingKit, Meeting, Product, Sale, Signal
from app.services import finance

router = APIRouter(prefix="/api")


def office(request: Request) -> Any:
    return request.app.state.office


def require_admin(request: Request, x_admin_token: str = Header(default="")) -> None:
    expected = secret(request.app.state.settings.admin_token)
    if not expected or expected == "change-me" or not hmac.compare_digest(expected, x_admin_token):
        raise HTTPException(status_code=401, detail="admin token required (set ADMIN_TOKEN)")


def _product_row(p: Product, listings: list[Listing], kit: MarketingKit | None, sales: tuple[int, int]) -> dict[str, Any]:
    return {"id": p.id, "name": p.name, "slug": p.slug, "tagline": p.tagline, "type": p.solution_type,
            "price_usd": p.price_cents / 100, "status": p.status, "created_at": p.created_at.isoformat(),
            "published_at": p.published_at.isoformat() if p.published_at else None, "size_kb": round(p.size_bytes / 1024, 1),
            "files": p.files, "qa": p.qa, "headline": (kit.landing or {}).get("headline") if kit else None,
            "listings": [{"platform": x.platform, "url": x.url, "status": x.status, "note": x.note} for x in listings],
            "sales": sales[0], "revenue_usd": sales[1] / 100, "niche": p.idea.niche if p.idea else ""}


@router.get("/dashboard/summary")
async def summary(request: Request) -> dict[str, Any]:
    st = request.app.state
    db, s = st.db, st.settings
    today = date.today()
    async with db.session() as ses:
        by_status = dict((await ses.execute(select(Product.status, func.count(Product.id)).group_by(Product.status))).all())
        ideas = dict((await ses.execute(select(Idea.status, func.count(Idea.id)).group_by(Idea.status))).all())
    return {
        "all_time": await finance.pnl(db, s), "month": await finance.pnl(db, s, today.replace(day=1)),
        "today": await finance.pnl(db, s, today), "goals": await finance.goals(db, s),
        "products": by_status, "ideas": ideas, "active_products": by_status.get("published", 0),
        "monthly_costs": {"railway_usd": s.railway_monthly_usd, "other_usd": s.other_monthly_usd, "llm_usd": 0.0,
                          "breakdown": {"web": s.railway_web_usd, "postgres": s.railway_postgres_usd,
                                        "redis": s.railway_redis_usd, "n8n": s.railway_n8n_usd}},
        "platforms": s.platforms(), "autopublish": s.autopublish, "llm": st.llm.status(),
        "agents": {k: {"name": p.name, "role": p.role, "color": p.color, "room": p.room, **st.bus.agent_state.get(k, {})}
                   for k, p in PERSONAS.items()},
        "meeting": st.bus.meeting, "last_tick": st.office.last_tick,
    }


@router.get("/dashboard/sales/daily")
async def sales_daily(request: Request, days: int = 30) -> list[dict[str, Any]]:
    return await finance.daily_series(request.app.state.db, max(1, min(days, 365)))


@router.get("/products")
async def products(request: Request) -> list[dict[str, Any]]:
    async with request.app.state.db.session() as ses:
        rows = (await ses.scalars(select(Product).order_by(Product.id.desc()))).unique().all()
        listings = (await ses.scalars(select(Listing))).all()
        kits = {k.product_id: k for k in (await ses.scalars(select(MarketingKit))).all()}
        agg = {pid: (n, g) for pid, n, g in (await ses.execute(
            select(Sale.product_id, func.count(Sale.id), func.coalesce(func.sum(Sale.gross_cents), 0))
            .where(Sale.refunded.is_(False)).group_by(Sale.product_id))).all()}
    return [_product_row(p, [x for x in listings if x.product_id == p.id], kits.get(p.id), agg.get(p.id, (0, 0))) for p in rows]


@router.get("/products/{pid}")
async def product_detail(request: Request, pid: int) -> dict[str, Any]:
    async with request.app.state.db.session() as ses:
        p = await ses.get(Product, pid)
        if not p:
            raise HTTPException(404)
        listings = (await ses.scalars(select(Listing).where(Listing.product_id == pid))).all()
        kit = await ses.scalar(select(MarketingKit).where(MarketingKit.product_id == pid))
        n, g = (await ses.execute(select(func.count(Sale.id), func.coalesce(func.sum(Sale.gross_cents), 0))
                                  .where(Sale.product_id == pid, Sale.refunded.is_(False)))).one()
    row = _product_row(p, list(listings), kit, (n, g))
    row["kit"] = {"landing": kit.landing, "emails": kit.emails, "social": kit.social, "provider": kit.provider} if kit else None
    row["idea"] = {"problem": p.idea.problem, "mvp": p.idea.mvp, "evidence": p.idea.evidence} if p.idea else None
    return row


@router.post("/products/{pid}/approve", dependencies=[Depends(require_admin)])
async def approve(request: Request, pid: int) -> dict[str, Any]:
    async with request.app.state.db.session() as ses:
        p = await ses.get(Product, pid)
        if not p or p.status not in ("ready", "paused"):
            raise HTTPException(409, "product is not ready for publishing")
    await request.app.state.queue.put({"job": "publish", "product_id": pid})
    return {"queued": True}


@router.post("/run/{job}", dependencies=[Depends(require_admin)])
async def run(request: Request, job: str) -> dict[str, Any]:
    if job not in ("tick", "research"):
        raise HTTPException(400, "job must be tick or research")
    await request.app.state.queue.put({"job": job})
    return {"queued": job}


@router.get("/ideas")
async def ideas(request: Request, status: str = "", limit: int = 50) -> list[dict[str, Any]]:
    q = select(Idea).order_by(Idea.id.desc()).limit(min(limit, 200))
    if status:
        q = q.where(Idea.status == status)
    async with request.app.state.db.session() as ses:
        rows = (await ses.scalars(q)).all()
    return [{"id": i.id, "topic": i.topic, "title": i.title, "problem": i.problem, "niche": i.niche, "type": i.solution_type,
             "mvp": i.mvp, "price_usd": i.price_cents / 100, "score": i.score, "status": i.status, "evidence": i.evidence,
             "rationale": i.rationale, "provider": i.provider, "assessment": i.assessment or {},
             "created_at": i.created_at.isoformat()} for i in rows]


@router.get("/meetings")
async def meetings(request: Request, limit: int = 10) -> list[dict[str, Any]]:
    async with request.app.state.db.session() as ses:
        rows = (await ses.scalars(select(Meeting).order_by(Meeting.id.desc()).limit(min(limit, 50)))).all()
    return [{"id": m.id, "kind": m.kind, "subject": m.subject, "decision": m.decision, "summary": m.summary,
             "started_at": m.started_at.isoformat(), "ended_at": m.ended_at.isoformat() if m.ended_at else None,
             "messages": [{"agent": x.agent, "text": x.text, "vote": x.vote, "data": x.data, "at": x.created_at.isoformat()}
                          for x in m.messages]} for m in rows]


@router.get("/activity")
async def activity(request: Request, agent: str = "", limit: int = 60) -> list[dict[str, Any]]:
    q = select(AgentEvent).order_by(AgentEvent.id.desc()).limit(min(limit, 300))
    if agent:
        q = q.where(AgentEvent.agent == agent)
    async with request.app.state.db.session() as ses:
        rows = (await ses.scalars(q)).all()
    return [{"agent": e.agent, "kind": e.kind, "text": e.text, "data": e.data, "at": e.created_at.isoformat()} for e in rows]


@router.get("/signals")
async def signals(request: Request, limit: int = 40) -> list[dict[str, Any]]:
    async with request.app.state.db.session() as ses:
        rows = (await ses.scalars(select(Signal).order_by(Signal.id.desc()).limit(min(limit, 200)))).all()
    return [{"source": x.source, "title": x.title, "url": x.url, "topic": x.topic, "engagement": x.engagement,
             "intent": x.intent, "assessment": x.assessment or {}, "at": x.created_at.isoformat()} for x in rows]


@router.get("/sectors")
async def sectors(request: Request) -> dict[str, Any]:
    """One panel per department of the office."""
    st = request.app.state
    db = st.db
    async with db.session() as ses:
        sig_src = dict((await ses.execute(select(Signal.source, func.count(Signal.id)).group_by(Signal.source))).all())
        idea_st = dict((await ses.execute(select(Idea.status, func.count(Idea.id)).group_by(Idea.status))).all())
        prods = (await ses.scalars(select(Product))).unique().all()
        kits = (await ses.scalars(select(MarketingKit))).all()
        listings = (await ses.scalars(select(Listing))).all()
        by_plat = {p: (n, g) for p, n, g in (await ses.execute(select(Sale.platform, func.count(Sale.id),
                                                                      func.coalesce(func.sum(Sale.gross_cents), 0))
                                                               .where(Sale.refunded.is_(False)).group_by(Sale.platform))).all()}
        hot = await ses.scalar(select(func.count(Signal.id)).where(Signal.assessment["urgency"].as_string() == "hair_on_fire")) or 0
        discarded = await ses.scalar(select(func.count(Idea.id)).where(Idea.provider == "frameworks")) or 0
        undelivered = await ses.scalar(select(func.count(Sale.id)).where(Sale.delivered.is_(False), Sale.product_id.isnot(None))) or 0
    topics = await st.office.product.topic_scores()
    qa_ok = sum(1 for p in prods if p.qa.get("ok"))
    return {
        "product": {"signals_by_source": sig_src, "ideas_by_status": idea_st,
                    "top_topics": sorted(({"topic": k, **v} for k, v in topics.items()), key=lambda x: -x["score"])[:6],
                    "focus": (await db.get_state("strategy")).get("focus_topics", []), "hair_on_fire": hot, "discarded": discarded},
        "builder": {"built": len(prods), "qa_pass": qa_ok, "n8n": sum(p.solution_type == "n8n" for p in prods),
                    "python": sum(p.solution_type == "python" for p in prods),
                    "total_kb": round(sum(p.size_bytes for p in prods) / 1024, 1)},
        "growth": {"kits": len(kits), "emails": sum(len(k.emails) for k in kits),
                   "posts": sum(len(k.social.get("x", [])) + len(k.social.get("linkedin", [])) + len(k.social.get("reddit", [])) for k in kits)},
        "sales": {"listings_live": sum(x.status == "live" for x in listings), "listings_manual": sum(x.status == "manual" for x in listings),
                  "listings_error": sum(x.status == "error" for x in listings), "undelivered": undelivered,
                  "by_platform": {k: {"sales": n, "gross_usd": g / 100} for k, (n, g) in by_plat.items()},
                  "checkout": {k: v for k, v in (await db.get_state("checkout")).items() if k != "recovered_to"},
                  "funnel": await st.office.sales.funnel(14)},
        "cfo": {**await finance.goals(db, st.settings), "portfolio": await st.office.cfo.portfolio(),
                "topic_margins": await st.office.cfo.topic_margins(30), "topic_roi": await st.office.cfo.topic_roi(30),
                "ltv_cac": await finance.ltv_cac(db, st.settings), "infra": await db.get_state("infra"),
                "pnl_month": await finance.pnl(db, st.settings, date.today().replace(day=1))},
    }
