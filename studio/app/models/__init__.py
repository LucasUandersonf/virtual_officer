"""Database schema (SQLAlchemy 2.0). Money is always integer US cents."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class Signal(Base):
    """One post/comment that shows a pain (Reddit, Hacker News, Product Hunt, built-in catalogue)."""

    __tablename__ = "signals"
    __table_args__ = (UniqueConstraint("source", "external_id"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(32))
    external_id: Mapped[str] = mapped_column(String(128))
    title: Mapped[str] = mapped_column(String(500))
    url: Mapped[str] = mapped_column(String(1000), default="")
    text: Mapped[str] = mapped_column(Text, default="")
    topic: Mapped[str] = mapped_column(String(64), default="", index=True)
    engagement: Mapped[int] = mapped_column(Integer, default=0)
    intent: Mapped[float] = mapped_column(Float, default=0.0)
    # Iris's post-level frameworks: urgency, money/time loss, workaround steps, WTP, opportunity score
    assessment: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Idea(Base):
    __tablename__ = "ideas"
    id: Mapped[int] = mapped_column(primary_key=True)
    topic: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(200))
    problem: Mapped[str] = mapped_column(Text)
    niche: Mapped[str] = mapped_column(String(200))
    solution_type: Mapped[str] = mapped_column(String(16))  # n8n | python
    mvp: Mapped[str] = mapped_column(Text)
    price_cents: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    evidence: Mapped[list[Any]] = mapped_column(default=list)
    # proposed -> approved | rejected -> building -> built | failed
    status: Mapped[str] = mapped_column(String(16), default="proposed", index=True)
    rationale: Mapped[str] = mapped_column(Text, default="")
    provider: Mapped[str] = mapped_column(String(32), default="")
    # Iris's idea-level verdict (opportunity, urgency, TTV, distribution, moat, value anchor, MVP score)
    assessment: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Product(Base):
    __tablename__ = "products"
    id: Mapped[int] = mapped_column(primary_key=True)
    idea_id: Mapped[int] = mapped_column(ForeignKey("ideas.id"))
    name: Mapped[str] = mapped_column(String(200))
    slug: Mapped[str] = mapped_column(String(120), unique=True)
    tagline: Mapped[str] = mapped_column(String(300), default="")
    solution_type: Mapped[str] = mapped_column(String(16))
    price_cents: Mapped[int] = mapped_column(Integer)
    # built -> marketing -> ready -> published | paused ; qa_failed
    status: Mapped[str] = mapped_column(String(16), default="built", index=True)
    artifact_path: Mapped[str] = mapped_column(String(500), default="")
    artifact_sha256: Mapped[str] = mapped_column(String(64), default="")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    files: Mapped[list[Any]] = mapped_column(default=list)
    qa: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    published_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    idea: Mapped[Idea] = relationship(lazy="joined")


class MarketingKit(Base):
    __tablename__ = "marketing_kits"
    id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), unique=True)
    landing: Mapped[dict[str, Any]] = mapped_column(default=dict)
    emails: Mapped[list[Any]] = mapped_column(default=list)
    social: Mapped[dict[str, Any]] = mapped_column(default=dict)
    provider: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Listing(Base):
    __tablename__ = "listings"
    __table_args__ = (UniqueConstraint("platform", "product_id"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    platform: Mapped[str] = mapped_column(String(16))  # stripe | gumroad | whop
    external_id: Mapped[str] = mapped_column(String(200), default="", index=True)
    url: Mapped[str] = mapped_column(String(1000), default="")
    status: Mapped[str] = mapped_column(String(16), default="live")  # live | manual | error | paused
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Sale(Base):
    __tablename__ = "sales"
    __table_args__ = (UniqueConstraint("platform", "external_id"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    platform: Mapped[str] = mapped_column(String(16))
    external_id: Mapped[str] = mapped_column(String(200))
    product_id: Mapped[int | None] = mapped_column(ForeignKey("products.id"), nullable=True, index=True)
    buyer_email: Mapped[str] = mapped_column(String(320), default="")
    gross_cents: Mapped[int] = mapped_column(Integer)
    fee_cents: Mapped[int] = mapped_column(Integer, default=0)
    fee_estimated: Mapped[bool] = mapped_column(Boolean, default=True)
    refunded: Mapped[bool] = mapped_column(Boolean, default=False)
    delivered: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)

    @property
    def net_cents(self) -> int:
        return 0 if self.refunded else self.gross_cents - self.fee_cents


class CostEntry(Base):
    """Daily accrual of fixed costs (Railway services, other subscriptions)."""

    __tablename__ = "costs"
    __table_args__ = (UniqueConstraint("category", "day"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    category: Mapped[str] = mapped_column(String(32))  # railway | other | llm
    day: Mapped[date] = mapped_column(Date, index=True)
    amount_cents: Mapped[int] = mapped_column(Integer)
    note: Mapped[str] = mapped_column(String(300), default="")


class Meeting(Base):
    __tablename__ = "meetings"
    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(24))  # idea_review | weekly_goals
    subject: Mapped[str] = mapped_column(String(300))
    idea_id: Mapped[int | None] = mapped_column(ForeignKey("ideas.id"), nullable=True)
    decision: Mapped[str] = mapped_column(String(300), default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    messages: Mapped[list[MeetingMessage]] = relationship(order_by="MeetingMessage.id", lazy="selectin")


class MeetingMessage(Base):
    __tablename__ = "meeting_messages"
    id: Mapped[int] = mapped_column(primary_key=True)
    meeting_id: Mapped[int] = mapped_column(ForeignKey("meetings.id"), index=True)
    agent: Mapped[str] = mapped_column(String(16))
    text: Mapped[str] = mapped_column(Text)
    vote: Mapped[str] = mapped_column(String(12), default="")
    data: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class AgentEvent(Base):
    """Activity feed: what each agent did, in order."""

    __tablename__ = "agent_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    agent: Mapped[str] = mapped_column(String(16), index=True)
    kind: Mapped[str] = mapped_column(String(24))
    text: Mapped[str] = mapped_column(Text)
    data: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class LLMCall(Base):
    __tablename__ = "llm_calls"
    id: Mapped[int] = mapped_column(primary_key=True)
    agent: Mapped[str] = mapped_column(String(16))
    provider: Mapped[str] = mapped_column(String(24))
    model: Mapped[str] = mapped_column(String(120))
    ok: Mapped[bool] = mapped_column(Boolean)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class Download(Base):
    __tablename__ = "downloads"
    token: Mapped[str] = mapped_column(String(64), primary_key=True)
    sale_id: Mapped[int] = mapped_column(ForeignKey("sales.id"))
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"))
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    count: Mapped[int] = mapped_column(Integer, default=0)
    max_count: Mapped[int] = mapped_column(Integer, default=10)


class OfficeState(Base):
    """Small key/value store for strategy decided in meetings (focus niches, last runs)."""

    __tablename__ = "office_state"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict[str, Any]] = mapped_column(default=dict)
