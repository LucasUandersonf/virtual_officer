"""Typed domain models shared by every layer (quant, agents, risk, ws)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- quant


class OrderBookMetrics(BaseModel):
    best_bid: float
    best_ask: float
    mid: float
    spread_pct: float
    bid_volume: float
    ask_volume: float
    imbalance: float = Field(description="(bid-ask)/(bid+ask) on top N levels, in [-1, 1]")
    bid_ask_ratio: float
    depth_delta: float = Field(description="Change in imbalance vs previous snapshot")
    microprice: float


class VolumeProfile(BaseModel):
    poc: float
    value_area_high: float
    value_area_low: float


class Forecast(BaseModel):
    degree: int
    horizon: int
    r_squared: float
    slope_per_bar: float
    expected_price: float
    lower_95: float
    upper_95: float
    expected_return_pct: float


class QuantSnapshot(BaseModel):
    ts: datetime = Field(default_factory=utcnow)
    symbol: str
    price: float
    zscore: float
    rolling_mean: float
    rolling_std: float
    vwap: float
    vwap_upper: float
    vwap_lower: float
    volume_profile: VolumeProfile
    atr: float
    atr_pct: float
    orderbook: OrderBookMetrics
    forecast: Forecast


class SetupStats(BaseModel):
    """Statistical evidence of the mean-reversion setup (backtest + live)."""

    backtest_trades: int
    backtest_wins: int
    live_trades: int
    live_wins: int
    win_rate: float = Field(description="Beta-Binomial posterior mean")
    win_rate_lower_95: float
    payoff_ratio: float
    expected_return_pct: float
    trade_sharpe: float
    kelly_full: float
    kelly_applied: float


class ExitProfile(BaseModel):
    """Bracket exit inside the mandate: TP in [0.6%, 0.8%], ATR stop clamped to [0.3%, 0.6%]."""

    model_config = ConfigDict(frozen=True)

    tp_pct: float
    stop_atr_mult: float


class SetupMetrics(BaseModel):
    """Per-trade statistics, net of fees and slippage, in fractions of notional."""

    trades: int = 0
    wins: int = 0
    win_rate: float = Field(default=0.5, description="Beta(1+w, 1+l) posterior mean")
    win_rate_lower_95: float = 0.0
    expectancy_pct: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    total_return_pct: float = 0.0
    p_edge: float = Field(default=0.0, description="P(win rate > break-even) under the posterior")


class LeagueEntry(BaseModel):
    key: str
    name: str
    family: str
    thesis: str
    profile: ExitProfile
    oos: SetupMetrics = Field(description="Walk-forward out-of-sample: the number that matters")
    in_sample: SetupMetrics
    folds: list[float] = Field(default_factory=list, description="OOS net return per test fold")
    approved: bool = False
    reasons: list[str] = Field(default_factory=list)
    score: float = 0.0
    rank: int = 0
    last_signal: datetime | None = None


class LeagueReport(BaseModel):
    generated_at: datetime = Field(default_factory=utcnow)
    symbol: str
    timeframe: str
    bars: int
    start: datetime | None = None
    end: datetime | None = None
    folds: int
    max_hold_bars: int
    entries: list[LeagueEntry]

    @property
    def approved(self) -> list[LeagueEntry]:
        return [e for e in self.entries if e.approved]


class FiredSetup(BaseModel):
    """A setup whose entry rule fired on the last closed candle."""

    key: str
    name: str
    family: str
    thesis: str
    profile: ExitProfile
    oos: SetupMetrics
    score: float
    approved: bool
    bar_ts: datetime


class SizingPlan(BaseModel):
    equity: float
    risk_budget_usdt: float
    entry_price: float
    stop_price: float
    take_profit_price: float
    stop_pct: float
    take_profit_pct: float
    quantity: float
    notional: float
    max_loss_usdt: float = Field(description="Loss at stop incl. fees and slippage")
    expected_gain_usdt: float


# --------------------------------------------------------------------------- agents


class AgentRole(str, Enum):
    CHIEF_QUANT = "chief_quant"
    ECONOMETRICIAN = "econometrician"
    CRO = "cro"
    EXECUTION = "execution"
    SYSTEM = "system"


class Vote(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    ABSTAIN = "ABSTAIN"


class Scenario(BaseModel):
    name: Literal["bullish", "neutral", "bearish"]
    probability: float = Field(ge=0, le=1)
    target_price: float | None = None
    rationale: str = ""


class ChiefQuantOpinion(BaseModel):
    scenarios: list[Scenario] = Field(min_length=3, max_length=3)
    thesis: str
    vote: Vote


class EconometricianOpinion(BaseModel):
    assessment: str
    vote: Vote


class FailureMode(BaseModel):
    risk: str
    likelihood: Literal["low", "medium", "high"]
    mitigation: str = ""


class CroOpinion(BaseModel):
    pre_mortem: list[FailureMode] = Field(min_length=1)
    verdict: str
    vote: Vote


class ExecutionOpinion(BaseModel):
    plan: str
    vote: Vote


class AgentMessage(BaseModel):
    """One speech bubble in the virtual office."""

    debate_id: str | None = None
    role: AgentRole
    agent_name: str
    text: str
    vote: Vote | None = None
    provider: str | None = None  # which LLM produced it ("rules" if deterministic)
    data: dict[str, Any] = Field(default_factory=dict)
    ts: datetime = Field(default_factory=utcnow)


class GateResult(BaseModel):
    name: str
    passed: bool
    detail: str


class Decision(BaseModel):
    debate_id: str
    approved: bool
    reason: str
    gates: list[GateResult]
    votes: dict[str, Vote]
    sizing: SizingPlan | None = None
    setup: str | None = None
    confluence: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- trading


class PositionStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class ExitReason(str, Enum):
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    MANUAL = "manual"
    ENTRY_TIMEOUT = "entry_timeout"
    TIME_STOP = "time_stop"


class Position(BaseModel):
    id: int | None = None
    debate_id: str
    mode: str
    symbol: str
    status: PositionStatus
    quantity: float
    entry_price: float
    stop_price: float
    take_profit_price: float
    opened_at: datetime = Field(default_factory=utcnow)
    closed_at: datetime | None = None
    exit_price: float | None = None
    exit_reason: ExitReason | None = None
    fees_usdt: float = 0.0
    pnl_usdt: float | None = None
    entry_order_id: str | None = None
    stop_order_id: str | None = None
    setup: str | None = None


# --------------------------------------------------------------------------- websocket


EventType = Literal[
    "hello", "telemetry", "agent_message", "debate_start", "decision", "position", "risk", "system",
    "league",
]


class WsEvent(BaseModel):
    type: EventType
    ts: datetime = Field(default_factory=utcnow)
    payload: dict[str, Any]
