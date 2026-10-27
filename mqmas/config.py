"""Centralised, validated runtime configuration.

Every tunable of the system lives here and is loaded from environment
variables (or a ``.env`` file). Risk parameters are *validated* at boot: a
configuration that would allow more risk than the mandate permits refuses to
start instead of silently trading.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LIVE_CONFIRMATION_PHRASE = "I_UNDERSTAND_THE_RISKS"

# Absolute ceilings of the mandate. Env vars may tighten them, never loosen.
MANDATE_MAX_RISK_PER_TRADE = 0.006
MANDATE_MAX_STOP_PCT = 0.006


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class MarketDataSource(str, Enum):
    EXCHANGE = "exchange"
    SYNTHETIC = "synthetic"  # offline demo feed (GBM); paper mode only


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Runtime --------------------------------------------------------------
    app_name: str = "MQ-MAS"
    log_level: str = "INFO"
    port: int = Field(default=8000, alias="PORT")
    admin_token: SecretStr | None = None  # protects POST /api/kill-switch
    db_path: Path = Path("./data/mqmas.db")

    # --- Market ---------------------------------------------------------------
    exchange_id: str = "binance"  # any ccxt id with spot support: binance, bybit
    symbol: str = "BTC/USDT"
    timeframe: str = "15m"
    ohlcv_limit: int = Field(default=1000, ge=200, le=1500)
    market_data_source: MarketDataSource = MarketDataSource.EXCHANGE
    exchange_api_key: SecretStr | None = None
    exchange_api_secret: SecretStr | None = None
    exchange_sandbox: bool = False

    # --- Trading mode -----------------------------------------------------------
    trading_mode: TradingMode = TradingMode.PAPER
    live_trading_confirm: str = ""
    initial_capital_usdt: float = Field(default=50.0, gt=0)

    # --- Hard risk constraints ----------------------------------------------------
    max_risk_per_trade_pct: float = Field(default=0.006, gt=0)
    stop_loss_max_pct: float = Field(default=0.006, gt=0)
    stop_loss_min_pct: float = Field(default=0.003, gt=0)
    stop_atr_multiplier: float = Field(default=1.5, gt=0)
    take_profit_min_pct: float = Field(default=0.006, gt=0)
    take_profit_max_pct: float = Field(default=0.008, gt=0)
    fee_rate: float = Field(default=0.001, ge=0)  # per side (Binance spot standard tier)
    max_slippage_pct: float = Field(default=0.0005, ge=0)
    consecutive_loss_limit: int = Field(default=2, ge=1)
    lockout_hours: float = Field(default=24.0, gt=0)
    max_daily_loss_pct: float = Field(default=0.015, gt=0)
    max_spread_pct: float = Field(default=0.0004, gt=0)
    entry_order_timeout_sec: int = Field(default=30, ge=5)

    # --- Statistical gates --------------------------------------------------------
    zscore_window: int = Field(default=50, ge=20)
    zscore_entry: float = -2.0
    zscore_precheck: float = -1.5  # below this the (costly) LLM debate is convened
    ob_levels: int = Field(default=5, ge=1, le=50)
    ob_min_imbalance: float = 0.10
    regression_lookback: int = Field(default=48, ge=16)
    forecast_horizon: int = Field(default=4, ge=1)
    min_win_rate: float = Field(default=0.75, gt=0, lt=1)
    min_sample_size: int = Field(default=20, ge=5)
    min_trade_sharpe: float = 0.20

    # --- Setup League (walk-forward tournament) -------------------------------------
    league_history_days: int = Field(default=730, ge=60, le=2000)
    league_refresh_hours: float = Field(default=24.0, ge=1)
    league_folds: int = Field(default=5, ge=3, le=10)
    league_min_profit_factor: float = Field(default=1.15, ge=1.0)
    max_hold_minutes: int = Field(default=480, ge=30)  # time stop, identical in backtest and live
    research_on_boot: bool = False

    # --- Strategy book (multi-coin, research-approved setups; paper only) -----------
    book_enabled: bool = True
    book_strategies: str = "confluence|1d|trigger=breakout,min_extra=2,rs_min=0.6|stop=2.5|tp=4.5|trail=None|hold=20"
    book_risk_per_trade: float = Field(default=0.006, gt=0, le=0.015)
    book_max_positions: int = Field(default=3, ge=1, le=5)
    book_pool_size: int = Field(default=40, ge=10, le=80)
    book_top_n: int = Field(default=15, ge=3, le=40)
    book_min_range: float = 0.025
    book_max_range: float = 0.15
    book_live_enabled: bool = False  # second switch: real book orders need TRADING_MODE=live AND this

    # --- Alerts, heartbeat, backups ------------------------------------------------------
    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str | None = None
    heartbeat_url: str | None = None  # e.g. a healthchecks.io ping URL: alerts you if pings stop
    heartbeat_interval_sec: int = Field(default=300, ge=60)
    backup_hour_utc: int = Field(default=3, ge=0, le=23)
    backup_keep: int = Field(default=7, ge=1, le=60)  # run the multi-asset research lab once (subprocess) if no report exists
    kelly_fraction: float = Field(default=0.25, gt=0, le=1)

    # --- Agent loop ---------------------------------------------------------------
    decision_interval_sec: int = Field(default=60, ge=10)
    debate_cooldown_sec: int = Field(default=900, ge=60)
    commentary_interval_sec: int = Field(default=1800, ge=300)

    # --- LLM router -----------------------------------------------------------------
    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-120b"
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-flash-latest"
    openrouter_api_key: SecretStr | None = None
    openrouter_model: str = "openai/gpt-oss-120b:free"
    llm_timeout_sec: float = Field(default=25.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)
    llm_breaker_threshold: int = Field(default=3, ge=1)
    llm_breaker_cooldown_sec: float = Field(default=120.0, gt=0)

    @model_validator(mode="after")
    def _enforce_mandate(self) -> "Settings":
        if self.max_risk_per_trade_pct > MANDATE_MAX_RISK_PER_TRADE:
            raise ValueError(
                f"max_risk_per_trade_pct={self.max_risk_per_trade_pct} exceeds mandate "
                f"ceiling {MANDATE_MAX_RISK_PER_TRADE}"
            )
        if self.stop_loss_max_pct > MANDATE_MAX_STOP_PCT:
            raise ValueError(f"stop_loss_max_pct exceeds mandate ceiling {MANDATE_MAX_STOP_PCT}")
        if self.stop_loss_min_pct > self.stop_loss_max_pct:
            raise ValueError("stop_loss_min_pct must be <= stop_loss_max_pct")
        if self.take_profit_min_pct > self.take_profit_max_pct:
            raise ValueError("take_profit_min_pct must be <= take_profit_max_pct")
        if self.trading_mode is TradingMode.LIVE:
            if self.live_trading_confirm != LIVE_CONFIRMATION_PHRASE:
                raise ValueError(
                    f"LIVE mode requires LIVE_TRADING_CONFIRM={LIVE_CONFIRMATION_PHRASE}"
                )
            if self.market_data_source is not MarketDataSource.EXCHANGE:
                raise ValueError("LIVE mode cannot run on a synthetic market feed")
            if not (self.exchange_api_key and self.exchange_api_secret):
                raise ValueError("LIVE mode requires EXCHANGE_API_KEY and EXCHANGE_API_SECRET")
        return self

    @property
    def is_live(self) -> bool:
        return self.trading_mode is TradingMode.LIVE

    @property
    def round_trip_cost_pct(self) -> float:
        """Fees on both legs plus worst tolerated slippage on entry."""
        return 2 * self.fee_rate + self.max_slippage_pct


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
