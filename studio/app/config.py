"""Settings for the Product Studio (env vars / Railway Variables).

Every integration is optional: a missing key turns that piece off instead of
crashing the office. With no LLM key at all the agents still work, using the
built-in playbooks (see ``app/agents/playbook.py``).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- core -------------------------------------------------------------
    database_url: str = "sqlite+aiosqlite:///./data/studio.db"
    redis_url: str = ""  # empty = in-process queue (cheapest: no Redis service on Railway)
    public_base_url: str = "http://localhost:8000"
    admin_token: SecretStr = SecretStr("change-me")
    artifacts_dir: str = "./data/artifacts"
    log_level: str = "INFO"

    # --- autonomy -----------------------------------------------------------
    scheduler_enabled: bool = True
    research_interval_min: int = 360
    pipeline_interval_min: int = 30
    ideas_per_research: int = 3
    max_builds_per_day: int = 2
    # False = products stop at "ready" and wait for one click from the CEO (/api/products/{id}/approve).
    # Turn it on once you have reviewed the first products.
    autopublish: bool = False
    # CFO audit: pause (deactivate the Stripe link of) products with no sale after 42 days. Off = recommend only.
    autopause_stale: bool = False
    # Pause between lines in meetings so the office UI can show the conversation (0 in tests).
    theatre_delay_sec: float = 3.0

    # --- free LLM providers (cascade: Groq -> Gemini -> OpenRouter -> built-in playbook) ----
    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-120b"
    groq_fast_model: str = "llama-3.1-8b-instant"
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-flash-latest"
    gemini_fast_model: str = "gemini-flash-lite-latest"
    openrouter_api_key: SecretStr | None = None
    openrouter_model: str = "openai/gpt-oss-120b:free"
    openrouter_fast_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    llm_timeout_sec: float = 60.0
    llm_daily_call_budget: int = 400  # stays well inside the free tiers

    # --- research sources (all free) ------------------------------------------
    reddit_client_id: str = ""
    reddit_client_secret: SecretStr | None = None
    reddit_user_agent: str = "product-studio/1.0 (research bot)"
    reddit_subreddits: str = "n8n,automation,Entrepreneur,smallbusiness,SaaS,indiehackers,solopreneur,freelance"
    producthunt_token: SecretStr | None = None
    research_offline: bool = False  # True = only the built-in pain catalogue (tests / no network)

    # --- sales platforms ------------------------------------------------------
    stripe_secret_key: SecretStr | None = None
    stripe_webhook_secret: SecretStr | None = None
    gumroad_access_token: SecretStr | None = None
    gumroad_webhook_secret: SecretStr | None = None
    whop_api_key: SecretStr | None = None
    whop_company_id: str = ""
    whop_webhook_secret: SecretStr | None = None

    # --- delivery + notifications ----------------------------------------------------
    resend_api_key: SecretStr | None = None
    email_from: str = "Studio <onboarding@resend.dev>"
    download_ttl_days: int = 30
    download_max: int = 10
    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str = ""
    discord_webhook_url: SecretStr | None = None

    # --- finance (USD) ------------------------------------------------------------------
    # Railway bills usage; these are your monthly estimates per service (check Usage on Railway).
    railway_web_usd: float = 5.0
    railway_postgres_usd: float = 2.0
    railway_redis_usd: float = 0.0
    railway_n8n_usd: float = 0.0
    other_monthly_usd: float = 0.0  # domain, email plan, ...
    stripe_fee_pct: float = 0.029
    stripe_fee_fixed_usd: float = 0.30
    gumroad_fee_pct: float = 0.10
    gumroad_fee_fixed_usd: float = 0.50
    whop_fee_pct: float = 0.03

    # --- goals ------------------------------------------------------------------------
    goal_monthly_revenue_usd: float = 500.0
    goal_products_per_week: int = 2
    goal_ideas_per_week: int = 15

    @field_validator("database_url")
    @classmethod
    def _async_driver(cls, v: str) -> str:
        # Railway gives postgres:// or postgresql://; SQLAlchemy async needs the asyncpg driver.
        if v.startswith("postgres://"):
            v = "postgresql://" + v[len("postgres://"):]
        if v.startswith("postgresql://"):
            v = "postgresql+asyncpg://" + v[len("postgresql://"):]
        return v

    @property
    def railway_monthly_usd(self) -> float:
        return self.railway_web_usd + self.railway_postgres_usd + self.railway_redis_usd + self.railway_n8n_usd

    @property
    def monthly_fixed_costs_usd(self) -> float:
        return self.railway_monthly_usd + self.other_monthly_usd

    @property
    def subreddits(self) -> list[str]:
        return [s.strip() for s in self.reddit_subreddits.split(",") if s.strip()]

    def platforms(self) -> list[str]:
        out = []
        if self.stripe_secret_key:
            out.append("stripe")
        if self.gumroad_access_token:
            out.append("gumroad")
        if self.whop_api_key and self.whop_company_id:
            out.append("whop")
        return out


@lru_cache
def get_settings() -> Settings:
    return Settings()


def secret(v: SecretStr | None) -> str:
    return v.get_secret_value() if v else ""

