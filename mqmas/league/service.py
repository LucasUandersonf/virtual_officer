"""Runs the Setup League in production and answers "which setups fired now?".

* History: candles are cached in SQLite and only the missing tail is fetched
  from the exchange (public endpoint, no API key, ~70 requests for 2 years
  of 15m bars the first time, then one or two per refresh).
* Tournament: rerun every ``league_refresh_hours`` in a worker thread (well
  under a second of CPU), result persisted in ``kv`` so a restart is ready
  immediately. No LLM tokens are spent here.
* Live: on each newly closed candle the approved setups' rules are evaluated
  on the live frame; the orchestrator convenes a debate only if one fires.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import pandas as pd

from mqmas.config import Settings
from mqmas.data.market_feed import TIMEFRAME_MS, MarketFeed, with_backoff
from mqmas.league.backtest import run_tournament
from mqmas.league.setups import SETUPS_BY_KEY, WARMUP_BARS, features
from mqmas.models import FiredSetup, LeagueReport
from mqmas.persistence.db import Database
from mqmas.ws.hub import WebSocketHub

log = logging.getLogger(__name__)

KV_KEY = "league_report"
LIVE_BARS = 1500  # enough for EMA200 and the 120-bar squeeze window to settle


class SetupLeague:
    def __init__(self, settings: Settings, db: Database, feed: MarketFeed, hub: WebSocketHub) -> None:
        self._s = settings
        self._db = db
        self._feed = feed
        self._hub = hub
        self.report: LeagueReport | None = None
        self.running = False

    # ------------------------------------------------------------------ state

    async def load_cached(self) -> None:
        raw = await self._db.get_kv(KV_KEY)
        if not raw:
            return
        try:
            report = LeagueReport.model_validate_json(raw)
        except ValueError:
            log.warning("cached league report unreadable; it will be rebuilt")
            return
        if report.symbol == self._s.symbol and report.timeframe == self._s.timeframe:
            self.report = report

    def age_hours(self) -> float | None:
        if self.report is None:
            return None
        return (datetime.now(timezone.utc) - self.report.generated_at).total_seconds() / 3600

    def needs_refresh(self) -> bool:
        age = self.age_hours()
        return age is None or age >= self._s.league_refresh_hours

    # ------------------------------------------------------------------ history

    async def _history(self) -> pd.DataFrame:
        s, ex = self._s, self._feed.exchange
        if ex is None:  # synthetic feed (demo/tests): the live frame is all there is
            return self._feed.candles().iloc[:-1]
        tf_ms = TIMEFRAME_MS.get(s.timeframe, 900_000)
        now = int(ex.milliseconds())
        horizon = now - s.league_history_days * 86_400_000
        last = await self._db.last_candle_ts(s.symbol, s.timeframe)
        since = max(horizon, (last + tf_ms) if last else horizon)
        fetched = 0
        while since < now - tf_ms:
            batch = await with_backoff(
                lambda x=since: ex.fetch_ohlcv(s.symbol, s.timeframe, since=x, limit=s.ohlcv_limit),
                "fetch_ohlcv(league)",
            )
            closed = [r for r in batch or [] if int(r[0]) + tf_ms <= now]
            if not closed:
                break
            await self._db.save_candles(s.symbol, s.timeframe, closed)
            fetched += len(closed)
            since = int(closed[-1][0]) + tf_ms
        await self._db.prune_candles(s.symbol, s.timeframe, horizon)
        rows = await self._db.load_candles(s.symbol, s.timeframe)
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        log.info("league history: %d candles (%d new)", len(df), fetched)
        return df.set_index("ts").astype(float)

    # ------------------------------------------------------------------ tournament

    async def refresh(self) -> LeagueReport | None:
        if self.running:
            return self.report
        self.running = True
        try:
            df = await self._history()
            if len(df) < WARMUP_BARS + 500:
                log.warning("league: only %d candles, tournament postponed", len(df))
                return self.report
            report = await asyncio.to_thread(run_tournament, df, self._s)
            self.report = report
            await self._db.set_kv(KV_KEY, report.model_dump_json())
            await self._hub.publish("league", report.model_dump(mode="json"))
            log.info("league refreshed: %d/%d setups approved (%d candles)",
                     len(report.approved), len(report.entries), report.bars)
            for e in report.entries:
                log.info("league #%d %s %s: oos n=%d acerto=%.1f%% exp=%+.3f%% pf=%.2f | %s", e.rank, e.key,
                         "APROVADO" if e.approved else "reprovado", e.oos.trades, e.oos.win_rate * 100,
                         e.oos.expectancy_pct * 100, e.oos.profit_factor, "; ".join(e.reasons) or "ok")
            return report
        except Exception:
            log.exception("league refresh failed")
            return self.report
        finally:
            self.running = False

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            if self.needs_refresh():
                await self.refresh()
            try:
                await asyncio.wait_for(stop.wait(), timeout=600)
            except TimeoutError:
                pass

    # ------------------------------------------------------------------ live signals

    def fired(self, df: pd.DataFrame) -> list[FiredSetup]:
        """Every league setup whose rule fires on the last *closed* candle of ``df``."""
        if self.report is None or len(df) < WARMUP_BARS + 2:
            return []
        closed = df.iloc[:-1].iloc[-LIVE_BARS:]
        f = features(closed)
        out: list[FiredSetup] = []
        for e in self.report.entries:
            setup = SETUPS_BY_KEY.get(e.key)
            if setup is None:
                continue
            sig = setup.rule(f).fillna(False)
            if bool(sig.iloc[-1]):
                out.append(FiredSetup(key=e.key, name=e.name, family=e.family, thesis=e.thesis,
                                      profile=e.profile, oos=e.oos, score=e.score, approved=e.approved,
                                      bar_ts=closed.index[-1].to_pydatetime()))
        return out
