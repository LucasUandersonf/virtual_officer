"""Strategy book: trades the research-approved setups across the liquid altcoin universe.

Same rules as the research backtest, so live ≡ tested:

* universe: monthly top-N by trailing 30-day dollar volume with daily range in
  [min_range, max_range] (point in time, from daily candles);
* context: BTC regime, each coin's daily trend and relative-strength rank;
* signal: the strategy's rule on the last *closed* candle of its timeframe,
  entry right after the close (≈ next bar open);
* exits: ATR stop and target fixed at entry, optional chandelier trail, time
  stop after ``hold`` bars; checked every ``SUPERVISE_SEC`` against tickers;
* portfolio: risk-based sizing, max concurrent positions, one per coin, cash
  limit, plus the shared kill switch / circuit breaker / daily loss rules.

The four agents still have a say: the CRO runs an LLM pre-mortem on every
candidate and can veto it (never force it). Fail-closed: no LLM → no entry.

Paper only for now: in live mode the book refuses to open positions, because
multi-symbol live order routing is not implemented yet.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from mqmas.agents.cro import ChiefRiskOfficer
from mqmas.agents.personas import PERSONAS
from mqmas.config import Settings, TradingMode
from mqmas.data.market_feed import with_backoff
from mqmas.llm.router import LLMRouter
from mqmas.models import AgentMessage, AgentRole, CroOpinion, ExitReason, Position, PositionStatus, Vote
from mqmas.persistence.db import Database
from mqmas.research.data import EXCLUDED_BASES, LEVERAGED_SUFFIXES, TF_MS
from mqmas.research.engine import Exit
from mqmas.research.strategies import (
    FAMILIES,
    WARMUP,
    Config,
    btc_regime,
    config_by_key,
    daily_trend,
    features,
    monthly_eligibility,
    relative_strength,
)
from mqmas.risk.constraints import RiskManager
from mqmas.ta.confluence import RULE, live_score, order_proposal
from mqmas.ta.levels import compact_map, price_map
from mqmas.ta.live import agent_view, load_study
from mqmas.ws.hub import WebSocketHub

log = logging.getLogger(__name__)

SUPERVISE_SEC = 30
EVAL_GRACE_SEC = 90  # wait for the exchange to publish the closed candle
MAX_SIGNAL_LAG_MS = 4 * 3_600_000  # a daily signal is only actionable within 4h of the close (≈ next open)
DAILY_BARS = 400
SETUP_PREFIX = "book:"
SCORE_TFS = ("4h", "1d")


@dataclass
class Signal:
    symbol: str
    strategy: str
    price: float
    atr: float
    rs: float
    layers: dict[str, Any]
    ta: dict[str, Any] = field(default_factory=dict)
    cfg: Config | None = None  # set for confluence-score entries (not in the research strategy table)
    evidence: str = "aprovado no walk-forward 2023-07→2026-10: PF 2,29, acerto 56%, DD 2,9%"


@dataclass
class BookState:
    last_eval: dict[str, int] = field(default_factory=dict)  # tf -> close ts (ms) evaluated
    last_signals: list[dict[str, Any]] = field(default_factory=list)
    universe: list[str] = field(default_factory=list)
    btc_regime_ok: bool | None = None
    last_run: str | None = None
    notes: list[str] = field(default_factory=list)
    confluence: list[dict[str, Any]] = field(default_factory=list)


def _score_and_map(df: pd.DataFrame, study: dict[str, Any] | None, tf: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return live_score(df, study, tf), price_map(df)


def short_name(cfg: Config) -> str:
    p = cfg.p
    return f"{cfg.family}-{cfg.tf}" + (f"-{p['trigger']}" if "trigger" in p else "")


class StrategyBook:
    def __init__(self, settings: Settings, db: Database, risk: RiskManager, router: LLMRouter,
                 hub: WebSocketHub, exchange: Any) -> None:
        self._s = settings
        self._db = db
        self._risk = risk
        self._hub = hub
        self._ex = exchange
        self.cro = ChiefRiskOfficer(router)
        known = config_by_key()
        keys = [k.strip() for k in settings.book_strategies.split(";") if k.strip()]
        self.strategies = [known[k] for k in keys if k in known]
        for k in keys:
            if k not in known:
                log.warning("book: unknown strategy key ignored: %s", k)
        self.state = BookState()
        self._lock = asyncio.Lock()

    @property
    def mode(self) -> str:
        return self._s.trading_mode.value

    # ------------------------------------------------------------------ helpers

    async def _say(self, role: AgentRole, text: str, vote: Vote | None = None,
                   provider: str = "rules", data: dict[str, Any] | None = None) -> None:
        p = PERSONAS[role]
        m = AgentMessage(role=role, agent_name=p.name, text=text, vote=vote, provider=provider, data=data or {})
        await self._db.save_message(m)
        await self._hub.publish("agent_message", m.model_dump(mode="json"))

    async def positions(self) -> list[Position]:
        return [p for p in await self._db.active_positions(self.mode) if (p.setup or "").startswith(SETUP_PREFIX)]

    async def _ohlcv(self, symbol: str, tf: str, limit: int) -> pd.DataFrame:
        rows = await with_backoff(lambda: self._ex.fetch_ohlcv(symbol, tf, limit=limit), f"ohlcv {symbol} {tf}")
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        now = int(time.time() * 1000)
        df = df[df["ts"] + TF_MS[tf] <= now]  # closed candles only
        df.index = pd.to_datetime(df.pop("ts").astype("int64"), unit="ms", utc=True)
        return df.astype(float)

    async def _pool(self) -> list[str]:
        markets = self._ex.markets or await with_backoff(self._ex.load_markets, "load_markets")
        tickers = await with_backoff(self._ex.fetch_tickers, "fetch_tickers")
        rows: list[tuple[float, str]] = []
        for sym, m in markets.items():
            if not (m.get("spot") and m.get("active") and m.get("quote") == "USDT"):
                continue
            base = str(m.get("base", ""))
            if base in EXCLUDED_BASES or base.endswith(LEVERAGED_SUFFIXES):
                continue
            rows.append((float((tickers.get(sym) or {}).get("quoteVolume") or 0.0), sym))
        rows.sort(reverse=True)
        pool = [s for _, s in rows[: self._s.book_pool_size]]
        return pool if "BTC/USDT" in pool else ["BTC/USDT", *pool]

    # ------------------------------------------------------------------ evaluation

    def _due(self, tf: str, now_ms: int, step: int | None = None) -> int | None:
        step = step or TF_MS[tf]
        closed = (now_ms - EVAL_GRACE_SEC * 1000) // step * step  # open time of the bar that just closed + step
        if self.state.last_eval.get(tf) == closed:
            return None
        return closed

    async def evaluate(self, now_ms: int | None = None) -> list[Signal]:
        now_ms = now_ms or int(time.time() * 1000)
        due = {c.tf for c in self.strategies if self._due(c.tf, now_ms) is not None}
        if not due:
            return []
        pool = await self._pool()
        daily: dict[str, pd.DataFrame] = {}
        for sym in pool:
            try:
                d = await self._ohlcv(sym, "1d", DAILY_BARS)
            except Exception:
                log.warning("book: daily candles unavailable for %s", sym)
                continue
            if len(d) > WARMUP + 50:
                daily[sym] = d
        if "BTC/USDT" not in daily:
            self.state.notes = ["sem candles diários do BTC: avaliação adiada"]
            return []
        eligible = monthly_eligibility(daily, self._s.book_top_n, self._s.book_min_range, self._s.book_max_range)
        month = pd.Timestamp(now_ms, unit="ms", tz="UTC").tz_localize(None).to_period("M")
        universe = sorted(eligible.get(month, set()))
        rs = relative_strength(daily, eligible)
        regime = btc_regime(daily["BTC/USDT"])
        self.state.universe = universe
        self.state.btc_regime_ok = bool(regime.iloc[-1]) if len(regime) else None
        signals: list[Signal] = []
        for tf in sorted(due):
            cfgs = [c for c in self.strategies if c.tf == tf]
            for sym in universe:
                df = daily[sym] if tf == "1d" else await self._ohlcv(sym, tf, DAILY_BARS)
                if len(df) < WARMUP + 2:
                    continue
                f = features(df)
                idx = f.index.normalize()
                f["rs"] = rs[sym].reindex(idx).to_numpy() if sym in rs.columns else np.nan
                f["d_trend"] = daily_trend(daily[sym]).reindex(idx).fillna(False).to_numpy(dtype=bool)
                reg = regime.reindex(idx).fillna(False).to_numpy(dtype=bool)
                for c in cfgs:
                    sig = FAMILIES[c.family](f, c.p, reg)
                    if sig.size and bool(sig[-1]):
                        last = f.iloc[-1]
                        signals.append(Signal(sym, c.key, float(last["close"]), float(last["atr"]),
                                              float(last["rs"]) if np.isfinite(last["rs"]) else 0.0,
                                              {"rs_rank": round(float(last["rs"]), 2) if np.isfinite(last["rs"]) else None,
                                               "volume_x": round(float(last["volume"] / last["vol20"]), 2)
                                               if last["vol20"] else None,
                                               "rsi14": round(float(last["rsi14"]), 1),
                                               "atr_pct": round(float(last["atr"] / last["close"]) * 100, 2)},
                                              self._ta(df, tf)))
            self.state.last_eval[tf] = self._due(tf, now_ms) or self.state.last_eval.get(tf, 0)
        signals.sort(key=lambda x: x.rs, reverse=True)
        stale = [tf for tf in due if now_ms - self.state.last_eval.get(tf, 0) > min(TF_MS[tf] // 6, MAX_SIGNAL_LAG_MS)]
        self.state.notes = [f"sinais {tf} só para leitura: o candle fechou há muito tempo (o teste entra na abertura "
                            f"seguinte)" for tf in stale]
        actionable = [s for s in signals if config_by_key()[s.strategy].tf not in stale]
        self.state.last_signals = [{"symbol": s.symbol, "strategy": short_name(config_by_key()[s.strategy]),
                                    "price": s.price, **s.layers} for s in signals]
        self.state.last_run = datetime.now(timezone.utc).isoformat()
        log.info("book: evaluated %s on %d coins → %d signal(s)", ",".join(sorted(due)), len(universe), len(signals))
        await self._hub.publish("book", self.snapshot_light())
        return actionable

    # ------------------------------------------------------------------ confluence score

    def _validation(self) -> dict[str, Any] | None:
        path = self._s.db_path.parent / "research" / "confluence_report.json"
        try:
            data = json.loads(path.read_text()) if path.exists() else None
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    async def scan_confluence(self, now_ms: int | None = None) -> list[Signal]:
        """Score every coin of the universe at each 4h/1d close; log it; return the validated entries."""
        now_ms = now_ms or int(time.time() * 1000)
        if not self.state.universe:
            return []
        root = self._s.db_path.parent / "research"
        study, validation = load_study(root), self._validation()
        equity = await self._risk.equity_estimate()
        out: list[Signal] = []
        for tf in SCORE_TFS:
            closed = self._due(f"score:{tf}", now_ms, TF_MS[tf])
            if closed is None:
                continue
            rows = []
            for sym in self.state.universe:
                try:
                    df = await self._ohlcv(sym, tf, DAILY_BARS)
                    if len(df) < WARMUP + 50:
                        continue
                    sc, pm = await asyncio.to_thread(_score_and_map, df, study, tf)
                except Exception:
                    log.exception("book: confluence scan failed for %s %s", sym, tf)
                    continue
                prop = order_proposal(sc, tf, validation, equity, self._s.book_risk_per_trade, pm)
                await self._db.save_ta_signal(df.index[-1].isoformat(), sym, tf, sc["score"], sc["componentes"],
                                              sc["padroes_com_historico"], compact_map(pm), prop["decisao"],
                                              prop["ordem"])
                rows.append({"symbol": sym, "tf": tf, "score": sc["score"], "decisao": prop["decisao"],
                             "motivo": prop["motivo"], "componentes": sc["componentes"], "ordem": prop["ordem"],
                             "alertas": prop["alertas"]})
                if prop["decisao"] == "EXECUTAR":
                    th = prop["limiar"]
                    cfg = Config("score", tf, (("limiar", th),),
                                 Exit(RULE.stop_atr, RULE.stop_atr * RULE.target_r, None, RULE.hold_for(tf)))
                    v = (validation or {})["tf"][tf]
                    best = next(r for r in v["linhas"] if r["limiar"] == th)["periodo2"]
                    out.append(Signal(sym, cfg.key, sc["preco"], sc["atr"], sc["score"],
                                      {"score": sc["score"], "componentes": sc["componentes"]},
                                      {"confluencia": sc, "mapa_de_precos": compact_map(pm), "alertas": prop["alertas"]},
                                      cfg=cfg, evidence=(f"score ≥ {th} validado fora da amostra: {best['trades']} "
                                                         f"trades, acerto {best['acerto']}, PF {best['pf']}")))
            self.state.last_eval[f"score:{tf}"] = closed
            self.state.confluence = sorted(rows + [r for r in self.state.confluence if r["tf"] != tf],
                                           key=lambda r: -r["score"])[:20]
            log.info("book: confluence %s scored %d coins → %d entry(ies)", tf, len(rows),
                     sum(r["decisao"] == "EXECUTAR" for r in rows))
        return out

    # ------------------------------------------------------------------ entries

    def _ta(self, df: pd.DataFrame, tf: str) -> dict[str, Any]:
        try:
            return agent_view(df[["open", "high", "low", "close", "volume"]], tf, self._s.db_path.parent / "research")
        except Exception:
            log.exception("book: technical analysis failed")
            return {}

    async def _cro_veto(self, sig: Signal, cfg: Config, stop: float, target: float) -> tuple[bool, str, str]:
        payload = {"ativo": sig.symbol, "setup": short_name(cfg), "preco": sig.price,
                   "stop": round(stop, 8), "alvo": round(target, 8), "camadas_confluencia": sig.layers,
                   "regime_btc_ok": self.state.btc_regime_ok, "analise_tecnica": sig.ta,
                   "evidencia": sig.evidence}
        res = await self.cro.ask(payload, CroOpinion)
        if res is None:
            return False, "LLMs indisponíveis — fail-closed, sem entrada.", "rules"
        op, meta = res
        high = [f for f in op.pre_mortem if f.likelihood == "high" and not f.mitigation.strip()]
        ok = op.vote is Vote.APPROVE and not high
        risks = " · ".join(f"[{f.likelihood}] {f.risk}" for f in op.pre_mortem[:3])
        return ok, f"Pre-mortem {sig.symbol}: {risks}. {op.verdict}", meta.provider

    async def try_entries(self, signals: list[Signal]) -> list[Position]:
        if not signals:
            return []
        if self._s.trading_mode is TradingMode.LIVE:
            await self._say(AgentRole.SYSTEM, "Carteira multi-moeda: modo LIVE ainda não suportado — sinais "
                                              "registrados, nenhuma ordem enviada.")
            return []
        opened: list[Position] = []
        known = config_by_key()
        async with self._lock:
            if await self._risk.kill_switch_engaged() or await self._risk.lock_until():
                await self._say(AgentRole.CRO, "Sinais da carteira ignorados: kill switch/circuit breaker ativo.")
                return []
            for sig in signals:
                active = await self.positions()
                if len(active) >= self._s.book_max_positions:
                    break
                if any(p.symbol == sig.symbol for p in active):
                    continue
                cfg = sig.cfg or known[sig.strategy]
                ex = cfg.exit
                stop = sig.price - ex.stop_atr * sig.atr
                target = sig.price + ex.target_atr * sig.atr if ex.target_atr else sig.price * 10
                if stop <= 0:
                    continue
                await self._say(AgentRole.CHIEF_QUANT,
                                f"Score de confluência {sig.layers.get('score')} em {sig.symbol} ({cfg.tf}): "
                                f"{sig.layers.get('componentes')}." if sig.cfg else
                                f"Confluência em {sig.symbol} ({short_name(cfg)}): tendência diária ✔, regime BTC ✔, "
                                f"gatilho ✔, extras {sig.layers}.")
                ok, text, provider = await self._cro_veto(sig, cfg, stop, target)
                await self._say(AgentRole.CRO, text, vote=Vote.APPROVE if ok else Vote.REJECT, provider=provider)
                if not ok:
                    continue
                pos = await self._open_paper(sig, cfg, stop, target)
                if pos:
                    opened.append(pos)
        return opened

    async def _open_paper(self, sig: Signal, cfg: Config, stop: float, target: float) -> Position | None:
        s = self._s
        ticker = await with_backoff(lambda: self._ex.fetch_ticker(sig.symbol), f"ticker {sig.symbol}")
        ask = float(ticker.get("ask") or ticker.get("last") or sig.price)
        fill = ask * (1 + s.max_slippage_pct)
        stop_pct = (fill - stop) / fill
        if stop_pct <= 0:
            return None
        equity = await self._risk.equity_estimate()
        committed = sum(p.quantity * p.entry_price for p in await self._db.active_positions(self.mode))
        loss_per_unit = stop_pct + 2 * (s.fee_rate + s.max_slippage_pct)
        notional = min(equity * s.book_risk_per_trade / loss_per_unit, max(equity - committed, 0.0))
        below_min = notional < 5.0
        if notional <= 0:
            return None
        qty = notional / fill
        hold_minutes = cfg.exit.hold * TF_MS[cfg.tf] // 60_000
        pos = Position(debate_id=f"book-{int(time.time())}", mode=self.mode, symbol=sig.symbol,
                       status=PositionStatus.OPEN, quantity=qty, entry_price=fill, stop_price=stop,
                       take_profit_price=target, fees_usdt=notional * s.fee_rate,
                       setup=f"{SETUP_PREFIX}{short_name(cfg)}|hold={hold_minutes}"
                             f"|trail={cfg.exit.trail_atr or 0}|atr={sig.atr:.8g}")
        pos.id = await self._db.insert_position(pos)
        await self._db.update_position(pos)
        await self._hub.publish("position", {"event": "opened", "position": pos.model_dump(mode="json")})
        await self._say(AgentRole.EXECUTION,
                        f"PAPER BUY {sig.symbol}: {qty:.6g} @ {fill:.6g} (US$ {notional:.2f}). Stop {stop:.6g} "
                        f"({stop_pct:.1%}) · alvo {target:.6g} · saída por tempo em {hold_minutes // 1440} dias."
                        + (" ⚠ abaixo do mínimo de US$ 5 da Binance: no modo real esta ordem seria recusada."
                           if below_min else ""))
        return pos

    # ------------------------------------------------------------------ exits

    @staticmethod
    def _meta(p: Position) -> dict[str, str]:
        parts = (p.setup or "").removeprefix(SETUP_PREFIX).split("|")
        return dict(x.split("=", 1) for x in parts[1:] if "=" in x)

    async def supervise(self) -> None:
        active = await self.positions()
        if not active:
            return
        kill = await self._risk.kill_switch_engaged()
        tickers = await with_backoff(lambda: self._ex.fetch_tickers([p.symbol for p in active]), "book tickers")
        now = datetime.now(timezone.utc)
        for p in active:
            t = tickers.get(p.symbol) or {}
            bid = float(t.get("bid") or t.get("last") or 0.0)
            if bid <= 0:
                continue
            meta = self._meta(p)
            trail = float(meta.get("trail", 0) or 0)
            atr = float(meta.get("atr", 0) or 0)
            if trail and atr:  # chandelier approximated with the polled bid (never loosens)
                new_stop = max(p.stop_price, bid - trail * atr)
                if new_stop > p.stop_price:
                    p.stop_price = new_stop
                    await self._db.update_position(p)
            hold = timedelta(minutes=int(meta.get("hold", 0) or 0))
            reason: ExitReason | None = None
            price = bid
            if kill:
                reason = ExitReason.MANUAL
            elif bid <= p.stop_price:
                reason = ExitReason.STOP_LOSS  # a gap below the stop fills at the bid, not the stop
            elif bid >= p.take_profit_price:
                reason, price = ExitReason.TAKE_PROFIT, p.take_profit_price
            elif hold and now - p.opened_at >= hold:
                reason = ExitReason.TIME_STOP
            if reason:
                await self._close(p, reason, price * (1 - self._s.max_slippage_pct))

    async def _close(self, p: Position, reason: ExitReason, price: float) -> None:
        fee = price * p.quantity * self._s.fee_rate
        p.exit_price = price
        p.fees_usdt += fee
        p.pnl_usdt = (price - p.entry_price) * p.quantity - p.fees_usdt
        p.exit_reason = reason
        p.status = PositionStatus.CLOSED
        p.closed_at = datetime.now(timezone.utc)
        await self._db.update_position(p)
        await self._hub.publish("position", {"event": "closed", "position": p.model_dump(mode="json")})
        await self._say(AgentRole.EXECUTION, f"PAPER SELL {p.symbol} ({reason.value}) @ {price:.6g} · "
                                             f"PnL US$ {p.pnl_usdt:+.4f}")
        lock = await self._risk.on_trade_closed()
        if lock:
            await self._hub.publish("risk", {"event": "circuit_breaker", "lock_until": lock.isoformat()})

    # ------------------------------------------------------------------ loop & API

    def snapshot_light(self) -> dict[str, Any]:
        return {"strategies": [short_name(c) for c in self.strategies], "universe": self.state.universe,
                "btc_regime_ok": self.state.btc_regime_ok, "last_run": self.state.last_run,
                "last_signals": self.state.last_signals[:10], "notes": self.state.notes,
                "confluence": self.state.confluence[:10]}

    async def snapshot(self) -> dict[str, Any]:
        return {**self.snapshot_light(), "risk_per_trade": self._s.book_risk_per_trade,
                "max_positions": self._s.book_max_positions,
                "open_positions": [json.loads(p.model_dump_json()) for p in await self.positions()]}

    async def run(self, stop: asyncio.Event) -> None:
        if not self.strategies:
            log.warning("book: no valid strategies configured; idle")
            return
        log.info("book: running %s", ", ".join(short_name(c) for c in self.strategies))
        while not stop.is_set():
            try:
                await self.supervise()
                signals = await self.evaluate()
                await self.try_entries(signals)
                await self.try_entries(await self.scan_confluence())
            except Exception:
                log.exception("book cycle failed")
            try:
                await asyncio.wait_for(stop.wait(), timeout=SUPERVISE_SEC)
            except TimeoutError:
                pass
