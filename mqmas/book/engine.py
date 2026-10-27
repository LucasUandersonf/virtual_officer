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
from mqmas.book import swing
from mqmas.config import Settings, TradingMode
from mqmas.data.market_feed import with_backoff
from mqmas.execution.spot import Fill, OrderRefused, SpotExecutor, client_id
from mqmas.llm.router import LLMRouter
from mqmas.models import AgentMessage, AgentRole, CroOpinion, ExitReason, Position, PositionStatus, Vote
from mqmas.ops import trim_memory
from mqmas.persistence.db import Database
from mqmas.research.data import TF_MS, is_crypto
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
EVIDENCE = {
    "confluence": "aprovado no walk-forward 2023-07→2026-10: PF 2,29, acerto 56%, DD 2,9%",
    "classic": "zero parâmetros otimizados; 2022-07→2026-10 com taxas: 119 trades, acerto 46%, payoff 2,45, PF 2,11 "
               "(2,02 com custo dobrado), maior perda 1,06R, 75% das janelas de 6 meses positivas",
}


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
    risk: float | None = None  # swing: stop distance from the signal close, re-anchored on the fill
    target_r: float = 0.0  # swing: target in R multiples (0 = trailing only)
    meta: str = ""  # extra setup metadata stored with the position


def tf_ms(tf: str) -> int:
    return TF_MS.get(tf) or swing.DAYS[tf] * TF_MS["1d"]


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
        # real orders only with BOTH switches: TRADING_MODE=live (+ confirmation phrase) and BOOK_LIVE_ENABLED=true
        self._live = settings.is_live and settings.book_live_enabled
        self._spot = SpotExecutor(exchange, settings.fee_rate) if exchange is not None else None

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
            if not is_crypto(base):
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
        swing_closed = self._due("swing", now_ms, TF_MS["1d"]) if self._s.book_swing else None
        if not due and swing_closed is None:
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
        stale: list[str] = []
        if swing_closed is not None:
            signals += await self._swing(daily, universe, rs)
            self.state.last_eval["swing"] = swing_closed
            if now_ms - swing_closed > MAX_SIGNAL_LAG_MS:
                stale.append("swing")
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
                                              self._ta(df, tf), evidence=EVIDENCE.get(c.family, EVIDENCE["confluence"])))
            self.state.last_eval[tf] = self._due(tf, now_ms) or self.state.last_eval.get(tf, 0)
        signals.sort(key=lambda x: x.rs, reverse=True)
        stale += [tf for tf in due if now_ms - self.state.last_eval.get(tf, 0) > min(TF_MS[tf] // 6, MAX_SIGNAL_LAG_MS)]
        self.state.notes = [f"sinais {tf} só para leitura: o candle fechou há muito tempo (o teste entra na abertura "
                            f"seguinte)" for tf in stale]
        known = config_by_key()
        actionable = [s for s in signals
                      if ("swing" if s.risk is not None else (s.cfg or known[s.strategy]).tf) not in stale]
        self.state.last_signals = [{"symbol": s.symbol, "strategy": short_name(s.cfg or known[s.strategy]),
                                    "price": s.price, **s.layers} for s in signals]
        self.state.last_run = datetime.now(timezone.utc).isoformat()
        log.info("book: evaluated %s on %d coins → %d signal(s)",
                 ",".join(sorted(due) + (["swing 1d/2d/3d"] if swing_closed is not None else [])), len(universe),
                 len(signals))
        await self._hub.publish("book", self.snapshot_light())
        return actionable

    # ------------------------------------------------------------------ swing 1d / 2d / 3d

    async def _swing(self, daily: dict[str, pd.DataFrame], universe: list[str], rs: pd.DataFrame) -> list[Signal]:
        """At each daily close: manage the open swing positions, then look for new entries."""
        found: dict[str, dict[str, dict[str, Any]]] = {}
        out: list[Signal] = []
        held = {p.symbol for p in await self.positions() if self._meta(p).get("mgmt")}
        for sym in sorted(set(universe) | held):
            try:
                d = daily[sym] if sym in daily else await self._ohlcv(sym, "1d", DAILY_BARS)
                hits, mgmt = await asyncio.to_thread(swing.scan, d)
            except Exception:
                log.exception("book: swing scan failed for %s", sym)
                continue
            found[sym] = mgmt
            if sym not in universe:
                continue
            r = rs[sym].iloc[-1] if sym in rs.columns and len(rs) else np.nan
            for h in hits:
                s = h.setup
                out.append(Signal(sym, h.cfg.key, h.price, h.atr, float(r) if np.isfinite(r) else 0.0, h.layers,
                                  cfg=h.cfg, evidence=swing.EVIDENCE[(s["tf"], s["nome"])], risk=h.risk,
                                  target_r=s["alvo_r"], meta=f"|mgmt=be2r|tf={s['tf']}|r={h.risk:.8g}"))
        await self._swing_manage(found)
        out.sort(key=lambda x: x.rs, reverse=True)
        return out

    async def _swing_manage(self, found: dict[str, dict[str, dict[str, Any]]]) -> None:
        """Closed candle of a swing position's timeframe: refresh ATR / low of the last 3, exit on a bearish pattern."""
        for p in await self.positions():
            meta = self._meta(p)
            m = found.get(p.symbol, {}).get(meta.get("tf", ""))
            if not meta.get("mgmt") or m is None or p.status is not PositionStatus.OPEN:
                continue
            self._set_meta(p, atr=f"{m['atr']:.8g}", low3=f"{m['low3']:.8g}")
            await self._db.update_position(p)
            if m["exit"]:
                await self._say(AgentRole.EXECUTION, f"Padrão de reversão de baixa no candle {meta['tf']} de "
                                                     f"{p.symbol}: saída na abertura, como no teste.")
                await self._exit_now(p, ExitReason.SIGNAL_EXIT)

    @staticmethod
    def _set_meta(p: Position, **kv: str) -> None:
        parts = [x for x in (p.setup or "").split("|") if x.split("=", 1)[0] not in kv]
        p.setup = "|".join(parts + [f"{k}={v}" for k, v in kv.items()])

    async def _exit_now(self, p: Position, reason: ExitReason) -> None:
        if p.mode == "live" and self._live and self._spot is not None:
            f = await self._spot.sell_all(p.symbol, p.quantity, p.stop_order_id, urgent=True)
            await self._finish(p, reason, f.avg_price, f.fee_usdt)
            return
        t = await with_backoff(lambda: self._ex.fetch_ticker(p.symbol), f"ticker {p.symbol}")
        bid = float(t.get("bid") or t.get("last") or p.entry_price)
        await self._close(p, reason, bid * (1 - self._s.max_slippage_pct))

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
        if self._s.trading_mode is TradingMode.LIVE and not self._live:
            await self._say(AgentRole.SYSTEM, "Carteira multi-moeda: modo LIVE sem BOOK_LIVE_ENABLED=true — sinais "
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
                if sig.risk is not None:
                    stop = sig.price - sig.risk
                    target = sig.price + sig.target_r * sig.risk if sig.target_r else sig.price * 10
                else:
                    stop = sig.price - ex.stop_atr * sig.atr
                    target = sig.price + ex.target_atr * sig.atr if ex.target_atr else sig.price * 10
                if stop <= 0:
                    continue
                await self._say(AgentRole.CHIEF_QUANT,
                                f"Swing {sig.layers.get('setup')} em {sig.symbol}: stop {sig.layers.get('risco_pct')}% "
                                f"abaixo, alvo {sig.layers.get('alvo')}, breakeven em +1R e trailing a partir de +2R. "
                                f"Evidência: {sig.evidence}." if sig.risk is not None else
                                f"Score de confluência {sig.layers.get('score')} em {sig.symbol} ({cfg.tf}): "
                                f"{sig.layers.get('componentes')}." if sig.cfg else
                                f"Confluência em {sig.symbol} ({short_name(cfg)}): tendência diária ✔, regime BTC ✔, "
                                f"gatilho ✔, extras {sig.layers}.")
                ok, text, provider = await self._cro_veto(sig, cfg, stop, target)
                await self._say(AgentRole.CRO, text, vote=Vote.APPROVE if ok else Vote.REJECT, provider=provider)
                if not ok:
                    continue
                pos = await (self._open_live if self._live else self._open_paper)(sig, cfg, stop, target)
                if pos:
                    opened.append(pos)
        return opened

    async def _risk_budget(self, equity: float) -> float:
        """US$ at risk for a new entry: risk per trade, cut to half below −10 % and to a quarter below −20 % from
        the equity peak, and capped so the summed open risk of the book stays under ``book_heat_max``."""
        s = self._s
        closed = sorted(await self._db.closed_positions(self.mode, limit=100_000),
                        key=lambda p: p.closed_at or datetime.min.replace(tzinfo=timezone.utc))
        curve = s.initial_capital_usdt + np.cumsum([p.pnl_usdt or 0.0 for p in closed])
        peak = max([s.initial_capital_usdt, *curve.tolist()])
        dd = 1 - equity / peak if peak > 0 else 0.0
        mult = 0.25 if dd >= s.book_dd_severe else 0.5 if dd >= s.book_dd_throttle else 1.0
        open_risk = sum(max(p.entry_price - p.stop_price, 0.0) * p.quantity for p in await self.positions())
        return max(min(equity * s.book_risk_per_trade * mult, equity * s.book_heat_max - open_risk), 0.0)

    async def _open_paper(self, sig: Signal, cfg: Config, stop: float, target: float) -> Position | None:
        s = self._s
        ticker = await with_backoff(lambda: self._ex.fetch_ticker(sig.symbol), f"ticker {sig.symbol}")
        ask = float(ticker.get("ask") or ticker.get("last") or sig.price)
        fill = ask * (1 + s.max_slippage_pct)
        if sig.risk is not None:  # same as the backtest: the stop distance is kept, anchored on the fill
            stop, target = fill - (sig.price - stop), fill + (target - sig.price)
        stop_pct = (fill - stop) / fill
        if stop_pct <= 0:
            return None
        equity = await self._risk.equity_estimate()
        committed = sum(p.quantity * p.entry_price for p in await self._db.active_positions(self.mode))
        loss_per_unit = stop_pct + 2 * (s.fee_rate + s.max_slippage_pct)
        notional = min(await self._risk_budget(equity) / loss_per_unit, max(equity - committed, 0.0))
        below_min = notional < 5.0
        if notional <= 0:
            return None
        qty = notional / fill
        hold_minutes = cfg.exit.hold * tf_ms(cfg.tf) // 60_000
        pos = Position(debate_id=f"book-{int(time.time())}", mode=self.mode, symbol=sig.symbol,
                       status=PositionStatus.OPEN, quantity=qty, entry_price=fill, stop_price=stop,
                       take_profit_price=target, fees_usdt=notional * s.fee_rate,
                       setup=f"{SETUP_PREFIX}{short_name(cfg)}|hold={hold_minutes}"
                             f"|trail={cfg.exit.trail_atr or 0}|atr={sig.atr:.8g}{sig.meta}")
        pos.id = await self._db.insert_position(pos)
        await self._db.update_position(pos)
        await self._hub.publish("position", {"event": "opened", "position": pos.model_dump(mode="json")})
        await self._say(AgentRole.EXECUTION,
                        f"PAPER BUY {sig.symbol}: {qty:.6g} @ {fill:.6g} (US$ {notional:.2f}). Stop {stop:.6g} "
                        f"({stop_pct:.1%}) · alvo {target:.6g} · saída por tempo em {hold_minutes // 1440} dias."
                        + (" ⚠ abaixo do mínimo de US$ 5 da Binance: no modo real esta ordem seria recusada."
                           if below_min else ""))
        return pos

    async def _open_live(self, sig: Signal, cfg: Config, stop: float, target: float) -> Position | None:
        """Real order: persisted as PENDING with our clientOrderId *before* it is sent."""
        assert self._spot is not None
        s = self._s
        ticker = await with_backoff(lambda: self._ex.fetch_ticker(sig.symbol), f"ticker {sig.symbol}")
        ask = float(ticker.get("ask") or ticker.get("last") or sig.price)
        limit = ask * (1 + s.max_slippage_pct)
        stop_pct = (limit - stop) / limit
        if stop_pct <= 0:
            return None
        equity = await self._risk.equity_estimate()
        committed = sum(p.quantity * p.entry_price for p in await self._db.active_positions(self.mode))
        free = await self._spot.free_quote()
        loss_per_unit = stop_pct + 2 * (s.fee_rate + s.max_slippage_pct)
        notional = min(await self._risk_budget(equity) / loss_per_unit, max(equity - committed, 0.0), free * 0.98)
        cid = client_id("e")
        hold_minutes = cfg.exit.hold * tf_ms(cfg.tf) // 60_000
        pos = Position(debate_id=f"book-{int(time.time())}", mode=self.mode, symbol=sig.symbol,
                       status=PositionStatus.PENDING, quantity=notional / limit, entry_price=limit, stop_price=stop,
                       take_profit_price=target, entry_order_id=cid,
                       setup=f"{SETUP_PREFIX}{short_name(cfg)}|hold={hold_minutes}|trail={cfg.exit.trail_atr or 0}"
                             f"|atr={sig.atr:.8g}|sd={sig.price - stop:.8g}|td={target - sig.price:.8g}{sig.meta}|cid={cid}")
        pos.id = await self._db.insert_position(pos)
        try:
            fill = await self._spot.buy(sig.symbol, notional, limit, cid, s.entry_order_timeout_sec)
        except OrderRefused as exc:
            await self._cancel_pending(pos)
            await self._say(AgentRole.EXECUTION, f"Ordem recusada antes de enviar: {exc}. Com o capital atual o "
                                                 f"risco por operação não alcança o mínimo da Binance.")
            await self._hub.publish("risk", {"event": "order_refused", "symbol": sig.symbol, "detail": str(exc)})
            return None
        except Exception:
            log.exception("book: live buy %s failed; reconciling by clientOrderId", sig.symbol)
            await self._reconcile_pending(pos)
            return None
        return await self._activate(pos, fill)

    async def _cancel_pending(self, p: Position) -> None:
        p.status, p.exit_reason = PositionStatus.CANCELLED, ExitReason.ENTRY_TIMEOUT
        p.closed_at = datetime.now(timezone.utc)
        await self._db.update_position(p)

    async def _activate(self, p: Position, fill: Fill) -> Position | None:
        """Entry filled (fully or partly): re-anchor stop/target on the fill and put the stop on the exchange."""
        if fill.quantity <= 0:
            await self._cancel_pending(p)
            await self._say(AgentRole.EXECUTION, f"Compra de {p.symbol} não executada no prazo: cancelada.")
            return None
        meta = self._meta(p)
        p.quantity, p.entry_price, p.fees_usdt = fill.quantity, fill.avg_price, fill.fee_usdt
        p.stop_price = fill.avg_price - float(meta.get("sd", 0) or 0) if meta.get("sd") else p.stop_price
        p.take_profit_price = fill.avg_price + float(meta.get("td", 0) or 0) if meta.get("td") else p.take_profit_price
        p.status = PositionStatus.OPEN
        await self._db.update_position(p)
        await self._place_stop(p)
        await self._hub.publish("position", {"event": "opened", "position": p.model_dump(mode="json")})
        await self._say(AgentRole.EXECUTION,
                        f"LIVE BUY {p.symbol}: {p.quantity:.8g} @ {p.entry_price:.6g} (US$ {p.quantity * p.entry_price:.2f}). "
                        f"Stop {p.stop_price:.6g} {'na Binance ✔' if p.stop_order_id else '⚠ por software'} · "
                        f"alvo {p.take_profit_price:.6g}.")
        return p

    async def _place_stop(self, p: Position) -> None:
        assert self._spot is not None
        scid = client_id("s")
        p.setup = "|".join(x for x in (p.setup or "").split("|") if not x.startswith("scid=")) + f"|scid={scid}"
        await self._db.update_position(p)  # the stop's id is known before it is sent
        try:
            p.stop_order_id = await self._spot.place_stop(p.symbol, p.quantity, p.stop_price, scid)
        except Exception as exc:
            log.exception("book: exchange stop for %s refused", p.symbol)
            p.stop_order_id = None
            await self._hub.publish("risk", {"event": "stop_missing", "symbol": p.symbol, "stop": p.stop_price,
                                             "detail": str(exc)[:200]})
        await self._db.update_position(p)

    async def _reconcile_pending(self, p: Position) -> None:
        """A PENDING row: the order may or may not exist. Ask the exchange by our id; never assume."""
        assert self._spot is not None
        cid = self._meta(p).get("cid") or p.entry_order_id or ""
        order = await self._spot.find(p.symbol, cid)
        if order is None:
            await self._cancel_pending(p)
            return
        if order.get("status") not in ("closed", "canceled", "expired", "rejected"):
            await self._spot.cancel(p.symbol, str(order["id"]))
            order = await self._spot.order(p.symbol, str(order["id"]))
        await self._activate(p, await self._spot.fill_of(p.symbol, order, "buy"))

    async def reconcile(self) -> None:
        """At boot: resolve PENDING entries and recover stops sent just before a crash."""
        if not self._live or self._spot is None:
            return
        for p in await self.positions():
            try:
                if p.status is PositionStatus.PENDING:
                    await self._reconcile_pending(p)
                elif p.status is PositionStatus.OPEN and not p.stop_order_id and self._meta(p).get("scid"):
                    order = await self._spot.find(p.symbol, self._meta(p)["scid"])
                    if order and order.get("status") in ("open", "closed"):
                        p.stop_order_id = str(order["id"])
                        await self._db.update_position(p)
            except Exception:
                log.exception("book: reconcile of position %s failed", p.id)
                await self._hub.publish("risk", {"event": "reconcile_failed", "symbol": p.symbol, "position_id": p.id})

    async def _check_exchange_stop(self, p: Position) -> bool:
        """True if the exchange stop has fired (position finished). Re-places a stop that vanished."""
        assert self._spot is not None
        if p.stop_order_id:
            order = await self._spot.order(p.symbol, p.stop_order_id)
            status = order.get("status")
            if status == "closed":
                f = await self._spot.fill_of(p.symbol, order, "sell")
                await self._finish(p, ExitReason.STOP_LOSS, f.avg_price, f.fee_usdt)
                return True
            if status in ("canceled", "expired", "rejected"):
                p.stop_order_id = None
                await self._hub.publish("risk", {"event": "stop_vanished", "symbol": p.symbol, "status": status})
        if not p.stop_order_id:
            await self._place_stop(p)
        return False

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
            if p.status is not PositionStatus.OPEN:
                continue
            live = p.mode == "live" and self._live
            if live and await self._check_exchange_stop(p):
                continue
            t = tickers.get(p.symbol) or {}
            bid = float(t.get("bid") or t.get("last") or 0.0)
            if bid <= 0:
                continue
            meta = self._meta(p)
            trail = float(meta.get("trail", 0) or 0)
            atr = float(meta.get("atr", 0) or 0)
            new_stop = p.stop_price
            if meta.get("mgmt") == "be2r":  # swing: breakeven at +1R, trailing from +2R (peak tracked by the bid)
                peak = max(float(meta.get("peak", 0) or 0), p.entry_price, bid)
                if peak > float(meta.get("peak", 0) or 0):
                    self._set_meta(p, peak=f"{peak:.8g}")
                    await self._db.update_position(p)
                rt = 2 * (self._s.fee_rate + self._s.max_slippage_pct)
                new_stop = swing.new_stop(p.entry_price, p.stop_price, float(meta.get("r", 0) or 0), peak, atr,
                                          float(meta.get("low3", 0) or 0), rt)
                new_stop = min(new_stop, bid)  # never above the market: that would be an instant stop-out
            elif trail and atr:  # chandelier approximated with the polled bid (never loosens)
                new_stop = max(p.stop_price, bid - trail * atr)
            if new_stop > p.stop_price:
                p.stop_price = new_stop
                await self._db.update_position(p)
                if live and p.stop_order_id and self._spot is not None:  # move the exchange stop up
                    await self._spot.cancel(p.symbol, p.stop_order_id)
                    p.stop_order_id = None
                    await self._place_stop(p)
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
            if reason and live and self._spot is not None:
                f = await self._spot.sell_all(p.symbol, p.quantity, p.stop_order_id,
                                              urgent=reason in (ExitReason.STOP_LOSS, ExitReason.MANUAL))
                await self._finish(p, reason, f.avg_price, f.fee_usdt)
            elif reason:
                await self._close(p, reason, price * (1 - self._s.max_slippage_pct))

    async def _close(self, p: Position, reason: ExitReason, price: float) -> None:
        await self._finish(p, reason, price, price * p.quantity * self._s.fee_rate)

    async def _finish(self, p: Position, reason: ExitReason, price: float, fee: float) -> None:
        p.exit_price = price
        p.fees_usdt += fee
        p.pnl_usdt = (price - p.entry_price) * p.quantity - p.fees_usdt
        p.exit_reason = reason
        p.status = PositionStatus.CLOSED
        p.closed_at = datetime.now(timezone.utc)
        await self._db.update_position(p)
        await self._hub.publish("position", {"event": "closed", "position": p.model_dump(mode="json")})
        await self._say(AgentRole.EXECUTION, f"{p.mode.upper()} SELL {p.symbol} ({reason.value}) @ {price:.6g} · "
                                             f"PnL US$ {p.pnl_usdt:+.4f}")
        lock = await self._risk.on_trade_closed()
        if lock:
            await self._hub.publish("risk", {"event": "circuit_breaker", "lock_until": lock.isoformat()})

    # ------------------------------------------------------------------ loop & API

    def names(self) -> list[str]:
        sw = [f"swing-{s['tf']}-{s['nome']}" for s in swing.SETUPS] if self._s.book_swing else []
        return [short_name(c) for c in self.strategies] + sw

    def snapshot_light(self) -> dict[str, Any]:
        return {"strategies": self.names(), "universe": self.state.universe,
                "btc_regime_ok": self.state.btc_regime_ok, "last_run": self.state.last_run,
                "last_signals": self.state.last_signals[:10], "notes": self.state.notes,
                "confluence": self.state.confluence[:10]}

    async def snapshot(self) -> dict[str, Any]:
        return {**self.snapshot_light(), "risk_per_trade": self._s.book_risk_per_trade,
                "max_positions": self._s.book_max_positions,
                "open_positions": [json.loads(p.model_dump_json()) for p in await self.positions()]}

    async def run(self, stop: asyncio.Event) -> None:
        if not self.names():
            log.warning("book: no valid strategies configured; idle")
            return
        log.info("book: running %s (%s)", ", ".join(self.names()),
                 "REAL ORDERS" if self._live else "paper")
        await self.reconcile()
        while not stop.is_set():
            try:
                await self.supervise()
                signals = await self.evaluate()
                await self.try_entries(signals)
                await self.try_entries(await self.scan_confluence())
                trim_memory()
            except Exception:
                log.exception("book cycle failed")
            try:
                await asyncio.wait_for(stop.wait(), timeout=SUPERVISE_SEC)
            except TimeoutError:
                pass
