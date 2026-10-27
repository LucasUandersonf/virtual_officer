"""Swing lab: long-only spot setups held for at most 48 hours, on multidimensional data.

Data layers (all with multi-year free history, so everything is backtestable):

* price/volume on 1h bars for the liquid universe, 4h and 1d resampled from them;
* **order flow**: Binance's taker-buy volume per candle → aggressive-buyer ratio (TBR) and its z-score;
* **derivatives**: perpetual funding rate per coin (as of each bar) and its z-score;
* **sentiment**: Crypto Fear & Greed index (previous day's value);
* context: BTC regime, each coin's daily trend and relative-strength rank, market breadth.

Open interest and liquidations are *not* used: Binance only serves 30 days of history, so a rule
built on them cannot be validated over years.

Three layers of "intelligence", all walk-forward (fit on the past, trade the next window blind):

1. **Setups** — seven rule families × small parameter grids (scored like the main lab).
2. **AI filter (meta-labelling)** — a gradient-boosted classifier learns, from every past candidate
   trade of every family, which feature combinations precede winners; in each test window only the
   candidates it rates above a threshold (chosen on an inner validation split) are taken. A 48 h
   embargo separates training labels from the test window so no outcome leaks.
3. **Market scenarios** — k-means on daily market features (BTC trend and volatility, sentiment,
   funding, breadth), fitted on the first 18 months only; every setup is scored per scenario and a
   router enables in each window only the setups that had an edge in the current scenario before.

Dynamic risk: the same signals are replayed with fixed 0.6 % risk, volatility-scaled risk and a
drawdown throttle (risk halves while equity is > 5 % below its peak).
"""

from __future__ import annotations

import itertools
import logging
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mqmas.research.data import candidate_pool
from mqmas.research.engine import Costs, RiskRules, simulate_portfolio
from mqmas.research.lab import (
    CRITERIA,
    FIRST_TEST,
    HISTORY_START,
    deflated_sharpe,
    monte_carlo,
    test_windows,
    trade_tstat,
    window_slice,
)
from mqmas.research.strategies import (
    atr,
    btc_regime,
    daily_regime,
    daily_trend,
    ema,
    monthly_eligibility,
    relative_strength,
    rsi,
)
from mqmas.research.templates import Plan, structured_trades

log = logging.getLogger(__name__)

VERSION = 1
HOLD_BARS = {"1h": 48, "4h": 12, "1d": 2}  # 48 hours max on every timeframe
TF_HOURS = {"1h": 1, "4h": 4, "1d": 24}
EMBARGO = pd.Timedelta(hours=48)
SWING_CRITERIA = {**CRITERIA, "trades": 100}
CRASH_DAYS = ("2024-08-05", "2025-10-10")


# --------------------------------------------------------------------------- data


def fetch_klines(ex: Any, symbol: str, since_ms: int, root: Path) -> pd.DataFrame:
    """1h klines with taker-buy volume (ccxt's OHLCV drops it), cached on disk."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{symbol.replace('/', '_')}_1h_tb.npz"
    data = np.load(path)["rows"] if path.exists() else np.empty((0, 7))
    step = 3_600_000
    now = int(time.time() * 1000)
    cursor = int(data[-1, 0]) + step if len(data) else since_ms
    new: list[list[float]] = []
    market_id = symbol.replace("/", "")
    while cursor < now - step:
        for attempt in range(5):
            try:
                raw = ex.public_get_klines({"symbol": market_id, "interval": "1h", "startTime": cursor, "limit": 1000})
                break
            except Exception as exc:
                log.warning("klines %s failed (%s), retry %d", symbol, type(exc).__name__, attempt + 1)
                time.sleep(2 ** attempt)
        else:
            break
        rows = [[float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[9])]
                for r in raw if int(r[0]) + step <= now]
        if not rows:
            break
        new.extend(rows)
        cursor = int(rows[-1][0]) + step
    if new:
        data = np.vstack([data, np.asarray(new)]) if len(data) else np.asarray(new)
        np.savez_compressed(path, rows=data)
    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume", "taker_buy"])
    df.index = pd.to_datetime(df.pop("ts").astype("int64"), unit="ms", utc=True)
    return df[~df.index.duplicated(keep="last")].sort_index()


def resample(h1: pd.DataFrame, rule: str, per: int) -> pd.DataFrame:
    agg = h1.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                 "volume": "sum", "taker_buy": "sum"})
    n = h1["close"].resample(rule).count()
    return agg[n == per].dropna()


def fetch_funding(fx: Any, symbol: str, since_ms: int, root: Path) -> pd.Series:
    path = root / f"{symbol.replace('/', '_')}_funding.npz"
    data = np.load(path)["rows"] if path.exists() else np.empty((0, 2))
    cursor = int(data[-1, 0]) + 1 if len(data) else since_ms
    perp = f"{symbol}:USDT"
    new: list[list[float]] = []
    try:
        if perp not in fx.markets:
            return pd.Series(dtype=float)
        while cursor < int(time.time() * 1000):
            batch = fx.fetch_funding_rate_history(perp, since=cursor, limit=1000)
            if not batch:
                break
            new += [[float(r["timestamp"]), float(r["fundingRate"])] for r in batch]
            cursor = int(batch[-1]["timestamp"]) + 1
            if len(batch) < 1000:
                break
    except Exception as exc:
        log.warning("funding %s unavailable: %s", symbol, type(exc).__name__)
    if new:
        data = np.vstack([data, np.asarray(new)]) if len(data) else np.asarray(new)
        np.savez_compressed(path, rows=data)
    if not len(data):
        return pd.Series(dtype=float)
    s = pd.Series(data[:, 1], index=pd.to_datetime(data[:, 0].astype("int64"), unit="ms", utc=True))
    return s[~s.index.duplicated(keep="last")].sort_index()


def fetch_fear_greed() -> pd.Series:
    import httpx

    r = httpx.get("https://api.alternative.me/fng/", params={"limit": 0, "format": "json"}, timeout=30)
    r.raise_for_status()
    rows = r.json()["data"]
    s = pd.Series([float(x["value"]) for x in rows],
                  index=pd.to_datetime([int(x["timestamp"]) for x in rows], unit="s", utc=True).normalize())
    return s[~s.index.duplicated()].sort_index()


# --------------------------------------------------------------------------- features


FEATURES = [
    "ret_1", "ret_6", "ret_24", "rsi2", "rsi14", "atr_pct", "dist_ema20", "dist_ema50", "dist_ema200",
    "bb_pos", "vol_z", "tbr", "tbr_24", "tbr_z", "funding", "funding_z", "fng", "fng_chg3", "rs",
    "d_trend", "btc_ok", "breadth", "hour",
]


def tf_features(df: pd.DataFrame, ctx: pd.DataFrame, funding: pd.Series) -> pd.DataFrame:
    f = df.copy()
    c = f["close"]
    f["ret_1"], f["ret_6"], f["ret_24"] = c.pct_change(), c.pct_change(6), c.pct_change(24)
    f["rsi2"], f["rsi14"] = rsi(c, 2), rsi(c, 14)
    f["atr"] = atr(df)
    f["atr_pct"] = f["atr"] / c
    for n in (20, 50, 200):
        f[f"ema{n}"] = ema(c, n)
        f[f"dist_ema{n}"] = c / f[f"ema{n}"] - 1
    f["sma5"] = c.rolling(5).mean()
    mid, sd = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
    f["bb_pos"] = (c - mid) / (2 * sd).replace(0, np.nan)
    v = f["volume"]
    f["vol_z"] = (v - v.rolling(50).mean()) / v.rolling(50).std(ddof=0).replace(0, np.nan)
    f["tbr"] = (f["taker_buy"] / v.replace(0, np.nan)).clip(0, 1)
    f["tbr_24"] = f["tbr"].rolling(24).mean()
    f["tbr_z"] = (f["tbr"] - f["tbr"].rolling(100).mean()) / f["tbr"].rolling(100).std(ddof=0).replace(0, np.nan)
    f["hh20"] = f["high"].shift(1).rolling(20).max()
    tp = (f["high"] + f["low"] + c) / 3
    day = f.index.normalize()
    f["vwap"] = (tp * v).groupby(day).cumsum() / v.groupby(day).cumsum().replace(0, np.nan)
    if len(funding):
        fz = (funding - funding.rolling(90).mean()) / funding.rolling(90).std(ddof=0).replace(0, np.nan)
        known = pd.DataFrame({"funding": funding, "funding_z": fz})
        f = pd.merge_asof(f.sort_index(), known.sort_index(), left_index=True, right_index=True, direction="backward")
    else:
        f["funding"], f["funding_z"] = np.nan, np.nan
    day_ctx = ctx.reindex(f.index.normalize())
    for col in ("fng", "fng_chg3", "rs", "d_trend", "btc_ok", "breadth", "scenario", "d_regime"):
        f[col] = day_ctx[col].to_numpy() if col in day_ctx else np.nan
    f["hour"] = f.index.hour
    return f


# --------------------------------------------------------------------------- setups


PlanFn = Callable[[pd.DataFrame, dict[str, Any], str], Plan]


def _plan(f: pd.DataFrame, mask: pd.Series, p: dict[str, Any], tf: str, exit_sig: pd.Series | None = None) -> Plan:
    i = np.flatnonzero(mask.fillna(False).to_numpy(dtype=bool))
    return Plan(i, np.full(i.size, np.nan), target_r=p["r"], stop_atr=p["stop"],
                exit_sig=None if exit_sig is None else exit_sig.fillna(False).to_numpy(dtype=bool),
                hold=HOLD_BARS[tf])


def s_trend_dip(f: pd.DataFrame, p: dict[str, Any], tf: str) -> Plan:
    """Buy a short-term oversold dip (RSI-2) inside a daily uptrend; exit on the first close above SMA5."""
    m = (f["d_trend"] == 1) & (f["btc_ok"] == 1) & (f["rsi2"] < p["rsi"]) & (f["close"] > f["ema200"])
    return _plan(f, m, p, tf, exit_sig=f["close"] > f["sma5"])


def s_flow_breakout(f: pd.DataFrame, p: dict[str, Any], tf: str) -> Plan:
    """Breakout of the 20-bar high on a volume spike with aggressive buyers in control (order flow)."""
    m = ((f["close"] > f["hh20"]) & (f["vol_z"] > p["volz"]) & (f["tbr"] > p["tbr"]) & (f["d_trend"] == 1)
         & (f["btc_ok"] == 1))
    return _plan(f, m, p, tf)


def s_funding_reset(f: pd.DataFrame, p: dict[str, Any], tf: str) -> Plan:
    """Derivatives: shorts crowded (funding well below its norm) while the daily trend is up → squeeze."""
    m = ((f["funding_z"] < -p["fz"]) & (f["d_trend"] == 1) & (f["close"] > f["ema20"])
         & (f["close"].shift(1) <= f["ema20"].shift(1)))
    return _plan(f, m, p, tf)


def s_fear_reversal(f: pd.DataFrame, p: dict[str, Any], tf: str) -> Plan:
    """Sentiment: extreme fear, then price reclaims its EMA20 with buyers stepping in."""
    m = ((f["fng"] <= p["fng"]) & (f["close"] > f["ema20"]) & (f["close"].shift(1) <= f["ema20"].shift(1))
         & (f["tbr"] > 0.5))
    return _plan(f, m, p, tf)


def s_capitulation(f: pd.DataFrame, p: dict[str, Any], tf: str) -> Plan:
    """Sell climax: a bar falling > k ATR on heavy volume with sellers dominant, then a green bar above its mid."""
    prev_drop = (f["open"].shift(1) - f["close"].shift(1)) > p["k"] * f["atr"].shift(1)
    climax = prev_drop & (f["vol_z"].shift(1) > 2) & (f["tbr"].shift(1) < 0.45)
    m = climax & (f["close"] > f["open"]) & (f["close"] > (f["high"].shift(1) + f["low"].shift(1)) / 2)
    return _plan(f, m, p, tf)


def s_momentum_2d(f: pd.DataFrame, p: dict[str, Any], tf: str) -> Plan:
    """Daily momentum continuation: strong close near the high, top relative strength, held ≤ 2 days."""
    rng = (f["high"] - f["low"]).replace(0, np.nan)
    m = (((f["close"] - f["low"]) / rng > 0.8) & (f["ret_1"] > p["k"] * f["atr_pct"].shift(1))
         & (f["rs"] >= p["rs"]) & (f["btc_ok"] == 1) & (f["close"] > f["ema50"]))
    return _plan(f, m, p, tf)


def s_vwap_pullback(f: pd.DataFrame, p: dict[str, Any], tf: str) -> Plan:
    """Intraday trend: touch of the session VWAP from above in a daily uptrend, buyers returning."""
    m = ((f["d_trend"] == 1) & (f["low"] <= f["vwap"]) & (f["close"] > f["vwap"]) & (f["dist_ema50"] > 0)
         & (f["tbr"] > p["tbr"]) & (f["tbr"] > f["tbr_24"]))
    return _plan(f, m, p, tf)


@dataclass(frozen=True)
class SwingSetup:
    name: str
    title: str
    scenario_hint: str
    fn: PlanFn
    tfs: tuple[str, ...]
    grid: dict[str, list[Any]]

    def configs(self) -> list[tuple[str, str, dict[str, Any]]]:
        out = []
        for tf in self.tfs:
            for combo in itertools.product(*self.grid.values()):
                p = dict(zip(self.grid, combo, strict=True))
                out.append((f"sw:{self.name}|{tf}|" + ",".join(f"{k}={v}" for k, v in p.items()), tf, p))
        return out


EXITS = {"stop": [1.0, 1.5, 2.5], "r": [1.0, 1.5, 2.5]}
SETUPS: tuple[SwingSetup, ...] = (
    SwingSetup("trend_dip", "Recuo RSI-2 em tendência diária", "alta", s_trend_dip, ("1h", "4h"),
               {"rsi": [5, 10, 20], **EXITS}),
    SwingSetup("flow_breakout", "Rompimento com fluxo comprador", "alta", s_flow_breakout, ("1h", "4h"),
               {"volz": [1.0, 2.0], "tbr": [0.55, 0.6], **EXITS}),
    SwingSetup("funding_reset", "Funding negativo em tendência (squeeze)", "alta com medo", s_funding_reset,
               ("1h", "4h"), {"fz": [1.0, 2.0], **EXITS}),
    SwingSetup("fear_reversal", "Medo extremo + retomada da EMA20", "queda/pânico", s_fear_reversal, ("1h", "4h"),
               {"fng": [20, 30], **EXITS}),
    SwingSetup("capitulation", "Clímax de venda + candle de reversão", "volátil", s_capitulation, ("1h", "4h"),
               {"k": [2.0, 3.0], **EXITS}),
    SwingSetup("momentum_2d", "Momentum diário de 2 dias", "alta", s_momentum_2d, ("1d",),
               {"k": [1.0, 1.5], "rs": [0.6, 0.8], **EXITS}),
    SwingSetup("vwap_pullback", "Pullback na VWAP do dia com fluxo", "alta/lateral", s_vwap_pullback, ("1h",),
               {"tbr": [0.5, 0.55], **EXITS}),
)
CANONICAL = {"stop": 1.5, "r": 1.5}  # bracket used to label candidates for the AI filter


# --------------------------------------------------------------------------- lab


class SwingLab:
    def __init__(self, root: Path, pool_size: int = 40, top_n: int = 20) -> None:
        self.root = root
        self.pool_size, self.top_n = pool_size, top_n
        self.costs = Costs()
        self.rules = RiskRules(equity0=10_000.0)

    def load(self, ex: Any = None, fx: Any = None) -> None:
        if ex is None:
            import ccxt

            ex = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
            fx = ccxt.binanceusdm({"enableRateLimit": True})
        self.pool = candidate_pool(ex, self.pool_size)
        since = int((HISTORY_START - pd.Timedelta(days=260)).timestamp() * 1000)
        cache = self.root / "swing"
        self.h1: dict[str, pd.DataFrame] = {}
        self.funding: dict[str, pd.Series] = {}
        if fx is not None:
            try:
                fx.load_markets()
            except Exception as exc:
                log.warning("futures markets unavailable: %s", type(exc).__name__)
                fx = None
        for sym in self.pool:
            df = fetch_klines(ex, sym, since, cache)
            if len(df) > 24 * 300:
                self.h1[sym] = df
                self.funding[sym] = fetch_funding(fx, sym, since, cache) if fx is not None else pd.Series(dtype=float)
        self.d1 = {s: resample(h, "1D", 24) for s, h in self.h1.items()}
        try:
            self.fng = fetch_fear_greed()
        except Exception as exc:
            log.warning("fear & greed unavailable: %s", type(exc).__name__)
            self.fng = pd.Series(dtype=float)
        self.end = max(h.index[-1] for h in self.h1.values()).normalize()
        log.info("swing data: %d coins, %d with funding, F&G %d days", len(self.h1),
                 sum(1 for s in self.funding.values() if len(s)), len(self.fng))
        self._context()

    def _context(self) -> None:
        """Per-day context known at the open of each day."""
        self.eligible = monthly_eligibility(self.d1, self.top_n, 0.02, 0.2)
        rs = relative_strength(self.d1, self.eligible)
        btc_ok = btc_regime(self.d1["BTC/USDT"]).astype(float)
        closes = pd.DataFrame({s: d["close"] for s, d in self.d1.items()})
        breadth = (closes > closes.ewm(span=50, adjust=False).mean()).mean(axis=1).shift(1)
        fng = self.fng.reindex(closes.index).ffill().shift(1)
        fund_mkt = pd.DataFrame({s: f.resample("1D").mean() for s, f in self.funding.items() if len(f)})
        fund_mkt = fund_mkt.reindex(closes.index).mean(axis=1).shift(1) if len(fund_mkt.columns) else \
            pd.Series(0.0, index=closes.index)
        btc = closes["BTC/USDT"]
        self.market = pd.DataFrame({
            "btc_ret20": (btc / btc.shift(20) - 1).shift(1),
            "btc_vol20": btc.pct_change().rolling(20).std().shift(1) * math.sqrt(365),
            "fng": fng, "funding": fund_mkt, "breadth": breadth})
        # a missing optional layer must not erase the other dimensions: neutral values instead
        self.market = self.market.fillna({"fng": 50.0, "funding": 0.0}).dropna()
        self.scenarios, self.scenario_names = self._fit_scenarios()
        self.ctx: dict[str, pd.DataFrame] = {}
        for s, d in self.d1.items():
            self.ctx[s] = pd.DataFrame({
                "fng": fng, "fng_chg3": fng - fng.shift(3), "rs": rs[s] if s in rs else np.nan,
                "d_trend": daily_trend(d).astype(float), "btc_ok": btc_ok, "breadth": breadth,
                "scenario": self.scenarios, "d_regime": daily_regime(d)}, index=closes.index)

    def _fit_scenarios(self, k: int = 4) -> tuple[pd.Series, dict[int, str]]:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        m = self.market
        train = m[m.index < FIRST_TEST - pd.DateOffset(months=0)]
        train = train[train.index >= HISTORY_START]
        scaler = StandardScaler().fit(train)
        km = KMeans(n_clusters=k, n_init=20, random_state=7).fit(scaler.transform(train))
        labels = pd.Series(km.predict(scaler.transform(m)), index=m.index)
        cents = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=m.columns)
        names: dict[int, str] = {}
        for i, c in cents.iterrows():
            trend = "alta" if c["btc_ret20"] > 0.05 else ("queda" if c["btc_ret20"] < -0.05 else "lateral")
            mood = "euforia" if c["fng"] >= 60 else ("medo" if c["fng"] <= 40 else "neutro")
            vol = "volátil" if c["btc_vol20"] > m["btc_vol20"].median() * 1.25 else "calmo"
            names[int(i)] = f"C{int(i) + 1} {trend} · {mood} · {vol} (BTC20d {c['btc_ret20']:+.0%}, F&G {c['fng']:.0f}, " \
                            f"vol {c['btc_vol20']:.0%}, amplitude {c['breadth']:.0%})"
        return labels, names

    # ------------------------------------------------------------------ trade tables

    def trade_tables(self) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]], pd.DataFrame]:
        rows: dict[str, list[pd.DataFrame]] = {}
        groups: dict[str, list[str]] = {}
        cands: list[pd.DataFrame] = []
        for st in SETUPS:
            for key, tf, _p in st.configs():
                rows[key] = []
                groups.setdefault(f"sw:{st.name}|{tf}", []).append(key)
        for coin, h1 in self.h1.items():
            frames = {"1h": h1, "4h": resample(h1, "4h", 4), "1d": self.d1[coin]}
            for tf in ("1h", "4h", "1d"):
                if not any(tf in st.tfs for st in SETUPS):
                    continue
                df = frames[tf]
                df = df[df.index >= HISTORY_START - pd.Timedelta(days=60)]
                f = tf_features(df, self.ctx[coin], self.funding.get(coin, pd.Series(dtype=float)))
                months = f.index.tz_localize(None).to_period("M")
                elig = np.array([coin in self.eligible.get(m, ()) for m in months], dtype=bool)
                o, h, lo, cl, a = (f[k].to_numpy() for k in ("open", "high", "low", "close", "atr"))
                warm = np.arange(len(f)) >= 200
                xs = f[FEATURES].to_numpy(dtype=float)
                for st in SETUPS:
                    if tf not in st.tfs:
                        continue
                    for key, ktf, p in st.configs():
                        if ktf != tf:
                            continue
                        plan = st.fn(f, p, tf)
                        keep = elig[plan.sig_idx] & warm[plan.sig_idx]
                        plan.sig_idx, plan.stop = plan.sig_idx[keep], plan.stop[keep]
                        t = structured_trades(o, h, lo, cl, a, plan, self.costs)
                        if t["entry"].size == 0:
                            continue
                        ein, eout = t["entry"].astype(int), t["exit"].astype(int)
                        frame = pd.DataFrame({
                            "coin": coin, "t_in": f.index[ein],
                            "t_out": f.index[eout] + pd.Timedelta(hours=TF_HOURS[tf]),
                            "ret": t["ret"], "stop_pct": t["stop_pct"], "reason": t["reason"],
                            "scenario": f["scenario"].to_numpy()[ein], "atr_pct": f["atr_pct"].to_numpy()[ein - 1]})
                        rows[key].append(frame)
                        if p["stop"] == CANONICAL["stop"] and p["r"] == CANONICAL["r"]:
                            feats = pd.DataFrame(xs[ein - 1], columns=FEATURES)
                            feats["family"] = SETUPS.index(st)
                            feats["tf"] = TF_HOURS[tf]
                            cands.append(pd.concat([frame.drop(columns=["atr_pct"]).reset_index(drop=True), feats],
                                                   axis=1)
                                         .assign(setup=f"{st.name}|{tf}"))
        empty = pd.DataFrame(columns=["coin", "t_in", "t_out", "ret", "stop_pct", "reason", "scenario", "atr_pct"])
        table = {k: (pd.concat(v, ignore_index=True) if v else empty) for k, v in rows.items()}
        cand = pd.concat(cands, ignore_index=True) if cands else pd.DataFrame()
        return table, groups, cand

    # ------------------------------------------------------------------ walk-forward layers

    def walk_forward(self, groups: dict[str, list[str]], table: dict[str, pd.DataFrame]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, keys in groups.items():
            parts, picks = [], []
            for s, e in test_windows(self.end):
                best, best_t = None, -math.inf
                for k in keys:
                    sc = trade_tstat(window_slice(table[k], HISTORY_START, s)["ret"].to_numpy(dtype=float))
                    if sc > best_t:
                        best, best_t = k, sc
                if best is None:
                    continue
                picks.append({"window": f"{s.date()}→{e.date()}", "start": s, "end": e, "config": best})
                parts.append(window_slice(table[best], s, e))
            oos = [window_slice(table[k], FIRST_TEST, self.end)["ret"] for k in keys]
            robust = float(np.mean([(r[r > 0].sum() / max(-r[r <= 0].sum(), 1e-12)) > 1 for r in oos if len(r)]))
            out[name] = {"picks": picks, "oos": pd.concat(parts) if parts else None, "family_robustness": robust}
        return out

    def ai_filter(self, cand: pd.DataFrame) -> tuple[pd.DataFrame | None, dict[str, Any]]:
        """Meta-labelling with a gradient-boosted classifier, retrained before every test window."""
        from sklearn.ensemble import HistGradientBoostingClassifier

        if cand.empty:
            return None, {}
        cand = cand.sort_values("t_in").reset_index(drop=True)
        # a data layer that is empty (e.g. sentiment API down) would break the binning: drop it
        cols = [c for c in FEATURES + ["family", "tf"] if cand[c].notna().mean() > 0.05 and cand[c].nunique() > 1]
        y = (cand["ret"] > 0).astype(int).to_numpy()
        taken, info = [], []
        importances: dict[str, Any] = {}
        for s, e in test_windows(self.end):
            tr = cand[cand["t_out"] < s - EMBARGO]
            te = cand[(cand["t_in"] >= s) & (cand["t_in"] < e)]
            if len(tr) < 500 or te.empty:
                continue
            split = int(len(tr) * 0.8)
            fit, val = tr.iloc[:split], tr.iloc[split:]
            fit = fit[fit["t_out"] < val["t_in"].min() - EMBARGO]
            model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                                   min_samples_leaf=80, random_state=7)
            model.fit(fit[cols].to_numpy(dtype=float), y[fit.index])
            pv = model.predict_proba(val[cols].to_numpy(dtype=float))[:, 1]
            best_th, best_score = None, -math.inf
            for th in np.quantile(pv, [0.5, 0.6, 0.7, 0.8, 0.9]):
                sel = val["ret"].to_numpy()[pv >= th]
                if sel.size >= 50 and sel.mean() > best_score:
                    best_th, best_score = float(th), float(sel.mean())
            if best_th is None or best_score <= 0:
                info.append({"window": f"{s.date()}→{e.date()}", "trades": 0, "note": "sem limiar lucrativo na validação"})
                continue
            model.fit(tr[cols].to_numpy(dtype=float), y[tr.index])  # refit on all past data
            pt = model.predict_proba(te[cols].to_numpy(dtype=float))[:, 1]
            chosen = te[pt >= best_th]
            # one candidate per coin and entry time (several families can flag the same bar)
            chosen = chosen.sort_values("t_in").drop_duplicates(["coin", "t_in"])
            taken.append(chosen)
            info.append({"window": f"{s.date()}→{e.date()}", "threshold": round(best_th, 3), "trades": len(chosen),
                         "val_expectancy": round(best_score, 5)})
            last = (model, val, y[val.index])
        if not taken:
            return None, {"windows": info}
        try:
            from sklearn.inspection import permutation_importance

            model, val, yv = last
            pi = permutation_importance(model, val[cols].to_numpy(dtype=float), yv, n_repeats=5, random_state=7,
                                        scoring="roc_auc")
            importances = dict(sorted(zip(cols, pi.importances_mean.round(4).tolist(), strict=True),
                                      key=lambda x: -x[1])[:10])
        except Exception as exc:
            importances = {"erro": str(exc)[:100]}
        return pd.concat(taken, ignore_index=True), {"windows": info, "top_features": importances}

    def scenario_router(self, wf: dict[str, Any], table: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame | None,
                                                                                         dict[str, Any]]:
        parts, maps = [], []
        for name, r in wf.items():
            for pk in r["picks"]:
                tr = table[pk["config"]]
                past = window_slice(tr, HISTORY_START, pk["start"])
                ok = [int(g) for g, grp in past.groupby("scenario")
                      if trade_tstat(grp["ret"].to_numpy(dtype=float)) >= 1.5]
                test = window_slice(tr, pk["start"], pk["end"])
                parts.append(test[test["scenario"].isin(ok)].assign(setup=name))
                maps.append({"setup": name, "window": pk["window"], "scenarios": ok})
        return (pd.concat(parts, ignore_index=True) if parts else None), {"allowed": maps}

    # ------------------------------------------------------------------ evaluation

    def evaluate(self, name: str, trades: pd.DataFrame, trial_srs: list[float], extra: dict[str, Any]) -> dict[str, Any]:
        res = simulate_portfolio(trades, self.rules, self.costs, FIRST_TEST, self.end)
        m = res.metrics
        stressed = simulate_portfolio(trades.assign(ret=trades["ret"] - self.costs.round_trip), self.rules,
                                      self.costs, FIRST_TEST, self.end).metrics
        daily = res.equity.groupby(res.equity.index.normalize()).last().reindex(
            pd.date_range(FIRST_TEST, self.end, freq="D", tz="UTC")).ffill().fillna(self.rules.equity0)
        monthly = daily.resample("ME").last().pct_change().dropna()
        dret = daily.pct_change().fillna(0)
        m["dsr"] = deflated_sharpe(m["daily_sr"], m["days"], m["daily_skew"], m["daily_kurt"], trial_srs) \
            if m["days"] > 2 and np.isfinite(m["daily_sr"]) else 0.0
        m["stressed_profit_factor"] = stressed["profit_factor"]
        m["trades_per_month"] = m["trades"] / max(m["months"], 1)
        m["avg_hold_hours"] = float(((trades["t_out"] - trades["t_in"]).dt.total_seconds() / 3600).mean()) \
            if len(trades) else 0.0
        sizing = {}
        vol_mult = (trades["atr_pct"].median() / trades["atr_pct"]).clip(0.5, 1.0).fillna(1.0)
        for label, tr, rules in (
                ("fixo 0,6%", trades, self.rules),
                ("volatilidade", trades.assign(risk_mult=vol_mult), self.rules),
                ("drawdown", trades, RiskRules(equity0=self.rules.equity0, dd_throttle=0.05)),
                ("vol+drawdown", trades.assign(risk_mult=vol_mult), RiskRules(equity0=self.rules.equity0,
                                                                              dd_throttle=0.05)),
                ("fixo 1%", trades, RiskRules(equity0=self.rules.equity0, risk_per_trade=0.01))):
            mm = simulate_portfolio(tr, rules, self.costs, FIRST_TEST, self.end).metrics
            sizing[label] = {"monthly_mean": mm["monthly_mean"], "max_drawdown": mm["max_drawdown"],
                             "sharpe": mm["sharpe"]}
        by_scen = {}
        for g, grp in trades[(trades["t_in"] >= FIRST_TEST)].groupby("scenario"):
            r = grp["ret"].to_numpy(dtype=float)
            loss = -r[r <= 0].sum()
            by_scen[self.scenario_names.get(int(g), str(g))] = {
                "trades": int(r.size), "win_rate": float((r > 0).mean()), "expectancy": float(r.mean()),
                "profit_factor": float(min(r[r > 0].sum() / loss, 99)) if loss > 0 else 99.0}
        years = {str(y): float(v) for y, v in (1 + monthly).groupby(monthly.index.year).prod().sub(1).items()}
        crash = {d: float(dret.get(pd.Timestamp(d, tz="UTC"), 0.0)) for d in CRASH_DAYS}
        fails = [f"{k}={m.get(k, 0):.3f} (limite {lim})" for k, lim in SWING_CRITERIA.items()
                 if (m.get(k, 0) > lim if k == "max_drawdown" else m.get(k, 0) < lim)]
        return {"name": name, "approved": not fails, "fails": fails, "metrics": m, "sizing": sizing,
                "by_scenario": by_scen, "by_year": years, "crash_days": crash, "monte_carlo": monte_carlo(monthly),
                **extra}

    def run(self, ex: Any = None, fx: Any = None) -> dict[str, Any]:
        t0 = time.time()
        self.load(ex, fx)
        table, groups, cand = self.trade_tables()
        log.info("swing tables: %d configs, %d AI candidates in %.0fs", len(table), len(cand), time.time() - t0)
        is_sr = {k: simulate_portfolio(v, self.rules, self.costs, HISTORY_START, FIRST_TEST).metrics["daily_sr"]
                 for k, v in table.items()}
        trials = list(is_sr.values())
        wf = self.walk_forward(groups, table)
        results = []
        for name, r in wf.items():
            if r["oos"] is None or r["oos"].empty:
                continue
            st = SETUPS[[s.name for s in SETUPS].index(name.split(":")[1].split("|")[0])]
            results.append(self.evaluate(name, r["oos"], trials, {
                "title": st.title, "scenario_hint": st.scenario_hint, "family_robustness": r["family_robustness"],
                "live_config": r["picks"][-1]["config"], "picks": [p["config"] for p in r["picks"]]}))
        ai, ai_info = self.ai_filter(cand)
        if ai is not None and len(ai):
            results.append(self.evaluate("IA: filtro de aprendizado de máquina", ai, trials + [0.0] * 50,
                                         {"title": "Todos os setups, filtrados pela IA", "ai": ai_info}))
        routed, rmap = self.scenario_router(wf, table)
        if routed is not None and len(routed):
            results.append(self.evaluate("Roteador de cenários", routed, trials,
                                         {"title": "Cada setup só no cenário em que provou vantagem", **rmap}))
        results.sort(key=lambda x: (x["approved"], x["metrics"]["sortino"]), reverse=True)
        scen_counts = self.scenarios[self.scenarios.index >= FIRST_TEST].value_counts().to_dict()
        return {"version": VERSION, "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "elapsed_sec": round(time.time() - t0), "coins": list(self.h1), "trials": len(trials),
                "oos_period": [str(FIRST_TEST.date()), str(self.end.date())], "rules": asdict(self.rules),
                "criteria": SWING_CRITERIA,
                "scenarios": {self.scenario_names[int(k)]: int(v) for k, v in scen_counts.items()},
                "today_scenario": self.scenario_names.get(int(self.scenarios.iloc[-1]), "?"),
                "data_layers": {"funding_coins": sum(1 for s in self.funding.values() if len(s)),
                                "fear_greed_days": len(self.fng)},
                "results": results}


def summarize(rep: dict[str, Any]) -> list[str]:
    lines = [f"SWING v{rep['version']} | OOS {rep['oos_period'][0]}→{rep['oos_period'][1]} | {rep['trials']} configs | "
             f"{len(rep['coins'])} moedas | camadas {rep['data_layers']} | {rep['elapsed_sec']}s",
             f"cenário de hoje: {rep['today_scenario']}"]
    lines += [f"cenário: {k} → {v} dias" for k, v in rep["scenarios"].items()]
    for r in rep["results"]:
        m = r["metrics"]
        lines.append(
            f"{'APROVADO ' if r['approved'] else 'reprovado'} {r['name']} | trades/mês {m['trades_per_month']:.1f} "
            f"hold {m['avg_hold_hours']:.0f}h | mês {m['monthly_mean']:+.2%} pior {m['monthly_worst']:+.2%} "
            f"meses+ {m['positive_months']:.0%} | acerto {m['win_rate']:.0%} PF {m['profit_factor']:.2f} "
            f"payoff {m['payoff']:.2f} DD {m['max_drawdown']:.1%} Sharpe {m['sharpe']:.2f} Sortino {m['sortino']:.2f} "
            f"DSR {m['dsr']:.2f} PFx2 {m['stressed_profit_factor']:.2f} robustez {r.get('family_robustness', 0):.0%} | "
            f"falhas: {'; '.join(r['fails']) or '-'}")
        if r.get("live_config"):
            lines.append(f"    config: {r['live_config']}")
        lines.append("    tamanho: " + " | ".join(f"{k}: {v['monthly_mean']:+.2%}/mês DD {v['max_drawdown']:.1%}"
                                                   for k, v in r["sizing"].items()))
        lines.append("    cenários: " + " | ".join(f"{k.split(' (')[0]}: n={v['trades']} PF {v['profit_factor']:.2f}"
                                                    for k, v in r["by_scenario"].items()))
        lines.append("    anos: " + " | ".join(f"{k}: {v:+.1%}" for k, v in r["by_year"].items())
                     + f" | crashes: {r['crash_days']}")
        if "ai" in r:
            lines.append(f"    IA: {r['ai'].get('top_features')} | janelas: {r['ai'].get('windows')}")
    return lines
