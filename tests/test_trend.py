from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mqmas.research.engine import Costs
from mqmas.research.trend_study import simulate
from mqmas.runner import run as runner_run
from mqmas.trend.live import TrendAllocator, closed_day
from mqmas.trend.model import (
    TrendParams,
    donchian_model,
    eligibility,
    ensemble_signal,
    target_weights,
    weights_frame,
)
from mqmas.trend.rebalance import Order, next_stop, plan_orders

NOW = int(pd.Timestamp("2026-10-27 01:00", tz="UTC").timestamp() * 1000)


def frame(n: int, drift: float, seed: int, end: str = "2026-10-26", vol: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp(end, tz="UTC"), periods=n, freq="1D")
    c = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.01, "low": np.minimum(o, c) * 0.99, "close": c,
                         "volume": 1e6 * (1 + seed)}, index=idx)


# ---------------------------------------------------------------- model


def test_donchian_enters_on_new_high_and_exits_under_the_ratcheted_midline():
    c = np.array([10, 11, 12, 13, 14, 13.9, 13.0, 12.4, 15.0], dtype=float)
    s = donchian_model(c, 3)
    assert s[:2].tolist() == [0, 0]  # not enough history
    assert s[2:6].tolist() == [1, 1, 1, 1]  # new 3-day highs, then above the midline
    assert s[6] == 0.0  # 13.0 < yesterday's stop: mid of the 13–14 channel = 13.5
    assert s[7] == 0.0
    assert s[8] == 1.0  # a fresh 3-day high re-enters


def test_ensemble_signal_is_full_in_a_steady_uptrend_and_zero_in_a_downtrend():
    up = ensemble_signal(pd.Series(np.linspace(100, 400, 500)))
    down = ensemble_signal(pd.Series(np.linspace(400, 100, 500)))
    assert up.iloc[-1] == pytest.approx(1.0) and 0 <= up.min() <= up.max() <= 1
    assert down.iloc[-1] == 0.0


def test_weights_are_long_only_capped_and_scaled_to_the_vol_target():
    cols = ["A", "B", "C"]
    sig = pd.Series([1.0, 1.0, 0.0], index=cols)
    vol = pd.Series([0.5, 1.0, 0.8], index=cols)
    elig = pd.Series([True, True, True], index=cols)
    w = target_weights(sig, vol, elig, None, TrendParams(max_weight=0.35))
    assert w["C"] == 0 and w["A"] == pytest.approx(0.35) and w["B"] == pytest.approx((1 / 1.0) / (1 / .5 + 1 + 1 / .8))
    cov = np.diag((vol.to_numpy() ** 2) / 365)
    scaled = target_weights(sig, vol, elig, cov, TrendParams(vol_target=0.1, max_weight=0.35))
    assert np.sqrt(scaled.to_numpy() @ cov @ scaled.to_numpy() * 365) == pytest.approx(0.1)
    assert target_weights(sig, vol, pd.Series([False] * 3, index=cols), None, TrendParams()).sum() == 0


def test_universe_is_point_in_time_and_needs_a_year_of_history():
    daily = {"OLD/USDT": frame(800, 0.001, 1), "NEW/USDT": frame(200, 0.001, 9)}
    el = eligibility(daily, 10)
    assert el["OLD/USDT"].iloc[-1] and not el["NEW/USDT"].any()


# ---------------------------------------------------------------- orders


def test_plan_orders_respects_the_minimum_the_band_and_cash():
    orders = plan_orders({"A": 0.5, "B": 0.03, "C": 0.2}, {"B": 10.0, "C": 21.0, "D": 4.0}, 65.0, band=0.25)
    by = {(o.symbol, o.side): o for o in orders}
    assert by[("B", "sell")].close_all  # 3 % of 100 = 3 USDT < 5: flat
    assert by[("D", "sell")].close_all  # not a target anymore
    assert ("C", "sell") not in by and ("C", "buy") not in by  # 21 vs 20: inside the band
    assert by[("A", "buy")].value == pytest.approx(50.0)
    assert [o.side for o in orders][:2] == ["sell", "sell"]
    assert plan_orders({"A": 1.0}, {}, 3.0) == []  # below the exchange minimum


def test_partial_sell_leaving_dust_sells_everything():
    [o] = plan_orders({"A": 0.06}, {"A": 30.0}, 70.0)  # target 6, sell 24 → 6 left: kept
    assert o == Order("A", "sell", pytest.approx(24.0), False)
    [o] = plan_orders({"A": 0.051}, {"A": 30.0}, 70.0, band=0.0, min_order=5.0)
    assert not o.close_all


def test_disaster_stop_only_moves_up():
    assert next_stop(None, 100.0, 5.0, 6.0) == 70.0
    assert next_stop(70.0, 90.0, 5.0, 6.0) == 70.0
    assert next_stop(70.0, 120.0, 5.0, 6.0) == 90.0


def test_simulated_account_without_signals_keeps_its_cash():
    daily = {"BTC/USDT": frame(500, -0.002, 1), "ETH/USDT": frame(500, -0.002, 2)}
    w = pd.DataFrame(0.0, index=daily["BTC/USDT"].index, columns=list(daily))
    res = simulate(daily, w, w, 100.0, Costs(), daily["BTC/USDT"].index[10])
    assert res["equity"].iloc[-1] == pytest.approx(100.0) and res["fees"] == 0


def test_simulated_account_pays_fees_and_tracks_round_trips():
    daily = {"BTC/USDT": frame(900, 0.003, 1), "ETH/USDT": frame(900, 0.003, 2)}
    w, sig = weights_frame(daily, TrendParams(top_n=2))
    res = simulate(daily, w, sig, 1000.0, Costs(), daily["BTC/USDT"].index[400])
    assert res["fees"] > 0 and res["exposure"] > 0.2


# ---------------------------------------------------------------- live allocator (paper)


class FakeEx:
    def __init__(self, daily: dict[str, pd.DataFrame]) -> None:
        self.daily = daily
        self.markets: dict = {}
        self.hourly_low: dict[str, float] = {}

    async def load_markets(self):
        self.markets = {s: {"spot": True, "active": True, "quote": "USDT", "base": s.split("/")[0]} for s in self.daily}
        return self.markets

    async def fetch_tickers(self, syms=None):
        return {s: {"quoteVolume": float(d["volume"].iloc[-1] * d["close"].iloc[-1])} for s, d in self.daily.items()}

    async def fetch_ticker(self, sym):
        p = float(self.daily[sym]["close"].iloc[-1])
        return {"bid": p, "ask": p, "last": p}

    async def fetch_ohlcv(self, sym, tf, since=None, limit=None):
        if tf == "1d":
            d = self.daily[sym].iloc[-(limit or 1000):]
            return [[int(t.timestamp() * 1000), *r] for t, r in zip(d.index, d.to_numpy().tolist(), strict=True)]
        p = float(self.daily[sym]["close"].iloc[-1])
        low = self.hourly_low.get(sym, p)
        t0 = NOW - 2 * 3_600_000
        return [[t0, p, p, low, p, 1.0], [t0 + 3_600_000, p, p, p, p, 1.0]]


def market() -> dict[str, pd.DataFrame]:
    return {"BTC/USDT": frame(900, 0.004, 1), "ETH/USDT": frame(900, 0.004, 2), "SOL/USDT": frame(900, 0.004, 3),
            "DOGE/USDT": frame(900, -0.004, 4)}


@pytest.fixture
def trend_settings(settings):
    return settings.model_copy(update={"trend_capital_usdt": 200.0, "trend_top_n": 4})


async def make_alloc(settings, ex):
    from mqmas.persistence.db import Database

    db = Database(settings.db_path)
    await db.connect()
    return db, TrendAllocator(settings, db, ex)


@pytest.mark.asyncio
async def test_paper_run_buys_the_trends_once_per_day_and_persists(trend_settings):
    ex = FakeEx(market())
    await ex.load_markets()
    db, alloc = await make_alloc(trend_settings, ex)
    rep = await alloc.run_once(NOW)
    assert rep["last_close"] == closed_day(NOW) == "2026-10-26"
    assert set(rep["holdings"]) and "DOGE/USDT" not in rep["holdings"]  # downtrend: no weight
    assert all(h["stop"] for h in rep["holdings"].values())
    assert rep["equity"] == pytest.approx(200.0, rel=0.02)  # only fees and slippage spent
    alloc2 = TrendAllocator(trend_settings, db, ex)
    rep2 = await alloc2.run_once(NOW + 3_600_000)  # an hour later: state reloaded, no second rebalance
    assert rep2["holdings"].keys() == rep["holdings"].keys() and not any("COMPRA" in n for n in rep2["notes"])
    await db.close()


@pytest.mark.asyncio
async def test_paper_stop_closes_the_position_and_blocks_the_coin(trend_settings):
    ex = FakeEx(market())
    await ex.load_markets()
    db, alloc = await make_alloc(trend_settings, ex)
    rep = await alloc.run_once(NOW)
    sym, h = next(iter(rep["holdings"].items()))
    ex.hourly_low[sym] = h["stop"] * 0.9
    alloc2 = TrendAllocator(trend_settings, db, ex)
    rep2 = await alloc2.run_once(NOW + 3_600_000)
    assert sym not in rep2["holdings"] and sym in rep2["blocked"]
    assert alloc2.state.trips[-1]["reason"] == "stop"
    await db.close()


@pytest.mark.asyncio
async def test_kill_switch_and_drawdown_halt_sell_everything(trend_settings):
    ex = FakeEx(market())
    await ex.load_markets()
    db, alloc = await make_alloc(trend_settings, ex)
    await alloc.run_once(NOW)
    killed = trend_settings.model_copy(update={"trend_kill_switch": True})
    rep = await TrendAllocator(killed, db, ex).run_once(NOW + 3_600_000)
    assert rep["halted"] == "kill_switch" and not rep["holdings"]
    await db.close()

    fresh = trend_settings.model_copy(update={"db_path": trend_settings.db_path.with_name("dd.db")})
    db, alloc = await make_alloc(fresh, ex)
    await alloc.run_once(NOW)
    alloc.state.peak = 1000.0  # equity far below a past peak
    await alloc.save()
    rep = await TrendAllocator(fresh, db, ex).run_once(NOW + 3_600_000)
    assert rep["halted"].startswith("max_drawdown") and not rep["holdings"]
    await db.close()
    status = await runner_run(["--reset-halt"], fresh, ex)
    assert status["tendencia"]["halted"] is None


@pytest.mark.asyncio
async def test_rebalance_waits_for_the_daily_candle(trend_settings):
    stale = {s: d.iloc[:-1] for s, d in market().items()}  # yesterday's candle not published yet
    ex = FakeEx(stale)
    await ex.load_markets()
    db, alloc = await make_alloc(trend_settings, ex)
    rep = await alloc.run_once(NOW)
    assert rep["last_close"] is None and not rep["holdings"]
    await db.close()


# ---------------------------------------------------------------- live allocator (exchange orders)


class FakeLiveEx(FakeEx):
    def __init__(self, daily: dict[str, pd.DataFrame], usdt: float) -> None:
        super().__init__(daily)
        self.orders: dict[str, dict] = {}
        self.bal: dict[str, float] = {"USDT": usdt}

    def market(self, sym):
        return {"base": sym.split("/")[0], "quote": "USDT", "limits": {"cost": {"min": 5.0}}}

    def amount_to_precision(self, sym, q):
        return f"{np.floor(q * 1e6) / 1e6:.6f}"

    def price_to_precision(self, sym, p):
        return f"{p:.6f}"

    async def fetch_balance(self):
        return {k: {"free": v, "total": v} for k, v in self.bal.items()}

    async def create_order(self, sym, kind, side, amount, price, params=None):
        params = params or {}
        oid = str(len(self.orders) + 1)
        base = sym.split("/")[0]
        last = float(self.daily[sym]["close"].iloc[-1])
        o = {"id": oid, "symbol": sym, "side": side, "amount": amount, "clientOrderId": params.get("newClientOrderId")}
        if "stopLossPrice" in params:
            o.update(status="open", filled=0.0, average=None, price=price, stop=params["stopLossPrice"])
        else:
            px = price if kind == "limit" else last
            fee = amount * px * 0.001
            o.update(status="closed", filled=amount, average=px, fee={"cost": fee, "currency": "USDT"})
            sign = 1 if side == "buy" else -1
            self.bal["USDT"] -= sign * amount * px + fee
            self.bal[base] = self.bal.get(base, 0.0) + sign * amount
        self.orders[oid] = o
        return dict(o)

    async def fetch_order(self, oid, sym=None, params=None):
        return dict(self.orders[oid])

    async def cancel_order(self, oid, sym=None):
        if self.orders[oid]["status"] == "open":
            self.orders[oid]["status"] = "canceled"

    def fire_stop(self, sym):
        o = next(o for o in self.orders.values() if o["symbol"] == sym and o["status"] == "open")
        o.update(status="closed", filled=o["amount"], average=o["price"],
                 fee={"cost": o["amount"] * o["price"] * 0.001, "currency": "USDT"})
        self.bal[sym.split("/")[0]] -= o["amount"]
        self.bal["USDT"] += o["amount"] * o["price"] * 0.999


@pytest.mark.asyncio
async def test_live_run_places_exchange_stops_and_books_a_fired_stop(trend_settings):
    from mqmas.config import TradingMode

    live = trend_settings.model_copy(update={"trading_mode": TradingMode.LIVE, "trend_live_enabled": True})
    ex = FakeLiveEx(market(), usdt=500.0)
    await ex.load_markets()
    db, alloc = await make_alloc(live, ex)
    assert alloc.mode == "live"
    rep = await alloc.run_once(NOW)
    assert rep["holdings"] and all(h["stop_order"] for h in rep["holdings"].values())
    resting = [o for o in ex.orders.values() if o["status"] == "open"]
    assert len(resting) == len(rep["holdings"])  # one exchange stop per position
    assert ex.bal["USDT"] >= 500.0 - 200.0 - 1e-6  # never spends more than TREND_CAPITAL_USDT
    sym = next(iter(rep["holdings"]))
    ex.fire_stop(sym)
    rep2 = await TrendAllocator(live, db, ex).run_once(NOW + 3_600_000)
    assert sym not in rep2["holdings"] and sym in rep2["blocked"]
    await db.close()


@pytest.mark.asyncio
async def test_turning_the_kill_switch_off_resumes(trend_settings):
    ex = FakeEx(market())
    await ex.load_markets()
    killed = trend_settings.model_copy(update={"trend_kill_switch": True})
    db, alloc = await make_alloc(killed, ex)
    assert (await alloc.run_once(NOW))["halted"] == "kill_switch"
    rep = await TrendAllocator(trend_settings, db, ex).run_once(NOW + 3_600_000)
    assert rep["halted"] is None and rep["holdings"]
    await db.close()


def test_event_study_measures_what_follows_a_big_day():
    from mqmas.research.waves import event_study

    idx = pd.date_range("2024-01-01", periods=60, freq="1D", tz="UTC")
    c = pd.Series(100.0, index=idx)
    c.iloc[30:] = 125.0  # +25 % on day 30, then flat
    c.iloc[35:] = 137.5  # +10 % more five days later: continuation
    closes = pd.DataFrame({f"C{i}/USDT": c for i in range(5)})  # the study needs at least 5 events
    elig = pd.DataFrame(True, index=idx, columns=closes.columns)
    ev = event_study(closes, elig, idx[0], Costs(fee=0.0, slippage=0.0))
    assert ev["+20%"]["eventos"] == 5
    assert ev["+20%"]["5d"]["media"] == pytest.approx(0.10)
    assert ev["+20%"]["1d"]["media"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_wave_hunter_desk_holds_the_strongest_coins_with_its_own_capital_and_state(trend_settings):
    from mqmas.trend.model import momentum_frame

    daily = market()
    w, score = momentum_frame(daily)
    monday = w.index[w.index.dayofweek == 0][-1]
    held = set(w.loc[monday][w.loc[monday] > 0].index)
    assert held and "DOGE/USDT" not in held  # the falling coin never makes the top 5
    assert w.loc[monday].sum() == pytest.approx(1.0)  # BTC above its 100-day average: fully invested
    ex = FakeEx(daily)
    await ex.load_markets()
    cfg = trend_settings.model_copy(update={"ondas_capital_usdt": 100.0})
    out = await runner_run([], cfg, ex)
    assert set(out) == {"tendencia", "ondas"}
    assert out["ondas"]["capital"] == 100.0 and out["tendencia"]["capital"] == 200.0
    assert set(out["ondas"]["holdings"]) <= held
    assert out["ondas"]["equity"] == pytest.approx(100.0, rel=0.02)


def test_wave_hunter_goes_to_cash_when_btc_is_under_its_average():
    from mqmas.trend.model import momentum_frame

    daily = market()
    daily["BTC/USDT"] = frame(900, -0.004, 1)  # BTC in a downtrend
    w, _ = momentum_frame(daily)
    assert w.iloc[-1].sum() == 0.0
