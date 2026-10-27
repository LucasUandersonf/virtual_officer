"""Intraday seasonality: which hours, weekdays, sessions and short movements repeat in crypto.

Everything is measured on the cached 1h klines (with taker-buy volume) of the liquid universe and
checked for *stability*: a pattern only counts when it has the same sign, and |t| ≥ 2, both in the
first period (2022-01 → 2023-06) and in the second (2023-07 → today). Then the effect size is
compared with the cost of trading it (0.30 % per round trip), because a repeatable pattern that
moves 0.02 % is real but not tradable.

All times are UTC; Brasília (BRT) is UTC−3.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mqmas.research.lab import FIRST_TEST, HISTORY_START

log = logging.getLogger(__name__)

VERSION = 1
ROUND_TRIP = 0.003  # 0.1 % fee + 0.05 % slippage, each side
SESSIONS = {"Ásia (21h–05h BRT)": range(0, 8), "Europa (05h–11h BRT)": range(8, 14),
            "EUA (11h–18h BRT)": range(14, 21), "Fim do dia EUA (18h–21h BRT)": range(21, 24)}
WEEKDAYS = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]


def load_cached(root: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for path in sorted((root / "swing").glob("*_1h_tb.npz")):
        rows = np.load(path)["rows"]
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "taker_buy"])
        df.index = pd.to_datetime(df.pop("ts").astype("int64"), unit="ms", utc=True)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df[df.index >= HISTORY_START]
        if len(df) > 24 * 365:
            out[path.name.replace("_1h_tb.npz", "").replace("_", "/")] = df
    return out


def tstat(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 30:
        return 0.0
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / math.sqrt(x.size))) if sd > 0 else 0.0


def split_stats(s: pd.Series) -> dict[str, float]:
    """Mean (bps) and t in both periods; stable = same sign and |t| ≥ 2 in each."""
    a, b = s[s.index < FIRST_TEST].to_numpy(), s[s.index >= FIRST_TEST].to_numpy()
    ta, tb = tstat(a), tstat(b)
    ma = float(np.nanmean(a)) * 1e4 if a.size else 0.0
    mb = float(np.nanmean(b)) * 1e4 if b.size else 0.0
    stable = bool(np.sign(ma) == np.sign(mb) and abs(ta) >= 2 and abs(tb) >= 2)
    return {"mean_bps_p1": round(ma, 2), "t_p1": round(ta, 2), "mean_bps_p2": round(mb, 2), "t_p2": round(tb, 2),
            "n": int(s.notna().sum()), "stable": stable,
            "pays_costs": bool(stable and min(abs(ma), abs(mb)) / 1e4 > ROUND_TRIP)}


def basket(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight market: mean hourly log return, total relative volume, mean taker-buy ratio."""
    rets = pd.DataFrame({s: np.log(d["close"] / d["open"]) for s, d in frames.items()})
    rel_vol = pd.DataFrame({s: d["volume"] / d["volume"].rolling(24 * 30, min_periods=24 * 7).mean()
                            for s, d in frames.items()})
    tbr = pd.DataFrame({s: d["taker_buy"] / d["volume"].replace(0, np.nan) for s, d in frames.items()})
    rng = pd.DataFrame({s: np.log(d["high"] / d["low"]) for s, d in frames.items()})
    return pd.DataFrame({"ret": rets.mean(axis=1), "rel_vol": rel_vol.mean(axis=1), "tbr": tbr.mean(axis=1),
                         "range": rng.mean(axis=1)}).dropna(subset=["ret"])


def by_hour(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for h in range(24):
        sub = df[df.index.hour == h]
        st = split_stats(sub["ret"])
        out.append({"hour_utc": h, "hour_brt": (h - 3) % 24, **st,
                    "range_bps": round(float(sub["range"].mean()) * 1e4, 1),
                    "rel_volume": round(float(sub["rel_vol"].mean()), 2),
                    "taker_buy": round(float(sub["tbr"].mean()), 4),
                    "p_up": round(float((sub["ret"] > 0).mean()), 3)})
    return out


def by_weekday(df: pd.DataFrame) -> list[dict[str, Any]]:
    daily = df["ret"].resample("1D").sum()
    rng = df["range"].resample("1D").mean()
    out = []
    for d in range(7):
        sub = daily[daily.index.dayofweek == d]
        out.append({"day": WEEKDAYS[d], **split_stats(sub), "p_up": round(float((sub > 0).mean()), 3),
                    "range_bps": round(float(rng[rng.index.dayofweek == d].mean()) * 1e4, 1)})
    return out


def by_session(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for name, hours in SESSIONS.items():
        sub = df[df.index.hour.isin(list(hours))]
        per_day = sub["ret"].resample("1D").sum()
        out.append({"session": name, **split_stats(per_day), "p_up": round(float((per_day > 0).mean()), 3),
                    "volume_share": round(float(sub["rel_vol"].sum() / df["rel_vol"].sum()), 3),
                    "range_bps_per_hour": round(float(sub["range"].mean()) * 1e4, 1)})
    return out


def heatmap(df: pd.DataFrame) -> dict[str, Any]:
    """Weekday × hour mean return (bps) on the whole sample, plus the stable cells."""
    cells, stable = [], []
    for d in range(7):
        row = []
        for h in range(24):
            sub = df[(df.index.dayofweek == d) & (df.index.hour == h)]["ret"]
            row.append(round(float(sub.mean()) * 1e4, 1))
            st = split_stats(sub)
            if st["stable"]:
                stable.append({"day": WEEKDAYS[d], "hour_utc": h, "hour_brt": (h - 3) % 24, **st})
        cells.append(row)
    return {"bps": cells, "stable_cells": stable, "tests": 168,
            "expected_false_positives": "com 168 células e |t| ≥ 2 nas duas metades, ~0,4 célula estável por puro acaso"}


def movements(frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Recurring short movements, pooled over coins, with the next 1h/4h/24h outcome."""
    rows: list[pd.DataFrame] = []
    for sym, d in frames.items():
        c = d["close"]
        r1 = np.log(c / d["open"])
        atr = (d["high"] - d["low"]).rolling(24 * 7).mean() / c
        vol_z = (d["volume"] - d["volume"].rolling(24 * 7).mean()) / d["volume"].rolling(24 * 7).std()
        tbr = d["taker_buy"] / d["volume"].replace(0, np.nan)
        fwd = {k: np.log(c.shift(-k) / c) for k in (1, 4, 24)}
        day_open = d["open"].groupby(d.index.normalize()).transform("first")
        first4 = np.log(c / day_open).where(d.index.hour == 3)  # close of 03:00 bar = first 4h of the UTC day
        rest = np.log(d["close"].groupby(d.index.normalize()).transform("last") / c)
        rows.append(pd.DataFrame({
            "big_drop": r1 < -2 * atr, "big_pump": r1 > 2 * atr, "climax_sell": (r1 < -2 * atr) & (vol_z > 2) & (tbr < 0.45),
            "climax_buy": (r1 > 2 * atr) & (vol_z > 2) & (tbr > 0.55),
            "fund_hour": d.index.hour.isin([0, 8, 16]),
            "first4_up": first4 > 0, "first4_down": first4 < 0,
            "f1": fwd[1], "f4": fwd[4], "f24": fwd[24], "rest": rest, "hour": d.index.hour, "coin": sym},
            index=d.index))
    allr = pd.concat(rows)
    out = []
    events = {
        "Queda forte em 1h (> 2× a média)": ("big_drop", None),
        "Alta forte em 1h (> 2× a média)": ("big_pump", None),
        "Clímax de venda (queda forte + volume 2σ + vendedores agressivos)": ("climax_sell", None),
        "Clímax de compra (alta forte + volume 2σ + compradores agressivos)": ("climax_buy", None),
        "Horários de funding (00h/08h/16h UTC)": ("fund_hour", None),
        "Primeiras 4h do dia (UTC) em alta → resto do dia": ("first4_up", "rest"),
        "Primeiras 4h do dia (UTC) em queda → resto do dia": ("first4_down", "rest"),
    }
    for label, (flag, target) in events.items():
        ev = allr[allr[flag].fillna(False).astype(bool)]
        item: dict[str, Any] = {"pattern": label, "occurrences": int(len(ev)),
                                "per_month": round(len(ev) / max((allr.index.max() - allr.index.min()).days / 30.4, 1), 1)}
        for col in ([target] if target else ["f1", "f4", "f24"]):
            s = ev[col].dropna()
            st = split_stats(s)
            item[col] = {**st, "p_up": round(float((s > 0).mean()), 3)}
        out.append(item)
    return out


def run(root: Path) -> dict[str, Any]:
    frames = load_cached(root)
    if "BTC/USDT" not in frames:
        raise RuntimeError("no cached 1h klines: run the swing lab first")
    mkt = basket(frames)
    btc = frames["BTC/USDT"]
    btc_df = pd.DataFrame({"ret": np.log(btc["close"] / btc["open"]), "range": np.log(btc["high"] / btc["low"]),
                           "rel_vol": btc["volume"] / btc["volume"].rolling(24 * 30, min_periods=24 * 7).mean(),
                           "tbr": btc["taker_buy"] / btc["volume"].replace(0, np.nan)})
    return {"version": VERSION, "generated_at": pd.Timestamp.now(tz="UTC").isoformat(), "coins": list(frames),
            "periods": {"p1": [str(HISTORY_START.date()), str(FIRST_TEST.date())],
                        "p2": [str(FIRST_TEST.date()), str(mkt.index.max().date())]},
            "round_trip_cost_bps": ROUND_TRIP * 1e4,
            "market": {"hours": by_hour(mkt), "weekdays": by_weekday(mkt), "sessions": by_session(mkt),
                       "heatmap": heatmap(mkt)},
            "btc": {"hours": by_hour(btc_df), "weekdays": by_weekday(btc_df), "sessions": by_session(btc_df)},
            "movements": movements(frames)}


def summarize(rep: dict[str, Any]) -> list[str]:
    m = rep["market"]
    lines = [f"SEASONALITY v{rep['version']} | {len(rep['coins'])} moedas | p1 {rep['periods']['p1']} p2 {rep['periods']['p2']}"
             f" | custo ida+volta {rep['round_trip_cost_bps']:.0f} bps"]
    for scope in ("market", "btc"):
        for h in rep[scope]["hours"]:
            lines.append(f"{scope} hora {h['hour_utc']:02d}UTC/{h['hour_brt']:02d}BRT: p1 {h['mean_bps_p1']:+.1f}bps "
                         f"(t {h['t_p1']:+.1f}) p2 {h['mean_bps_p2']:+.1f}bps (t {h['t_p2']:+.1f}) estável={h['stable']} "
                         f"sobe {h['p_up']:.0%} amplitude {h['range_bps']}bps volume {h['rel_volume']}x tbr {h['taker_buy']}")
        for d in rep[scope]["weekdays"]:
            lines.append(f"{scope} dia {d['day']}: p1 {d['mean_bps_p1']:+.1f}bps (t {d['t_p1']:+.1f}) p2 "
                         f"{d['mean_bps_p2']:+.1f}bps (t {d['t_p2']:+.1f}) estável={d['stable']} sobe {d['p_up']:.0%} "
                         f"amplitude {d['range_bps']}bps")
        for s in rep[scope]["sessions"]:
            lines.append(f"{scope} sessão {s['session']}: p1 {s['mean_bps_p1']:+.1f} (t {s['t_p1']:+.1f}) p2 "
                         f"{s['mean_bps_p2']:+.1f} (t {s['t_p2']:+.1f}) estável={s['stable']} sobe {s['p_up']:.0%} "
                         f"volume {s['volume_share']:.0%} amplitude/h {s['range_bps_per_hour']}bps")
    lines.append(f"heatmap estáveis: {m['heatmap']['stable_cells']}")
    for mv in rep["movements"]:
        parts = [f"{k}: p1 {v['mean_bps_p1']:+.1f} p2 {v['mean_bps_p2']:+.1f}bps estável={v['stable']} "
                 f"paga custo={v['pays_costs']} sobe {v['p_up']:.0%}"
                 for k, v in mv.items() if isinstance(v, dict)]
        lines.append(f"padrão {mv['pattern']} | {mv['occurrences']}x ({mv['per_month']}/mês) | " + " | ".join(parts))
    return lines
