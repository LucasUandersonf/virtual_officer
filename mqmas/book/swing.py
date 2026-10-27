"""Swing book: the arena survivors on 1d, 2d and 3d candles, traded with the library's management.

Setups (blind walk-forward survivors of ``mqmas.research.arena``, hit rate 40–60 %, PF ≥ 1.3 in both halves,
no macro gate — the arena found it did not help these):

* 1d  classic breakout (close above the 20-candle high with 1.5× volume);
* 1d  breakout + EMA20/SMA50 cross (Prates) within 3 candles;
* 2d  classic breakout;
* 3d  classic breakout;
* 3d  descending triangle broken upward (Stanzione).

No fixed target: a winner is never capped, the trailing stop decides when it ends (the arena's "alvo=0R"
variants — about the same PF as a 2R target, with the big moves kept).

Management, the same as the backtest: initial stop = farther of the technical stop and 1.5 ATR (cap 3 ATR),
re-anchored on the fill; breakeven (covering costs) once price reaches +1R; from +2R the stop trails at the
higher of the last 3 candles' low and peak − 2.5 ATR; a bearish exit pattern on the closed candle sells at
the next open; time stop after 20 / 15 / 12 candles. 2d and 3d candles are epoch-anchored (as in research)
and only act on the day they complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mqmas.research import arena, livro
from mqmas.research.engine import Exit
from mqmas.research.strategies import Config, features

TFS = ("1d", "2d", "3d")
RULES = {"1d": None, "2d": "2D", "3d": "3D"}
DAYS = {"1d": 1, "2d": 2, "3d": 3}
SETUPS: tuple[dict[str, Any], ...] = (
    {"nome": "rompimento", "tf": "1d", "tipo": "individual", "membros": ("cl_rompimento",), "alvo_r": 0.0},
    {"nome": "rompimento+medias", "tf": "1d", "tipo": "par", "membros": ("cl_rompimento", "tcc_medias"),
     "alvo_r": 0.0},
    {"nome": "rompimento", "tf": "2d", "tipo": "individual", "membros": ("cl_rompimento",), "alvo_r": 0.0},
    {"nome": "rompimento", "tf": "3d", "tipo": "individual", "membros": ("cl_rompimento",), "alvo_r": 0.0},
    {"nome": "triangulo", "tf": "3d", "tipo": "individual", "membros": ("triangulo_descendente_alta",),
     "alvo_r": 0.0},
)
EVIDENCE = {
    ("1d", "rompimento"): "arena 2023-07→2026-10 fora da amostra: n=274, acerto 42%, PF 1,62",
    ("1d", "rompimento+medias"): "arena 2023-07→2026-10 fora da amostra: n=73, acerto 48%, PF 2,12",
    ("2d", "rompimento"): "arena 2023-07→2026-10 fora da amostra: n=151, acerto 43%, PF 2,36",
    ("3d", "rompimento"): "arena 2023-07→2026-10 fora da amostra: n=112, acerto 50%, PF 2,23",
    ("3d", "triangulo"): "arena 2023-07→2026-10 fora da amostra: n=65, acerto 49%, PF 3,24",
}
BE_AT_R, TRAIL_AT_R, TRAIL_ATR = 1.0, 2.0, 2.5


@dataclass
class SwingHit:
    setup: dict[str, Any]
    price: float
    atr: float
    risk: float  # distance from the signal close to the initial stop
    layers: dict[str, Any]

    @property
    def cfg(self) -> Config:
        s = self.setup
        tgt = s["alvo_r"] * self.risk / self.atr if s["alvo_r"] else None
        return Config("swing", s["tf"], (("trigger", s["nome"]),),
                      Exit(round(self.risk / self.atr, 2), round(tgt, 2) if tgt else None, None,
                           arena.HOLD[s["tf"]]))


def frame(daily: pd.DataFrame, tf: str) -> pd.DataFrame | None:
    """The tf candles, or None when the last one did not complete on the latest daily close."""
    d = daily[["open", "high", "low", "close", "volume"]]
    rule = RULES[tf]
    if rule is None:
        return d
    df = arena.resample(d, rule)
    if df.empty or df.index[-1] + pd.Timedelta(days=DAYS[tf]) != d.index[-1] + pd.Timedelta(days=1):
        return None
    return df


def scan(daily: pd.DataFrame) -> tuple[list[SwingHit], dict[str, dict[str, Any]]]:
    """Entries on the candles that just closed, plus per-tf management data (low of last 3, ATR, exit pattern)."""
    hits: list[SwingHit] = []
    mgmt: dict[str, dict[str, Any]] = {}
    for tf in TFS:
        df = frame(daily, tf)
        if df is None or len(df) < 60:
            continue
        sig, st, exits = arena.signal_frame(df)
        atr = float(features(df)["atr"].iloc[-1])
        mgmt[tf] = {"low3": float(df["low"].iloc[-3:].min()), "atr": atr, "exit": bool(exits[-1])}
        if not np.isfinite(atr) or atr <= 0:
            continue
        recent = sig.astype(float).rolling(livro.CONFIRM, min_periods=1).max().to_numpy()
        cols = list(sig.columns)
        close = float(df["close"].iloc[-1])
        for s in (x for x in SETUPS if x["tf"] == tf):
            if not arena.entries_for(sig, recent, s)[-1]:
                continue
            tech = arena.stop_for(st, recent, s["membros"], cols)[-1]
            d_tech = close - tech if np.isfinite(tech) else 0.0
            risk = min(max(d_tech, 1.5 * atr), 3.0 * atr)
            hits.append(SwingHit(s, close, atr, risk, {"setup": f"{tf} {s['nome']}",
                                                       "alvo": f"{s['alvo_r']:g}R" if s["alvo_r"] else "trailing",
                                                       "risco_pct": round(risk / close * 100, 2),
                                                       "atr_pct": round(atr / close * 100, 2)}))
    return hits, mgmt


def hold_minutes(tf: str) -> int:
    return arena.HOLD[tf] * DAYS[tf] * 1440


def new_stop(entry: float, stop: float, risk: float, peak: float, atr: float, low3: float, round_trip: float) -> float:
    """Breakeven at +1R (covering costs), trailing from +2R; the stop never moves down."""
    out = stop
    if peak >= entry + BE_AT_R * risk:
        out = max(out, entry * (1 + round_trip))
    if peak >= entry + TRAIL_AT_R * risk and atr > 0:
        out = max(out, low3, peak - TRAIL_ATR * atr)
    return out
