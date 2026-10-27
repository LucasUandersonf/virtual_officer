"""Investment committee: every purchase the trend model proposes is reviewed by the specialists.

Flow for each proposed buy (a new position or a larger one):

1. the specialists speak in order (tecnico → fundamentalista → sentimento → risco); each sees the opinions
   given before it, so Vera's pre-mortem reads the whole team;
2. only **active** specialists change the order: any veto keeps the current size, otherwise any reduction
   halves the increase. Shadow specialists are heard and recorded but change nothing;
3. every opinion is stored with the coin's price, and the scoreboard later fills in what the coin did over
   the next 7 and 30 days. A specialist *helps* when the coins it flagged did worse than the coins it let
   through, by enough to rule out luck.

With no active specialist the desk trades exactly like the backtest.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from mqmas.persistence.db import Database
from mqmas.trend.specialists import ABSTAIN, CONFIRM, REDUCE, VETO, Opinion, Proposal, Specialist

log = logging.getLogger(__name__)

HORIZONS = (7, 30)
MIN_FLAGS = 30  # flagged proposals with a 30-day outcome before the scoreboard gives a verdict
T_MIN = 2.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS committee_opinions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    day         TEXT NOT NULL,          -- daily close the proposal was made on
    symbol      TEXT NOT NULL,
    specialist  TEXT NOT NULL,
    verdict     TEXT NOT NULL,          -- confirmar | reduzir | vetar | abster
    active      INTEGER NOT NULL,       -- 1 = it changed the order, 0 = shadow
    weight      REAL NOT NULL,          -- weight proposed by the trend model
    price       REAL NOT NULL,
    reason      TEXT NOT NULL,
    data        TEXT NOT NULL,          -- JSON
    fwd_7d      REAL,
    fwd_30d     REAL,
    UNIQUE (day, symbol, specialist)
);
"""


@dataclass
class Review:
    symbol: str
    proposed: float
    current: float
    final: float
    opinions: list[Opinion] = field(default_factory=list)

    @property
    def summary(self) -> str:
        votes = ", ".join(f"{o.specialist}: {o.verdict}" for o in self.opinions)
        return f"{self.symbol} {self.proposed:.1%} → {self.final:.1%} ({votes})"


def combine(opinions: list[Opinion], active: set[str]) -> float:
    """Multiplier on the proposed increase from the active specialists only: veto 0, reduce 0.5, else 1."""
    acting = [o.verdict for o in opinions if o.specialist in active]
    if VETO in acting:
        return 0.0
    if REDUCE in acting:
        return 0.5
    return 1.0


class Committee:
    def __init__(self, specialists: list[Specialist], active: set[str], db: Database) -> None:
        self.specialists = specialists
        self.active = active
        self._db = db

    async def ensure_schema(self) -> None:
        await self._db.conn.executescript(SCHEMA)
        await self._db.conn.commit()

    async def review(self, day: str, proposals: list[Proposal]) -> dict[str, Review]:
        out: dict[str, Review] = {}
        for p in proposals:
            opinions: list[Opinion] = []
            for sp in self.specialists:
                try:
                    op = await sp.opine(p, list(opinions))
                except Exception as exc:  # a specialist failing must never stop the desk
                    log.exception("specialist %s failed on %s", sp.name, p.symbol)
                    op = Opinion(sp.name, ABSTAIN, f"erro: {type(exc).__name__}")
                opinions.append(op)
            mult = combine(opinions, self.active)
            final = p.current_weight + (p.weight - p.current_weight) * mult
            out[p.symbol] = Review(p.symbol, p.weight, p.current_weight, final, opinions)
            await self._record(day, p, opinions)
        return out

    async def _record(self, day: str, p: Proposal, opinions: list[Opinion]) -> None:
        async with self._db._lock:
            await self._db.conn.executemany(
                "INSERT OR REPLACE INTO committee_opinions(day, symbol, specialist, verdict, active, weight, price, "
                "reason, data) VALUES (?,?,?,?,?,?,?,?,?)",
                [(day, p.symbol, o.specialist, o.verdict, int(o.specialist in self.active), p.weight, p.price,
                  o.reason, json.dumps(o.data, ensure_ascii=False, default=str)) for o in opinions])
            await self._db.conn.commit()

    # ------------------------------------------------------------------ outcomes and scoreboard

    async def fill_outcomes(self, closes: dict[str, pd.Series], today: str) -> int:
        """Forward returns of past proposals whose 7- or 30-day horizon has passed."""
        t = date.fromisoformat(today)
        async with self._db.conn.execute(
            "SELECT DISTINCT day, symbol, price FROM committee_opinions WHERE fwd_30d IS NULL") as cur:
            rows = [dict(r) for r in await cur.fetchall()]
        n = 0
        for r in rows:
            c = closes.get(r["symbol"])
            if c is None or not r["price"]:
                continue
            upd: dict[str, float] = {}
            for h in HORIZONS:
                d = date.fromisoformat(r["day"]) + timedelta(days=h)
                if d > t:
                    continue
                at = c.loc[:pd.Timestamp(d, tz="UTC")]
                if len(at):
                    upd[f"fwd_{h}d"] = float(at.iloc[-1]) / float(r["price"]) - 1
            if upd:
                sets = ", ".join(f"{k}=?" for k in upd)
                await self._db._write(f"UPDATE committee_opinions SET {sets} WHERE day=? AND symbol=?",
                                      (*upd.values(), r["day"], r["symbol"]))
                n += 1
        return n

    async def scoreboard(self) -> list[dict[str, Any]]:
        async with self._db.conn.execute(
            "SELECT specialist, verdict, active, weight, fwd_7d, fwd_30d FROM committee_opinions") as cur:
            rows = [dict(r) for r in await cur.fetchall()]
        board = []
        for name in sorted({r["specialist"] for r in rows}):
            mine = [r for r in rows if r["specialist"] == name]
            flagged = [r for r in mine if r["verdict"] in (REDUCE, VETO) and r["fwd_30d"] is not None]
            passed = [r for r in mine if r["verdict"] == CONFIRM and r["fwd_30d"] is not None]
            f = np.array([r["fwd_30d"] for r in flagged], dtype=float)
            g = np.array([r["fwd_30d"] for r in passed], dtype=float)
            edge = float(g.mean() - f.mean()) if len(f) and len(g) else None
            se = math.sqrt((f.var(ddof=1) / len(f) if len(f) > 1 else 0) + (g.var(ddof=1) / len(g) if len(g) > 1 else 0))
            t = edge / se if edge is not None and se > 0 else None
            # weight × return the flags would have avoided (+ = losses avoided, − = gains given up)
            avoided = float(sum(-(1.0 if r["verdict"] == VETO else 0.5) * r["weight"] * r["fwd_30d"] for r in flagged))
            if len(f) < MIN_FLAGS or t is None:
                verdict = "amostra insuficiente"
            elif t >= T_MIN:
                verdict = "ajuda"
            elif t <= -T_MIN:
                verdict = "atrapalha"
            else:
                verdict = "sem efeito comprovado"
            board.append({
                "especialista": name, "ativo": any(r["active"] for r in mine), "pareceres": len(mine),
                "confirmou": sum(r["verdict"] == CONFIRM for r in mine),
                "reduziu": sum(r["verdict"] == REDUCE for r in mine), "vetou": sum(r["verdict"] == VETO for r in mine),
                "absteve": sum(r["verdict"] == ABSTAIN for r in mine), "com_resultado_30d": len(flagged) + len(passed),
                "retorno_30d_sinalizados": float(f.mean()) if len(f) else None,
                "retorno_30d_liberados": float(g.mean()) if len(g) else None,
                "vantagem": edge, "t": t, "peso_x_retorno_evitado": avoided, "veredito": verdict})
        return board


def render_board(board: list[dict[str, Any]]) -> str:
    if not board:
        return "Placar dos especialistas: ainda sem pareceres."
    lines = ["Placar dos especialistas (resultado 30 dias depois):"]
    for b in board:
        mode = "ATIVO" if b["ativo"] else "sombra"
        edge = f", vantagem {b['vantagem']:+.1%} (t {b['t']:.1f})" if b["vantagem"] is not None and b["t"] else ""
        lines.append(f"• {b['especialista']} [{mode}]: {b['pareceres']} pareceres, {b['reduziu']} reduções, "
                     f"{b['vetou']} vetos{edge} → {b['veredito']}")
    return "\n".join(lines)
