"""Independent re-test with backtesting.py (kernc/backtesting.py, AGPL-3.0) on real Binance candles.

Our engine (``livro.managed_trades``) is replayed by a third-party backtester with its own order matching:
next-open fills, stop-loss orders that fill at the open when the market gaps through them, commission on
both sides. Same signals, same rules — initial stop = farther of the technical stop and 1.5 ATR (cap 3 ATR),
breakeven at +1R covering costs, trailing from +2R, exit patterns, time stop — one position per coin, all-in
(the library trades one asset at a time), so our side is compared on non-overlapping trades too.

Known small difference, stated in the report: the library must send the stop with the order, before the fill
price is known, so the entry candle uses a stop computed from the signal close; from the next candle on it is
re-anchored on the real fill, as in our engine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mqmas.research import livro
from mqmas.research.data import load_history
from mqmas.research.engine import Costs
from mqmas.research.strategies import features

log = logging.getLogger(__name__)

VERSION = 1
COINS = ("BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT", "LINK/USDT",
         "AVAX/USDT", "LTC/USDT", "DOT/USDT", "TRX/USDT")
SINCE = pd.Timestamp("2021-10-01", tz="UTC")
SETUPS: tuple[dict[str, Any], ...] = (
    {"nome": "≥2 estratégias do livro (1d)", "tf": "1d", "tipo": "≥2 estratégias", "membros": tuple(livro.LONGS),
     "kmin": 2, "saida_padrao": True},
    {"nome": "≥2 estratégias do livro (3d)", "tf": "3d", "tipo": "≥2 estratégias", "membros": tuple(livro.LONGS),
     "kmin": 2, "saida_padrao": True},
    {"nome": "Triângulo descendente, compra na LTB (3d)", "tf": "3d", "tipo": "individual",
     "membros": ("triangulo_descendente_alta",), "saida_padrao": True},
    {"nome": "Cunha descendente (3d)", "tf": "3d", "tipo": "individual", "membros": ("cunha_descendente",),
     "saida_padrao": False},
    {"nome": "MACD do TCC (3d)", "tf": "3d", "tipo": "individual", "membros": ("tcc_macd",), "saida_padrao": True},
)


def prepare(df: pd.DataFrame, setup: dict[str, Any]) -> dict[str, np.ndarray]:
    ev, stops, _ = livro.detect(df)
    ent = livro._entries(ev, setup).copy()
    ent[: livro.WARMUP[setup["tf"]]] = False
    recent = ev[list(livro.LONGS)].astype(float).rolling(livro.CONFIRM, min_periods=1).max().to_numpy()
    stop = np.full(len(df), np.nan)
    own = np.zeros(len(df), dtype=bool)
    for m in setup["membros"]:
        has = np.isnan(stop) & (recent[:, livro.LONGS.index(m)] > 0)
        stop[has] = stops[m].to_numpy()[has]
        if m in livro.OWN_EXIT:
            own |= ev[livro.OWN_EXIT[m]].to_numpy()
    exits = (ev[livro.EXITS].any(axis=1).to_numpy() | own) if setup["saida_padrao"] else np.zeros(len(df), dtype=bool)
    return {"entries": ent, "stop": stop, "exits": exits, "atr": features(df)["atr"].to_numpy()}


def ours(df: pd.DataFrame, arr: dict[str, np.ndarray], hold: int, costs: Costs) -> pd.DataFrame:
    o, h, lo, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    t = livro.managed_trades(o, h, lo, c, arr["atr"], arr["entries"], arr["stop"], None, arr["exits"], costs, hold)
    out, busy_until = [], -1
    for e, x, r in zip(t["entry"].astype(int), t["exit"].astype(int), t["ret"], strict=True):
        if e > busy_until:  # one position at a time, like the library
            out.append((df.index[e], df.index[x], float(r)))
            busy_until = x
    return pd.DataFrame(out, columns=["t_in", "t_out", "ret"])


def library(df: pd.DataFrame, arr: dict[str, np.ndarray], hold: int, costs: Costs) -> tuple[pd.DataFrame, dict[str, Any]]:
    from backtesting import Backtest, Strategy

    side = costs.fee + costs.slippage
    rt = costs.round_trip
    ent, stp, exs, atr = arr["entries"], arr["stop"], arr["exits"], arr["atr"]

    class Managed(Strategy):  # type: ignore[misc]
        def init(self) -> None:
            self.risk: dict[int, float] = {}

        def next(self) -> None:
            i = len(self.data) - 1
            if self.position:
                tr = self.trades[-1]
                e = tr.entry_bar
                if e not in self.risk:  # first candle after the fill: re-anchor on the real entry price
                    tech = tr.entry_price - stp[e - 1] if np.isfinite(stp[e - 1]) else 0.0
                    self.risk[e] = min(max(tech, 1.5 * atr[e - 1]), 3.0 * atr[e - 1])
                    tr.sl = tr.entry_price - self.risk[e]
                risk = self.risk[e]
                if i - e + 1 >= hold or exs[i]:
                    self.position.close()
                    return
                peak = float(np.max(self.data.High[e:i + 1]))
                sl = tr.sl
                if peak >= tr.entry_price + risk:
                    sl = max(sl, tr.entry_price * (1 + rt))
                if peak >= tr.entry_price + 2 * risk and np.isfinite(atr[i]):
                    sl = max(sl, float(np.min(self.data.Low[max(i - 2, e):i + 1])), peak - 2.5 * atr[i])
                tr.sl = sl
            elif ent[i] and np.isfinite(atr[i]) and atr[i] > 0:
                c = float(self.data.Close[-1])
                tech = c - stp[i] if np.isfinite(stp[i]) else 0.0
                risk = min(max(tech, 1.5 * atr[i]), 3.0 * atr[i])
                self.buy(size=0.95, sl=c - risk)

    data = df.rename(columns=str.capitalize)[["Open", "High", "Low", "Close", "Volume"]]
    bt = Backtest(data, Managed, cash=1_000_000, commission=side, trade_on_close=False, exclusive_orders=False,
                  finalize_trades=True)
    stats = bt.run()
    trades = stats["_trades"]
    tr = pd.DataFrame({"t_in": trades["EntryTime"], "t_out": trades["ExitTime"], "ret": trades["ReturnPct"]})
    keep = ("Return [%]", "Buy & Hold Return [%]", "Max. Drawdown [%]", "# Trades", "Win Rate [%]", "Profit Factor")
    return tr, {k: (round(float(stats[k]), 2) if pd.notna(stats[k]) else None) for k in keep}


def _summary(r: np.ndarray) -> dict[str, Any]:
    if r.size == 0:
        return {"trades": 0}
    gain, loss = float(r[r > 0].sum()), float(-r[r <= 0].sum())
    return {"trades": int(r.size), "acerto": round(float((r > 0).mean()), 3),
            "pf": round(min(gain / loss, 99.0), 2) if loss > 0 else 99.0,
            "retorno_medio_pct": round(float(r.mean()) * 100, 2),
            "composto_pct": round((float(np.prod(1 + r)) - 1) * 100, 1)}


def run(root: Path, ex: Any = None) -> dict[str, Any]:
    if ex is None:
        import ccxt

        ex = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    costs = Costs()
    daily = {}
    for coin in COINS:
        try:
            d = load_history(ex, coin, "1d", int(SINCE.timestamp() * 1000), root / "candles")
        except Exception:
            log.exception("crosscheck: no data for %s", coin)
            continue
        if len(d) > 400:
            daily[coin] = d[["open", "high", "low", "close", "volume"]]
    end = max(d.index[-1] for d in daily.values())
    out: dict[str, Any] = {"version": VERSION, "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
                           "biblioteca": "backtesting.py (kernc), motor independente", "moedas": list(daily),
                           "periodo": f"{SINCE.date()} → {end.date()}", "setups": []}
    for setup in SETUPS:
        tf = setup["tf"]
        hold = livro.HOLD[tf]
        mine, lib, per_coin = [], [], {}
        for coin, d in daily.items():
            df = d if tf == "1d" else livro.resample_3d(d, end)
            if len(df) < livro.WARMUP[tf] + 30:
                continue
            arr = prepare(df, setup)
            a = ours(df, arr, hold, costs)
            b, stats = library(df, arr, hold, costs)
            mine.append(a["ret"].to_numpy())
            lib.append(b["ret"].to_numpy())
            per_coin[coin] = {"nosso_motor": _summary(a["ret"].to_numpy()), "backtesting_py": stats}
        m = np.concatenate(mine) if mine else np.array([])
        lb = np.concatenate(lib) if lib else np.array([])
        out["setups"].append({"nome": setup["nome"], "tf": tf, "nosso_motor": _summary(m),
                              "backtesting_py": _summary(lb), "por_moeda": per_coin})
        log.info("crosscheck: %s done", setup["nome"])
    return out


def summarize(rep: dict[str, Any]) -> list[str]:
    lines = [f"CROSSCHECK v{rep['version']} | {rep['biblioteca']} | {len(rep['moedas'])} moedas | {rep['periodo']}"]
    for s in rep["setups"]:
        a, b = s["nosso_motor"], s["backtesting_py"]
        lines.append(f"CROSSCHECK {s['nome']} | nosso: n={a.get('trades')} acerto {a.get('acerto')} PF {a.get('pf')} "
                     f"médio {a.get('retorno_medio_pct')}% | backtesting.py: n={b.get('trades')} acerto {b.get('acerto')} "
                     f"PF {b.get('pf')} médio {b.get('retorno_medio_pct')}%")
        for coin, v in s["por_moeda"].items():
            st = v["backtesting_py"]
            lines.append(f"    {coin}: lib retorno {st['Return [%]']}% vs buy&hold {st['Buy & Hold Return [%]']}% | "
                         f"DD {st['Max. Drawdown [%]']}% | n={st['# Trades']} acerto {st['Win Rate [%]']}% "
                         f"PF {st['Profit Factor']}")
    return lines
