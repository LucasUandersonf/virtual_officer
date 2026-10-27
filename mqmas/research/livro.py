"""The strategy library from the two books, one isolated rule each, plus the confluence study.

Sources: Prates (2021, UFU) — four indicator models; Stanzione — "10 padrões de gráficos". Spot cannot
sell short, so every bearish pattern becomes an EXIT rule (and the matching sell signal of each indicator
model closes that model's long), never an entry. Signals use closed candles; orders fill at the next open.

Trade management, identical for every strategy (``managed_trades``):

* initial stop — the farther of the technical stop (pattern low) and 1.5 ATR(14), never farther than
  3 ATR: never glued to the price, never absurdly wide;
* breakeven — once the high reaches entry + 1R, the stop moves to entry + round-trip costs;
* trailing — from +2R the stop follows max(lowest low of the last 3 candles, highest high − 2.5 ATR),
  only upwards;
* target — the pattern's measured move (height added to the breakout) when the strategy has one, or
  none (the trailing stop rides the trend); both variants are tested;
* exits — the time stop, and (one variant) any bearish pattern of the library plus the model's own sell
  signal; the other variant leaves the exit entirely to stop / breakeven / trailing.

The confluence study tests, on the same data: every strategy alone, every pair (B confirming A within
3 candles), "at least k distinct strategies in 3 candles", each with and without the macro gate.
The setup is then chosen by walk-forward — at each 6-month window the best candidate on all earlier
data is traded blind in the next window — so the reported result is out of sample *including* the act
of choosing among hundreds of combinations.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mqmas.research.engine import Costs, RiskRules, simulate_portfolio
from mqmas.research.lab import FIRST_TEST, Lab, deflated_sharpe, monte_carlo
from mqmas.research.pattern_portfolio import equity_monthly
from mqmas.research.strategies import align_daily, btc_regime, classic_macro, ema, features
from mqmas.ta.structure import pivots, structure_events

log = logging.getLogger(__name__)

VERSION = 3
CONFIRM = 3  # candles within which a confluence partner must also have fired
TFS = ("1d", "3d")  # 4h lost after costs in v1/v2; 3-day candles = slower swing
HOLD = {"1d": 40, "3d": 20}  # time stop: 40 days / 60 days
WARMUP = {"1d": 250, "3d": 60}


def resample_3d(daily: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    """3-day candles from daily ones (only complete bars; each bar is stamped with its first day)."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    naive = daily[list(agg)].tz_localize(None)  # fixed epoch-anchored buckets need a tz-naive index
    out = naive.resample("3D", origin="epoch", label="left", closed="left").agg(agg).dropna().tz_localize("UTC")
    return out[out.index + pd.Timedelta(days=3) <= end + pd.Timedelta(days=1)]
WINDOW = 60  # look-back of the rounding patterns

# ---------------------------------------------------------------------- the catalogue (one entry per rule)

CATALOG: list[dict[str, Any]] = [
    {"id": "tcc_medias", "fonte": "Prates (2021)", "nome": "Cruzamento de Médias", "lado": "compra",
     "entrada": "EMA20 cruza para cima a SMA50 no fechamento", "saida_propria": "EMA20 cruza para baixo a SMA50",
     "stop_tecnico": "mínima das últimas 5 velas", "alvo": "sem alvo (trailing)"},
    {"id": "tcc_macd", "fonte": "Prates (2021)", "nome": "Cruzamento do MACD", "lado": "compra",
     "entrada": "MACD(12,26) cruza para cima a linha de sinal(9)", "saida_propria": "MACD cruza para baixo o sinal",
     "stop_tecnico": "mínima das últimas 5 velas", "alvo": "sem alvo (trailing)"},
    {"id": "tcc_estocastico", "fonte": "Prates (2021)", "nome": "Rompimento do Estocástico", "lado": "compra",
     "entrada": "%K lento (5,3,3) cruza 20 para cima", "saida_propria": "%K cruza 80 para baixo",
     "stop_tecnico": "mínima das últimas 5 velas", "alvo": "sem alvo (trailing)"},
    {"id": "tcc_fffd", "fonte": "Prates (2021)", "nome": "Fechou Fora Fechou Dentro (Bollinger 20,2)",
     "lado": "compra", "entrada": "fechamento anterior abaixo da banda inferior e atual de volta dentro",
     "saida_propria": "fechamento anterior acima da banda superior e atual de volta dentro",
     "stop_tecnico": "mínima das últimas 5 velas", "alvo": "média de 20 / sem alvo"},
    {"id": "oco_invertido", "fonte": "Stanzione #2", "nome": "Cabeça e ombros invertido", "lado": "compra",
     "entrada": "3 fundos confirmados (pivô 3), cabeça < ombros − 0,6 ATR, ombros iguais (±0,9 ATR); fechamento "
                "rompe a linha de pescoço (maior topo entre os ombros)",
     "stop_tecnico": "mínima do ombro direito (últimas 10 velas)", "alvo": "pescoço + (pescoço − cabeça)"},
    {"id": "fundo_duplo", "fonte": "Stanzione #3", "nome": "Fundo duplo", "lado": "compra",
     "entrada": "2 fundos iguais (±0,6 ATR, ≥ 5 velas entre eles); fechamento rompe o topo entre eles",
     "stop_tecnico": "mínima das últimas 10 velas (meio do movimento de alta)", "alvo": "pescoço + altura do W"},
    {"id": "xicara_alca", "fonte": "Stanzione #5", "nome": "Xícara e alça", "lado": "compra",
     "entrada": "bordas iguais (±3 %), ≥ 15 velas, profundidade 12–50 %, alça ≤ 1/3 da xícara; rompe a borda",
     "stop_tecnico": "mínima da alça", "alvo": "borda + profundidade da xícara"},
    {"id": "fundo_arredondado", "fonte": "Stanzione #7", "nome": "Fundo de arredondamento", "lado": "compra",
     "entrada": "em 60 velas: fundo no terço central, ≥ 10 % abaixo da borda esquerda, metade direita com "
                "mínimas subindo; fechamento rompe a borda esquerda",
     "stop_tecnico": "mínima das últimas 10 velas", "alvo": "borda + profundidade"},
    {"id": "triangulo_ascendente", "fonte": "Stanzione #8", "nome": "Triângulo ascendente (compra)",
     "lado": "compra", "entrada": "2 topos iguais (±0,6 ATR) e fundos subindo; fechamento rompe a resistência",
     "stop_tecnico": "último fundo", "alvo": "resistência + altura do triângulo"},
    {"id": "triangulo_descendente_alta", "fonte": "Stanzione #9", "nome": "Triângulo descendente (compra acima "
     "da LTB)", "lado": "compra", "entrada": "2 fundos iguais e topos caindo; fechamento rompe a linha dos topos",
     "stop_tecnico": "logo abaixo da linha rompida / último fundo", "alvo": "rompimento + altura do triângulo"},
    {"id": "cunha_descendente", "fonte": "Stanzione #10", "nome": "Cunha descendente", "lado": "compra",
     "entrada": "topos e fundos caindo, topos caindo mais rápido (convergente); fechamento rompe a linha dos topos",
     "stop_tecnico": "último fundo da cunha", "alvo": "rompimento + altura do início da cunha"},
    {"id": "oco", "fonte": "Stanzione #1", "nome": "Cabeça e ombros", "lado": "saída",
     "entrada": "spot sem venda a descoberto: rompimento da linha de pescoço para baixo FECHA compras"},
    {"id": "topo_duplo", "fonte": "Stanzione #4", "nome": "Topo duplo", "lado": "saída",
     "entrada": "rompimento do fundo entre os topos FECHA compras"},
    {"id": "topo_arredondado", "fonte": "Stanzione #6", "nome": "Topo de arredondamento", "lado": "saída",
     "entrada": "espelho do fundo arredondado; rompimento da borda esquerda para baixo FECHA compras"},
    {"id": "triangulo_descendente", "fonte": "Stanzione #9", "nome": "Triângulo descendente (venda)",
     "lado": "saída", "entrada": "rompimento do suporte horizontal FECHA compras"},
    {"id": "cunha_ascendente", "fonte": "Stanzione #10", "nome": "Cunha ascendente", "lado": "saída",
     "entrada": "topos e fundos subindo, fundos subindo mais rápido; rompimento da linha dos fundos FECHA compras"},
]
LONGS = [s["id"] for s in CATALOG if s["lado"] == "compra"]
EXITS = [s["id"] for s in CATALOG if s["lado"] == "saída"]
OWN_EXIT = {"tcc_medias": "tcc_medias_venda", "tcc_macd": "tcc_macd_venda", "tcc_estocastico": "tcc_estocastico_venda",
            "tcc_fffd": "tcc_fffd_venda"}


# ---------------------------------------------------------------------- detection

def _line(p1: Any, p2: Any, i: int) -> float:
    return float(p1.price + (p2.price - p1.price) * (i - p1.idx) / (p2.idx - p1.idx)) if p2.idx != p1.idx else p2.price


def detect(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Boolean column per rule (entries + exits), plus each entry rule's technical stop and measured target
    (NaN where the rule has no structure)."""
    o, h, lo, c = (df[k] for k in ("open", "high", "low", "close"))
    n = len(df)
    f = features(df)
    a = f["atr"].to_numpy()
    ev: dict[str, Any] = {}

    def up(x: pd.Series, y: pd.Series | float) -> pd.Series:
        yp = y.shift(1) if isinstance(y, pd.Series) else y
        return (x > y) & (x.shift(1) <= yp)

    def dn(x: pd.Series, y: pd.Series | float) -> pd.Series:
        yp = y.shift(1) if isinstance(y, pd.Series) else y
        return (x < y) & (x.shift(1) >= yp)

    fast, slow = ema(c, 20), c.rolling(50).mean()
    macd = ema(c, 12) - ema(c, 26)
    sig = macd.ewm(span=9, adjust=False).mean()
    k = (100 * (c - lo.rolling(5).min()) / (h.rolling(5).max() - lo.rolling(5).min()).replace(0, np.nan)).rolling(3).mean()
    mid, sd = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
    ev.update({"tcc_medias": up(fast, slow), "tcc_medias_venda": dn(fast, slow),
               "tcc_macd": up(macd, sig), "tcc_macd_venda": dn(macd, sig),
               "tcc_estocastico": up(k, 20.0), "tcc_estocastico_venda": dn(k, 80.0),
               "tcc_fffd": (c.shift(1) < (mid - 2 * sd).shift(1)) & (c > mid - 2 * sd),
               "tcc_fffd_venda": (c.shift(1) > (mid + 2 * sd).shift(1)) & (c < mid + 2 * sd)})
    struct, _ = structure_events(df)
    for col in ("oco_invertido", "fundo_duplo", "xicara_alca", "triangulo_ascendente", "oco", "topo_duplo",
                "triangulo_descendente"):
        ev[col] = struct[col]

    hv, lv, cv = h.to_numpy(), lo.to_numpy(), c.to_numpy()
    extra = {x: np.zeros(n, dtype=bool) for x in ("fundo_arredondado", "topo_arredondado", "triangulo_descendente_alta",
                                                   "cunha_descendente", "cunha_ascendente")}
    piv = pivots(df, 3)
    highs: list[Any] = []
    lows: list[Any] = []
    j = 0
    for i in range(1, n):
        while j < len(piv) and piv[j].conf <= i:
            (highs if piv[j].kind == "H" else lows).append(piv[j])
            j += 1
        if np.isnan(a[i]):
            continue
        tol = 0.6 * a[i]
        if len(highs) >= 2 and len(lows) >= 2 and i - highs[-1].conf <= 30:
            h1, h2, l1, l2 = highs[-2], highs[-1], lows[-2], lows[-1]
            crossed_top = cv[i - 1] <= _line(h1, h2, i - 1) and cv[i] > _line(h1, h2, i)
            falling_h, falling_l = h2.price < h1.price - tol, l2.price < l1.price - tol
            if _flat(l1.price, l2.price, tol) and falling_h and crossed_top:
                extra["triangulo_descendente_alta"][i] = True
            if falling_h and falling_l and (h1.price - h2.price) > (l1.price - l2.price) and crossed_top:
                extra["cunha_descendente"][i] = True
            bot_now, bot_prev = _line(l1, l2, i), _line(l1, l2, i - 1)
            rising_h, rising_l = h2.price > h1.price + tol, l2.price > l1.price + tol
            if rising_h and rising_l and (l2.price - l1.price) > (h2.price - h1.price) \
                    and cv[i - 1] >= bot_prev and cv[i] < bot_now:
                extra["cunha_ascendente"][i] = True
        if i >= WINDOW:
            seg_l, seg_h = lv[i - WINDOW:i], hv[i - WINDOW:i]
            rim = seg_h[:10].max()
            b = int(seg_l.argmin())
            if 15 <= b <= 45 and seg_l[b] <= rim * 0.9 and cv[i - 1] <= rim < cv[i]:
                right = seg_l[b:]
                if len(right) >= 6 and right[len(right) // 2:].min() > seg_l[b]:
                    extra["fundo_arredondado"][i] = True
            floor = seg_l[:10].min()
            t = int(seg_h.argmax())
            if 15 <= t <= 45 and seg_h[t] >= floor * 1.1 and cv[i - 1] >= floor > cv[i]:
                right_h = seg_h[t:]
                if len(right_h) >= 6 and right_h[len(right_h) // 2:].max() < seg_h[t]:
                    extra["topo_arredondado"][i] = True
    ev.update(extra)
    events = pd.DataFrame({k2: pd.Series(v, index=df.index) for k2, v in ev.items()}, index=df.index).fillna(False)
    events = events.astype(bool)
    # technical stop and measured move: indicator models use the last 5 candles; patterns their last swing
    low5, low10, low40 = lo.rolling(5).min(), lo.rolling(10).min(), lo.rolling(WINDOW).min()
    stops = pd.DataFrame(index=df.index)
    targets = pd.DataFrame(index=df.index)
    for s in LONGS:
        stops[s] = low5 if s.startswith("tcc_") else low10
        targets[s] = np.nan if s.startswith("tcc_") else c + (c - low40)
    return events, stops, targets


def _flat(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


# ---------------------------------------------------------------------- trade management

def managed_trades(o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray, atr: np.ndarray, entries: np.ndarray,
                   stop_tech: np.ndarray, target: np.ndarray | None, exit_sig: np.ndarray, costs: Costs,
                   hold: int) -> dict[str, np.ndarray]:
    n = o.size
    rt = costs.round_trip
    side = costs.fee + costs.slippage
    rows: list[tuple[int, int, float, float, int]] = []
    for i in (int(x) for x in np.flatnonzero(entries)):
        if i + 2 >= n or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        e = i + 1
        px = o[e]
        d_tech = px - stop_tech[i] if np.isfinite(stop_tech[i]) else 0.0
        risk = min(max(d_tech, 1.5 * atr[i]), 3.0 * atr[i])
        stop = px - risk
        tgt = target[i] if target is not None and np.isfinite(target[i]) and target[i] > px else np.nan
        peak = px
        exit_px, reason, k = np.nan, 0, e
        for k in range(e, min(e + hold, n)):
            if k > e and o[k] <= stop:  # gap through the stop
                exit_px, reason = o[k], 1
                break
            if lo[k] <= stop:
                exit_px, reason = stop, 1
                break
            if np.isfinite(tgt) and h[k] >= tgt:
                exit_px, reason = (max(o[k], tgt) if k > e else tgt), 2
                break
            if exit_sig[k] and k + 1 < n:
                exit_px, reason, k = o[k + 1], 3, k + 1
                break
            peak = max(peak, h[k])
            if peak >= px + risk:
                stop = max(stop, px * (1 + rt))  # breakeven covering costs
            if peak >= px + 2 * risk and np.isfinite(atr[k]):
                trail = max(lo[max(k - 2, e):k + 1].min(), peak - 2.5 * atr[k])
                stop = max(stop, trail)
        else:
            exit_px, reason = c[k], 4
        if not np.isfinite(exit_px):
            exit_px, reason = c[k], 4
        ret = (exit_px * (1 - side)) / (px * (1 + side)) - 1
        rows.append((e, k, float(ret), float(risk / px), reason))
    if not rows:
        return {x: np.array([]) for x in ("entry", "exit", "ret", "stop_pct", "reason")}
    arr = np.array(rows, dtype=float)
    return {"entry": arr[:, 0], "exit": arr[:, 1], "ret": arr[:, 2], "stop_pct": arr[:, 3], "reason": arr[:, 4]}


# ---------------------------------------------------------------------- candidates and study

def candidates() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tf in TFS:
        for macro in (False, True):
            for s in LONGS:
                for tgt in (True, False) if not s.startswith("tcc_") else (False,):
                    out.append({"tipo": "individual", "tf": tf, "membros": (s,), "macro": macro, "alvo": tgt})
            for a, b in itertools.combinations(LONGS, 2):
                out.append({"tipo": "par", "tf": tf, "membros": (a, b), "macro": macro, "alvo": False})
            for kmin in (2, 3):
                out.append({"tipo": f"≥{kmin} estratégias", "tf": tf, "membros": tuple(LONGS), "kmin": kmin,
                            "macro": macro, "alvo": False})
    out = [{**c, "saida_padrao": sp} for c in out for sp in (True, False)]
    for cand in out:
        who = "todas" if cand["tipo"].startswith("≥") else "+".join(cand["membros"])
        cand["key"] = (f"{cand['tf']}|{cand['tipo']}|{who}|macro={cand['macro']}|alvo={cand['alvo']}"
                       f"|saida_padrao={cand['saida_padrao']}")
    return out


def _entries(ev: pd.DataFrame, cand: dict[str, Any]) -> np.ndarray:
    recent = ev[list(LONGS)].astype(float).rolling(CONFIRM, min_periods=1).max()
    if cand["tipo"] == "individual":
        s: pd.Series = ev[cand["membros"][0]]
    elif cand["tipo"] == "par":
        a, b = cand["membros"]
        s = (ev[a] & (recent[b] > 0)) | (ev[b] & (recent[a] > 0))
    else:
        count = recent.sum(axis=1)
        s = ev[list(LONGS)].any(axis=1) & (count >= cand["kmin"])
    out: np.ndarray = s.fillna(False).to_numpy(dtype=bool)
    return out


def build_tables(lab: Lab, costs: Costs) -> dict[str, pd.DataFrame]:
    cands = candidates()
    rows: dict[str, list[pd.DataFrame]] = {c["key"]: [] for c in cands}
    regime = btc_regime(lab.btc_daily)
    for tf in TFS:
        for coin, daily in lab.data["1d"].items():
            raw = daily if tf == "1d" else resample_3d(daily, lab.end)
            df = raw[raw.index >= pd.Timestamp("2021-10-01", tz="UTC")][["open", "high", "low", "close", "volume"]]
            if len(df) < WARMUP[tf] + 50:
                continue
            ev, stops, targets = detect(df)
            f = features(df)
            months = df.index.tz_localize(None).to_period("M")
            elig = np.array([coin in lab.eligible.get(m, ()) for m in months], dtype=bool)
            if tf == "1d":
                gate = classic_macro(f, align_daily(regime, df.index))
            else:  # the daily gate as it stood on the last day of each 3-day candle
                d = daily[["open", "high", "low", "close", "volume"]]
                g = pd.Series(classic_macro(features(d), align_daily(regime, d.index)), index=d.index)
                gate = g.reindex(df.index + pd.Timedelta(days=2), method="ffill").fillna(False).to_numpy(dtype=bool)
            o, h, lo, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
            a = f["atr"].to_numpy()
            exits_any = ev[EXITS].any(axis=1).to_numpy()
            recent = ev[list(LONGS)].astype(float).rolling(CONFIRM, min_periods=1).max().to_numpy()
            for cand in (x for x in cands if x["tf"] == tf):
                ent = _entries(ev, cand) & elig
                ent[:WARMUP[tf]] = False
                if cand["macro"]:
                    ent &= gate
                if not ent.any():
                    continue
                members = cand["membros"]
                # stop/target from the member that fired most recently (first listed on ties)
                stop = np.full(len(df), np.nan)
                tgt = np.full(len(df), np.nan)
                for m in members:
                    idx = LONGS.index(m)
                    has = np.isnan(stop) & (recent[:, idx] > 0)
                    stop[has] = stops[m].to_numpy()[has]
                    tgt[has] = targets[m].to_numpy()[has]
                own = np.zeros(len(df), dtype=bool)
                for m in members:
                    if m in OWN_EXIT:
                        own |= ev[OWN_EXIT[m]].to_numpy()
                ex_sig = exits_any | own if cand["saida_padrao"] else np.zeros(len(df), dtype=bool)
                t = managed_trades(o, h, lo, c, a, ent, stop, tgt if cand["alvo"] else None, ex_sig,
                                   Costs(costs.fee, costs.slippage), HOLD[tf])
                if t["entry"].size:
                    rows[cand["key"]].append(pd.DataFrame({
                        "coin": coin, "t_in": df.index[t["entry"].astype(int)],
                        "t_out": df.index[t["exit"].astype(int)], "ret": t["ret"], "stop_pct": t["stop_pct"],
                        "reason": t["reason"], "regime": lab._regime_at(coin, df.index[t["entry"].astype(int)])}))
            log.info("livro: %s %s done", tf, coin)
    empty = pd.DataFrame(columns=["coin", "t_in", "t_out", "ret", "stop_pct", "reason", "regime"])
    return {k: (pd.concat(v, ignore_index=True) if v else empty) for k, v in rows.items()}


def _stats(tr: pd.DataFrame) -> dict[str, Any]:
    r = tr["ret"].to_numpy(dtype=float) if len(tr) else np.array([])
    if r.size == 0:
        return {"trades": 0}
    wins, losses = r[r > 0], r[r <= 0]
    gain, loss = float(wins.sum()), float(-losses.sum())
    rr = r / tr["stop_pct"].to_numpy(dtype=float)
    reasons = tr["reason"].value_counts().to_dict()
    names = {1: "stop", 2: "alvo", 3: "padrão de saída", 4: "tempo"}
    return {"trades": int(r.size), "acerto": round(float((r > 0).mean()), 3),
            "payoff": round(float(wins.mean() / -losses.mean()), 2) if wins.size and losses.size else None,
            "pf": round(min(gain / loss, 99.0), 2) if loss > 0 else 99.0,
            "expectativa_pct": round(float(r.mean()) * 100, 3), "expectativa_R": round(float(rr.mean()), 3),
            "maior_perda_R": round(float(-rr.min()), 2), "saidas": {names.get(int(k), str(k)): int(v)
                                                                   for k, v in reasons.items()}}


ACCOUNTS = {"US$50 risco 1,5% 3 posições": RiskRules(equity0=50.0, risk_per_trade=0.015),
            "US$50 risco 1,5% 5 posições": RiskRules(equity0=50.0, risk_per_trade=0.015, max_positions=5),
            "US$50 risco 1,0% 5 posições": RiskRules(equity0=50.0, risk_per_trade=0.01, max_positions=5),
            "US$120 risco 1,0% 5 posições": RiskRules(equity0=120.0, risk_per_trade=0.01, max_positions=5)}


def account(tr: pd.DataFrame, rules: RiskRules, costs: Costs, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    res = simulate_portfolio(tr, rules, costs, start, end)
    m = res.metrics
    monthly = equity_monthly(res.equity, rules.equity0, end)
    weeks = max((end - start).days / 7, 1e-9)
    return {"trades": int(m["trades"]), "por_semana": round(m["trades"] / weeks, 2), "pf": round(m["profit_factor"], 2),
            "lucro_usd": round(m["total_return"] * rules.equity0, 2),
            "media_mensal_pct": round(float(monthly.mean()) * 100, 2) if len(monthly) else None,
            "mediana_mensal_pct": round(float(monthly.median()) * 100, 2) if len(monthly) else None,
            "meses_positivos": round(float((monthly > 0).mean()), 2) if len(monthly) else None,
            "meses_acima_de_2pct": round(float((monthly >= 0.02).mean()), 2) if len(monthly) else None,
            "max_dd": round(m["max_drawdown"], 4), "monte_carlo": monte_carlo(monthly)}


def run(root: Path, ex: Any = None) -> dict[str, Any]:
    lab = Lab(root)
    lab.load(ex)
    costs = Costs()
    tables = build_tables(lab, costs)
    cands = {c["key"]: c for c in candidates()}
    split = FIRST_TEST
    rows = []
    for key, tr in tables.items():
        c = cands[key]
        p1, p2 = tr[tr["t_in"] < split], tr[tr["t_in"] >= split]
        rows.append({"key": key, "tipo": c["tipo"], "tf": c["tf"], "membros": list(c["membros"]),
                     "macro": c["macro"], "alvo": c["alvo"], "kmin": c.get("kmin"),
                     "total": _stats(tr), "periodo1": _stats(p1), "periodo2": _stats(p2)})
    # walk-forward selection across ALL candidates with enough history (the honest number)
    usable = {k: v for k, v in tables.items() if len(v) >= 30}
    wf = lab.walk_forward({"biblioteca": list(usable)}, usable)["biblioteca"]
    oos = wf["oos_trades"]
    picks = [{"janela": p["window"], "setup": p["config"], "t_treino": round(p["train_tstat"], 2)} for p in wf["picks"]]
    accounts: dict[str, Any] = {}
    if oos is not None and len(oos):
        accounts = {label: account(oos, rules, costs, split, lab.end) for label, rules in ACCOUNTS.items()}
        sr = simulate_portfolio(oos, lab.rules, costs, split, lab.end).metrics
        dsr = deflated_sharpe(sr["daily_sr"], sr["days"], sr["daily_skew"], sr["daily_kurt"], wf["trial_srs"]) \
            if sr["days"] > 2 and np.isfinite(sr["daily_sr"]) else 0.0
    else:
        dsr = 0.0
    # the candidates the walk-forward kept choosing, each run alone on the real accounts (for reference only:
    # they are known to be good *because* the selection saw them do well)
    chosen = list(dict.fromkeys(p["setup"] for p in picks))
    chosen_accounts = {k: {label: account(tables[k], rules, costs, split, lab.end) for label, rules in ACCOUNTS.items()}
                       for k in chosen}
    top = sorted((r for r in rows if r["periodo1"].get("trades", 0) >= 30 and r["periodo2"].get("trades", 0) >= 20),
                 key=lambda r: -(r["periodo1"]["pf"] or 0))[:15]
    return {"version": VERSION, "generated_at": pd.Timestamp.now(tz="UTC").isoformat(), "catalogo": CATALOG,
            "candidatos": len(rows), "linhas": rows, "top_periodo1": [r["key"] for r in top],
            "walk_forward": {"escolhas": picks, "oos": _stats(oos) if oos is not None else {}, "contas": accounts,
                             "dsr": round(float(dsr), 3)}, "escolhidos_sozinhos": chosen_accounts}


def summarize(rep: dict[str, Any]) -> list[str]:
    lines = [f"LIVRO v{rep['version']} | {rep['candidatos']} candidatos "
             "(16 estratégias, pares, k-de-n, ±filtro macro, ±saída por padrão)"]
    for r in rep["linhas"]:
        if r["tipo"] == "individual":
            t, p1, p2 = r["total"], r["periodo1"], r["periodo2"]
            if not t.get("trades"):
                continue
            lines.append(f"LIVRO individual {r['tf']} {r['membros'][0]} macro={r['macro']} alvo={r['alvo']} "
                         f"saida_padrao={r['key'].endswith('True')} | "
                         f"n={t['trades']} acerto {t['acerto']:.0%} payoff {t['payoff']} PF {t['pf']} "
                         f"exp {t['expectativa_R']}R | p1 PF {p1.get('pf')} p2 PF {p2.get('pf')} | saídas {t['saidas']}")
    for key in rep["top_periodo1"]:
        r = next(x for x in rep["linhas"] if x["key"] == key)
        lines.append(f"LIVRO top-p1 {key} | p1 n={r['periodo1']['trades']} PF {r['periodo1']['pf']} → "
                     f"p2 n={r['periodo2']['trades']} PF {r['periodo2']['pf']} acerto {r['periodo2']['acerto']:.0%}")
    wf = rep["walk_forward"]
    lines.append(f"LIVRO WALK-FORWARD (escolha cega a cada 6 meses entre {rep['candidatos']}) | OOS {wf['oos']} | "
                 f"DSR {wf['dsr']}")
    for p in wf["escolhas"]:
        lines.append(f"LIVRO escolha {p['janela']}: {p['setup']} (t treino {p['t_treino']})")
    def fmt(a: dict[str, Any]) -> str:
        mc = a["monte_carlo"]
        return (f"n={a['trades']} ({a['por_semana']}/sem) PF {a['pf']} lucro US${a['lucro_usd']:+.2f} | mês médio "
                f"{a['media_mensal_pct']}% mediana {a['mediana_mensal_pct']}% | meses+ {a['meses_positivos']} "
                f"meses≥2% {a['meses_acima_de_2pct']} | DD {a['max_dd']:.1%} | MC ano p5 {mc.get('year_p5', 0):+.1%} "
                f"p50 {mc.get('year_p50', 0):+.1%}")

    for label, a in wf["contas"].items():
        lines.append(f"LIVRO conta WF {label}: {fmt(a)}")
    for key, accs in rep.get("escolhidos_sozinhos", {}).items():
        for label, a in accs.items():
            lines.append(f"LIVRO sozinho {key} | {label}: {fmt(a)}")
    return lines
