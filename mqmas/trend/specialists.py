"""The specialists: each one reads one kind of evidence about a proposed purchase and gives an opinion.

The trend model *proposes* (it is the only part with strong out-of-sample evidence); a specialist can only
**confirm**, **reduce** (half size) or **veto**. None of them can create a trade or make one bigger.

Every rule below was written before looking at the live results, and every specialist starts in *shadow*
mode: its opinion is recorded with the coin's price, the scoreboard later measures what following it would
have done, and only a specialist that proved it helps should be switched to active (``TREND_ACTIVE_SPECIALISTS``).

Where the evidence comes from:

* **tecnico** confirms only with signals that passed our own out-of-sample studies (``EVIDENCE``). It never
  vetoes on a bearish pattern: the TA study found 8 bearish signals that worked *in reverse* and none that
  worked as advertised. Its one reducing rule (strong resistance right above the price) is unvalidated.
* **fundamentalista** reads supply and protocol data (CoinGecko, DefiLlama). No long public history exists
  to backtest it, so it can only prove itself live.
* **sentimento** reads the Fear & Greed index and the perpetual funding rate. In the swing study (v6) these
  explained the market but gave no edge on their own; here they only flag crowded, euphoric moments.
* **risco** (Vera, LLM) runs a pre-mortem over the proposal *and the other specialists' opinions*. Without an
  LLM key it abstains; it never blocks the desk by being offline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
import pandas as pd

log = logging.getLogger(__name__)

CONFIRM, REDUCE, VETO, ABSTAIN = "confirmar", "reduzir", "vetar", "abster"
MULTIPLIER = {CONFIRM: 1.0, REDUCE: 0.5, VETO: 0.0, ABSTAIN: 1.0}
HTTP_TIMEOUT = 15.0

# Out-of-sample evidence behind each confirming signal (Binance spot, fees included). Source: the research
# reports on the production volume, 27/out/2026.
EVIDENCE: dict[str, str] = {
    "swing 1d rompimento": "arena 2023-07→2026-10 fora da amostra: n=274, acerto 42%, PF 1,62",
    "swing 1d rompimento+medias": "arena 2023-07→2026-10 fora da amostra: n=73, acerto 48%, PF 2,12",
    "swing 2d rompimento": "arena 2023-07→2026-10 fora da amostra: n=151, acerto 43%, PF 2,36",
    "swing 3d rompimento": "arena 2023-07→2026-10 fora da amostra: n=112, acerto 50%, PF 2,23",
    "swing 3d triangulo": "arena 2023-07→2026-10 fora da amostra: n=65, acerto 49%, PF 3,24",
    "classico rompimento": "estudo clássico 2022-07→2026-10, zero parâmetros: n=119, acerto 46%, PF 2,11",
    "85_TREND_Mean_Reversion_Ignition": "estudo de padrões v5 (1d): n=131 operações, acerto 57%, PF 2,28",
}
RECENT_DAYS = 3
RESISTANCE_ATR = 1.0  # tecnico: a strong resistance closer than this above the price → reduce (shadow rule)
FDV_RATIO_MAX = 2.0  # fundamentalista: FDV ≥ 2× market cap → at least half the supply is still to come
TVL_DROP_MAX = -30.0  # fundamentalista: protocol TVL down more than 30 % in a month
GREED_MAX = 80  # sentimento: extreme greed
FUNDING_MAX = 0.0005  # sentimento: > 0.05 % per 8 h (≈ 55 %/year paid by longs) = crowded long


@dataclass
class Proposal:
    symbol: str
    weight: float  # target weight proposed by the trend model
    current_weight: float
    signal: float  # ensemble signal 0..1
    price: float
    daily: pd.DataFrame


@dataclass
class Opinion:
    specialist: str
    verdict: str
    reason: str
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def multiplier(self) -> float:
        return MULTIPLIER[self.verdict]


class Specialist(Protocol):
    name: str

    async def opine(self, p: Proposal, others: list[Opinion]) -> Opinion: ...


# ---------------------------------------------------------------------------------------------- technical


def validated_signals(daily: pd.DataFrame) -> list[str]:
    """Validated bullish signals that fired on the last ``RECENT_DAYS`` daily closes."""
    from mqmas.book import swing
    from mqmas.research.strategies import classic_triggers, features
    from mqmas.ta.signals import all_events

    found: list[str] = []
    for k in range(RECENT_DAYS):
        d = daily.iloc[: len(daily) - k] if k else daily
        if len(d) < 220:
            break
        hits, _ = swing.scan(d)
        found += [f"swing {h.setup['tf']} {h.setup['nome']}" for h in hits]
    f = features(daily)
    if bool(classic_triggers(f)["rompimento"].iloc[-RECENT_DAYS:].any()):
        found.append("classico rompimento")
    ev, _, _ = all_events(daily)
    name = "85_TREND_Mean_Reversion_Ignition"
    if name in ev.columns and bool(ev[name].iloc[-RECENT_DAYS:].any()):
        found.append(name)
    return sorted(set(found))


class Technical:
    name = "tecnico"

    async def opine(self, p: Proposal, others: list[Opinion]) -> Opinion:
        from mqmas.ta.confluence import live_score
        from mqmas.ta.levels import price_map

        d = p.daily
        if len(d) < 220:
            return Opinion(self.name, ABSTAIN, "histórico diário curto demais para a análise técnica")
        sigs = validated_signals(d)
        pm = price_map(d)
        sc = live_score(d, None, "1d")
        atr = pm["atr"]
        near = [z for z in pm["resistencias"] if z["forca"] >= 2 and z["preco"] - pm["preco"] <= RESISTANCE_ATR * atr]
        data = {"sinais_validados": {s: EVIDENCE.get(s, "") for s in sigs}, "score_confluencia": sc["score"],
                "resistencias": pm["resistencias"][:3], "suportes": pm["suportes"][:3], "atr_pct": pm["atr_pct"]}
        if near:
            z = near[0]
            return Opinion(self.name, REDUCE, f"resistência forte a {z['distancia_pct']:+.1f}% ({', '.join(z['fontes'][:3])}): "
                                              "pouco espaço até o próximo teto", data)
        if sigs:
            return Opinion(self.name, CONFIRM, "sinais validados nos últimos 3 dias: " + ", ".join(sigs), data)
        return Opinion(self.name, CONFIRM, f"sem resistência forte por perto; score de confluência {sc['score']:.0f} "
                                           "(informativo: o score não passou na validação)", data)


# ---------------------------------------------------------------------------------------------- fundamentals


class Fundamental:
    name = "fundamentalista"

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._http = client
        self._llama: list[dict[str, Any]] | None = None

    async def _get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        r = await self._http.get(url, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        return r.json()

    async def _coin(self, base: str) -> dict[str, Any] | None:
        found = await self._get("https://api.coingecko.com/api/v3/search", {"query": base})
        coins = [c for c in found.get("coins", []) if str(c.get("symbol", "")).upper() == base.upper()]
        if not coins:
            return None
        best = min(coins, key=lambda c: c.get("market_cap_rank") or 10**9)
        return dict(await self._get(f"https://api.coingecko.com/api/v3/coins/{best['id']}",
                                    {"localization": "false", "tickers": "false", "community_data": "false",
                                     "developer_data": "false"}))

    async def _protocol(self, gecko_id: str) -> dict[str, Any] | None:
        if self._llama is None:
            try:
                self._llama = list(await self._get("https://api.llama.fi/protocols"))
            except (httpx.HTTPError, ValueError):
                self._llama = []
        return next((p for p in self._llama if p.get("gecko_id") == gecko_id), None)

    async def opine(self, p: Proposal, others: list[Opinion]) -> Opinion:
        base = p.symbol.split("/")[0]
        try:
            coin = await self._coin(base)
        except (httpx.HTTPError, ValueError) as exc:
            return Opinion(self.name, ABSTAIN, f"dados fundamentalistas indisponíveis ({type(exc).__name__})")
        if not coin:
            return Opinion(self.name, ABSTAIN, f"{base} não encontrado na CoinGecko")
        md = coin.get("market_data") or {}
        mc = float((md.get("market_cap") or {}).get("usd") or 0.0)
        fdv = float((md.get("fully_diluted_valuation") or {}).get("usd") or 0.0)
        circ, total, mx = (float(md.get(k) or 0.0) for k in ("circulating_supply", "total_supply", "max_supply"))
        ratio = fdv / mc if mc > 0 and fdv > 0 else None
        proto = await self._protocol(str(coin.get("id")))
        tvl_1m = float(proto["change_1m"]) if proto and proto.get("change_1m") is not None else None
        data = {"rank": coin.get("market_cap_rank"), "market_cap_usd": mc, "fdv_usd": fdv,
                "fdv_sobre_mc": round(ratio, 2) if ratio else None,
                "circulante_pct": round(circ / (mx or total) * 100, 1) if (mx or total) else None,
                "categorias": (coin.get("categories") or [])[:4],
                "tvl_usd": proto.get("tvl") if proto else None, "tvl_var_30d_pct": tvl_1m}
        flags = []
        if ratio and ratio >= FDV_RATIO_MAX:
            flags.append(f"FDV {ratio:.1f}× o valor de mercado: muita oferta ainda por entrar (diluição)")
        if tvl_1m is not None and tvl_1m <= TVL_DROP_MAX:
            flags.append(f"TVL do protocolo caiu {tvl_1m:.0f}% em 30 dias")
        if flags:
            return Opinion(self.name, REDUCE, "; ".join(flags), data)
        rank = data["rank"]
        return Opinion(self.name, CONFIRM, f"sem alerta de oferta ou de protocolo (rank {rank}, FDV/MC "
                                           f"{ratio:.2f})" if ratio else f"sem alerta (rank {rank})", data)


# ---------------------------------------------------------------------------------------------- sentiment


class Sentiment:
    name = "sentimento"

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._http = client
        self._fng: int | None = None
        self._fng_fetched = False  # one request per run: the index is daily

    async def _fear_greed(self) -> int | None:
        if not self._fng_fetched:
            self._fng_fetched = True
            try:
                r = await self._http.get("https://api.alternative.me/fng/", params={"limit": 1}, timeout=HTTP_TIMEOUT)
                r.raise_for_status()
                self._fng = int(r.json()["data"][0]["value"])
            except (httpx.HTTPError, ValueError, KeyError, IndexError):
                self._fng = None
        return self._fng

    async def _funding(self, symbol: str) -> float | None:
        try:
            r = await self._http.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                                     params={"symbol": symbol.replace("/", "")}, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            return float(r.json()["lastFundingRate"])
        except (httpx.HTTPError, ValueError, KeyError):
            return None

    async def opine(self, p: Proposal, others: list[Opinion]) -> Opinion:
        fng, fund = await self._fear_greed(), await self._funding(p.symbol)
        data = {"medo_ganancia": fng, "funding_8h": fund}
        if fng is None and fund is None:
            return Opinion(self.name, ABSTAIN, "sentimento e funding indisponíveis", data)
        flags = []
        if fng is not None and fng >= GREED_MAX:
            flags.append(f"ganância extrema ({fng})")
        if fund is not None and fund > FUNDING_MAX:
            flags.append(f"funding {fund * 100:.3f}%/8h: comprados lotados")
        if flags:
            return Opinion(self.name, REDUCE, " e ".join(flags), data)
        parts = ([f"medo e ganância {fng}"] if fng is not None else []) + (
            [f"funding {fund * 100:.3f}%/8h"] if fund is not None else [])
        return Opinion(self.name, CONFIRM, "sem euforia: " + ", ".join(parts), data)


# ---------------------------------------------------------------------------------------------- risk (Vera)


class Risk:
    """Vera's pre-mortem over the proposal and the other opinions; abstains without an LLM."""

    name = "risco"

    def __init__(self, router: Any) -> None:
        from mqmas.agents.cro import ChiefRiskOfficer

        self._router = router
        self._cro = ChiefRiskOfficer(router)

    async def opine(self, p: Proposal, others: list[Opinion]) -> Opinion:
        from mqmas.models import CroOpinion, Vote

        if not getattr(self._router, "enabled", False):
            return Opinion(self.name, ABSTAIN, "sem chave de IA: a Vera não opina (não bloqueia)")
        payload = {"ativo": p.symbol, "proposta": f"alocar {p.weight:.1%} da banca (hoje {p.current_weight:.1%})",
                   "estrategia": "tendência: 9 canais Donchian, tamanho por volatilidade (validada 2020–2026)",
                   "sinal_tendencia": round(p.signal, 2), "preco": p.price,
                   "pareceres_da_equipe": [{"especialista": o.specialist, "voto": o.verdict, "motivo": o.reason}
                                           for o in others]}
        res = await self._cro.ask(payload, CroOpinion)
        if res is None:
            return Opinion(self.name, ABSTAIN, "IA indisponível agora: a Vera não opina (não bloqueia)")
        op, meta = res
        high = [f for f in op.pre_mortem if f.likelihood == "high" and not f.mitigation.strip()]
        risks = " · ".join(f"[{f.likelihood}] {f.risk}" for f in op.pre_mortem[:3])
        data = {"provedor": meta.provider, "pre_mortem": [f.model_dump() for f in op.pre_mortem]}
        if op.vote is Vote.REJECT or high:
            return Opinion(self.name, VETO, f"pre-mortem: {risks}. {op.verdict}", data)
        return Opinion(self.name, CONFIRM, f"pre-mortem: {risks}. {op.verdict}", data)

