"""Team review of an external backtest report: Python computes the facts, the agents debate them.

``report_metrics`` derives, deterministically, what a trade report really implies: expectancy in R,
the risk per trade the numbers imply, whether drawdown and return are even consistent with the stated
win rate/payoff (Monte Carlo), statistical significance, the odds of the losing streak, the trades
needed to tell the edge from luck, and risk of ruin. ``debate`` then runs Gauss → Ada → Vera, each
seeing the facts, the TA study's measured record of the patterns involved, and the previous speakers.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from mqmas.agents.base import Agent
from mqmas.llm import prompts
from mqmas.llm.router import LLMRouter
from mqmas.models import AgentMessage, AgentRole, Vote


class TradeReport(BaseModel):
    trades: int = Field(gt=0, le=1_000_000)
    win_rate: float = Field(gt=0, lt=1)
    payoff: float = Field(gt=0)  # average win / average loss
    net_return_pct: float  # total, after fees
    max_consecutive_losses: int = Field(ge=0)
    max_drawdown_pct: float = Field(ge=0)
    patterns: list[str] = Field(default_factory=list)  # ids from the pattern library
    description: str = ""


class ReviewOpinion(BaseModel):
    analysis: str
    vulnerabilities: list[str] = Field(default_factory=list)
    verdict: Literal["VIÁVEL", "NÃO VIÁVEL", "INCONCLUSIVO"]
    vote: Vote


def _streak_probability(n: int, p_loss: float, run: int) -> float:
    """P(at least one run of ``run`` consecutive losses in ``n`` independent trades)."""
    if run <= 0:
        return 1.0
    state = np.zeros(run)  # state[k] = P(current losing run == k, no full run yet)
    state[0] = 1.0
    hit = 0.0
    for _ in range(n):
        new = np.zeros(run)
        new[0] = state.sum() * (1 - p_loss)
        new[1:] = state[:-1] * p_loss
        hit += state[-1] * p_loss
        state = new
    return float(hit)


def _max_dd_r(w: float, b: float, n: int, paths: int, rng: np.random.Generator, drift: float = 0.0) -> np.ndarray:
    """Max drawdown, in R, of ``paths`` random sequences of n trades (+b or −1 R, plus a per-trade drift)."""
    r = np.where(rng.random((paths, n)) < w, b, -1.0) + drift
    eq = np.cumsum(r, axis=1)
    peak = np.maximum.accumulate(np.maximum(eq, 0.0), axis=1)
    return np.asarray((peak - eq).max(axis=1))


def report_metrics(rep: TradeReport, seed: int = 7) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n, w, b = rep.trades, rep.win_rate, rep.payoff
    exp_r = w * b - (1 - w)
    sd_r = math.sqrt(w * b * b + (1 - w) - exp_r * exp_r)
    out: dict[str, Any] = {"expectativa_bruta_R": round(exp_r, 3), "desvio_por_trade_R": round(sd_r, 3),
                           "kelly_cheio": round(w - (1 - w) / b, 3),
                           "retorno_sobre_drawdown": round(rep.net_return_pct / rep.max_drawdown_pct, 2)
                           if rep.max_drawdown_pct else None}
    # Hypothesis A — win rate and payoff are net of fees: then risk per trade follows from the return.
    r_a = rep.net_return_pct / (n * exp_r) if exp_r > 0 else None
    dd_sim = _max_dd_r(w, b, n, 20_000, rng)
    if r_a and r_a > 0:
        dd_r = rep.max_drawdown_pct / r_a
        out["hipotese_liquida"] = {
            "risco_implicito_por_trade_pct": round(r_a, 3),
            "drawdown_observado_em_R": round(dd_r, 1),
            "drawdown_simulado_R_mediana_p99": [round(float(np.median(dd_sim)), 1),
                                                 round(float(np.quantile(dd_sim, 0.99)), 1)],
            "probabilidade_de_um_drawdown_assim": round(float((dd_sim >= dd_r).mean()), 4),
        }
    # Hypothesis B — win rate/payoff are gross: risk per trade from the drawdown, costs from the gap.
    r_b = rep.max_drawdown_pct / float(np.median(dd_sim))
    gross = n * exp_r * r_b
    cost_per_trade = (gross - rep.net_return_pct) / n
    net_mean_r = rep.net_return_pct / (n * r_b)
    out["hipotese_bruta"] = {
        "risco_por_trade_pct_estimado": round(r_b, 3),
        "retorno_bruto_esperado_pct": round(gross, 1),
        "custo_implicito_por_trade_em_R": round(cost_per_trade / r_b, 3),
        "fracao_da_vantagem_comida_por_custos": round(cost_per_trade / (exp_r * r_b), 2) if exp_r > 0 else None,
        "expectativa_liquida_R": round(net_mean_r, 3),
    }
    t = net_mean_r / sd_r * math.sqrt(n)
    out["estatistica_t_liquida"] = round(t, 2)
    out["p_valor_aprox"] = round(0.5 * math.erfc(t / math.sqrt(2)), 3)
    out["trades_para_significancia_t2"] = int(math.ceil((2 * sd_r / net_mean_r) ** 2)) if net_mean_r > 0 else None
    p_loss = 1 - w
    out["prob_sequencia_de_losses"] = {
        f"{rep.max_consecutive_losses}_em_{n}": round(_streak_probability(n, p_loss, rep.max_consecutive_losses), 3),
        f"{rep.max_consecutive_losses}_em_1000": round(_streak_probability(1000, p_loss,
                                                                            rep.max_consecutive_losses), 3),
        "esperada_em_1000": round(math.log(1000 * w) / math.log(1 / p_loss), 1)}
    # Risk of ruin with the *net* edge over 1000 trades, compounding at fixed-fraction risk r_b
    drift = net_mean_r - exp_r  # shift the gross outcomes down to the net mean
    r = (np.where(rng.random((4000, 1000)) < w, b, -1.0) + drift) * r_b / 100
    eq = np.cumprod(1 + np.maximum(r, -0.99), axis=1)
    dd = 1 - eq / np.maximum.accumulate(np.maximum(eq, 1.0), axis=1)
    out["risco_de_ruina_em_1000_trades"] = {"cair_50pct_do_inicio": round(float((eq.min(axis=1) <= 0.5).mean()), 3),
                                            "drawdown_de_30pct": round(float((dd.max(axis=1) >= 0.3).mean()), 3)}
    out["drawdown_p95_em_1000_trades_pct"] = round(float(np.quantile(dd.max(axis=1), 0.95)) * 100, 1)
    return out


def pattern_evidence(study: dict[str, Any] | None, names: list[str]) -> dict[str, Any]:
    ev = (study or {}).get("events", {})
    return {n: ev.get(n, "não está na biblioteca") for n in names}


class _Speaker(Agent):
    def __init__(self, router: LLMRouter, role: AgentRole, prompt: str) -> None:
        self.role = role
        self.system_prompt = prompt
        super().__init__(router)


REVIEW_TASK = """

TAREFA ESPECIAL: revisão crítica de um relatório de backtest trazido pelo dono da mesa. Em
"fatos_calculados" estão as contas feitas em Python (trate como fatos; não recalcule). Em
"evidencia_da_biblioteca" está o histórico medido pela mesa dos padrões usados (veredito, acerto já
descontada a tendência, PF após taxas). Responda com severidade: o sistema é viável para capital real?
O par acerto/payoff compensa o drawdown? Que métricas mascaram vulnerabilidades? Cite números.
Responda em JSON com analysis (até 6 frases), vulnerabilities (lista curta), verdict
(VIÁVEL | NÃO VIÁVEL | INCONCLUSIVO) e vote (APPROVE só se for viável para capital real)."""


async def debate(router: LLMRouter, rep: TradeReport, study: dict[str, Any] | None) -> dict[str, Any]:
    facts = report_metrics(rep)
    names = rep.patterns or ["engolfo_alta", "ob_alta_reteste", "adx_tendencia_alta", "combo_adx_ob_engolfo"]
    evidence = pattern_evidence(study, names)
    base = {"relatorio": rep.model_dump(), "fatos_calculados": facts, "evidencia_da_biblioteca": evidence}
    order = [(AgentRole.ECONOMETRICIAN, prompts.ECONOMETRICIAN), (AgentRole.CHIEF_QUANT, prompts.CHIEF_QUANT),
             (AgentRole.CRO, prompts.CRO)]
    said: list[dict[str, Any]] = []
    messages: list[AgentMessage] = []
    for role, prompt in order:
        sp = _Speaker(router, role, prompt + REVIEW_TASK)
        res = await sp.ask({**base, "falas_anteriores": said}, ReviewOpinion)
        if res is None:
            messages.append(sp.abstain("review"))
            said.append({"agente": sp.persona.name, "texto": "absteve-se (LLM indisponível)"})
            continue
        op, meta = res
        text = f"{op.verdict}: {op.analysis}"
        messages.append(sp.message(text, debate_id="review", vote=op.vote, provider=meta.provider,
                                   data={"vulnerabilidades": op.vulnerabilities}))
        said.append({"agente": sp.persona.name, "veredito": op.verdict, "texto": op.analysis,
                     "vulnerabilidades": op.vulnerabilities})
    return {"fatos": facts, "evidencia": evidence, "debate": said, "messages": messages}
