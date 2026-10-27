from __future__ import annotations

import asyncio
import json

import httpx

from mqmas.agents.review import TradeReport, _streak_probability, debate, report_metrics
from mqmas.llm.router import LLMRouter, ProviderConfig

USER = TradeReport(trades=100, win_rate=0.52, payoff=1.8, net_return_pct=4.5, max_consecutive_losses=7,
                   max_drawdown_pct=11.2)


def test_streak_probability_matches_brute_force():
    import itertools

    n, p, run = 10, 0.5, 3
    hits = sum(any(all(x[i + k] for k in range(run)) for i in range(n - run + 1))
               for x in itertools.product([0, 1], repeat=n))
    assert abs(_streak_probability(n, p, run) - hits / 2 ** n) < 1e-12


def test_inconsistent_report_is_exposed():
    m = report_metrics(USER)
    assert m["expectativa_bruta_R"] == 0.456
    assert m["hipotese_liquida"]["probabilidade_de_um_drawdown_assim"] < 0.001  # net figures cannot give 11 % DD
    assert m["hipotese_bruta"]["fracao_da_vantagem_comida_por_custos"] > 0.9
    assert abs(m["estatistica_t_liquida"]) < 1  # indistinguishable from luck


def test_three_agents_debate_with_facts_and_evidence():
    seen: list[dict] = []
    body = {"analysis": "custos comem a vantagem", "vulnerabilities": ["t=0,18"], "verdict": "NÃO VIÁVEL",
            "vote": "REJECT"}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(json.loads(req.content)["messages"][-1]["content"]))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(body)}}]})

    router = LLMRouter([ProviderConfig("groq", "https://g.test", "k", "m")],
                       client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    study = {"events": {"engolfo_alta": {"4h": {"veredito": "sem efeito", "pf_operacao": 0.89}}}}
    res = asyncio.run(debate(router, USER, study))
    assert [d["veredito"] for d in res["debate"]] == ["NÃO VIÁVEL"] * 3
    assert seen[0]["evidencia_da_biblioteca"]["engolfo_alta"]["4h"]["pf_operacao"] == 0.89
    assert len(seen[2]["falas_anteriores"]) == 2  # Vera hears Gauss and Ada
