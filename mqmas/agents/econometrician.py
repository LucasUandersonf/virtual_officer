"""Agent 2 — Econometrician & Probability (memory + Sharpe + hit-rate gate)."""

from __future__ import annotations

from typing import Any

from mqmas.agents.base import Agent, snapshot_brief
from mqmas.config import Settings
from mqmas.llm import prompts
from mqmas.models import AgentMessage, AgentRole, EconometricianOpinion, QuantSnapshot, SetupStats, Vote


class Econometrician(Agent):
    role = AgentRole.ECONOMETRICIAN
    system_prompt = prompts.ECONOMETRICIAN

    @staticmethod
    def stats_pass(stats: SetupStats, s: Settings) -> tuple[bool, list[str]]:
        fails: list[str] = []
        n = stats.backtest_trades + stats.live_trades
        if n < s.min_sample_size:
            fails.append(f"amostra n={n} < {s.min_sample_size}")
        if stats.win_rate < s.min_win_rate:
            fails.append(f"taxa de acerto {stats.win_rate:.1%} < {s.min_win_rate:.0%}")
        if stats.trade_sharpe < s.min_trade_sharpe:
            fails.append(f"Sharpe/trade {stats.trade_sharpe:.2f} < {s.min_trade_sharpe}")
        if stats.kelly_full <= 0:
            fails.append(f"Kelly negativo ({stats.kelly_full:.3f}) — expectativa não paga o risco")
        return not fails, fails

    async def deliberate(self, debate_id: str, snap: QuantSnapshot, stats: SetupStats,
                         memory: list[dict[str, Any]], s: Settings) -> AgentMessage:
        ok, fails = self.stats_pass(stats, s)
        data = stats.model_dump()
        headline = (f"p̂={stats.win_rate:.1%} (IC95 inf {stats.win_rate_lower_95:.1%}), "
                    f"n={stats.backtest_trades}+{stats.live_trades}, payoff {stats.payoff_ratio:.2f}, "
                    f"Sharpe/trade {stats.trade_sharpe:.2f}, Kelly {stats.kelly_full:.3f}.")
        if not ok:
            # No LLM call needed: the evidence already fails the mandate.
            return self.message(f"📉 {headline} Reprovo: " + "; ".join(fails) + ".",
                                debate_id=debate_id, vote=Vote.REJECT, data=data)
        res = await self.ask({"snapshot": snapshot_brief(snap), "estatisticas_setup": data,
                              "memoria_trades_recentes": memory[:10]}, EconometricianOpinion)
        if res is None:
            return self.abstain(debate_id)
        op, meta = res
        return self.message(f"📊 {headline} {op.assessment}", debate_id=debate_id, vote=op.vote,
                            provider=meta.provider, data=data)
