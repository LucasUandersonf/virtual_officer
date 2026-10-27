"""Product-evaluation frameworks, as plain functions (no LLM): the numbers behind every agent's vote.

Iris (Head of Product)
* Opportunity Score (Dan Olsen / Ulwick): ``importance + max(importance - satisfaction, 0)``, 0-20.
* "Hair on fire" urgency: direct money loss, >= 5 h/week of critical time, or risk of losing clients.
  "Nice to have" posts are scored down hard — nobody pays $29 for them.
* Workarounds = willingness to pay: the more manual steps someone describes, the higher the WTP score.
* Time-to-Value: the buyer must be running in <= 5 minutes, or the MVP score collapses.
* Distribution with zero CAC: clear English search intent + riding a big ecosystem (n8n, Stripe, ...).
* Value-based price: hours saved x $50-150/h; the price is ~15 % of the low end of one month of value.
* Moat: >= 2-3 integrated systems; one-API "ChatGPT wrappers" are commodities.
* Workaround detection: 3+ tools glued together by hand is the definitive buying signal (WTP bonus).
* MVP scope: the smallest unit that solves 100 % of the job in one use; > 10 steps = feature creep.
* Cognitive friction: keys, credentials and servers the buyer must handle vs what the audience tolerates.
* Action bias: does this audience pay for software, or reach for free/pirated tools?
* Price anchors: $19 / $29 / $49 / $99 / $149 (Iris); Leo fine-tunes with 7/9 charm prices and elasticity.

Leo uses ``charm_price``; Maya uses ``value_anchor``; Nina uses the contribution margin helpers.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

HOURLY_LOW, HOURLY_HIGH = 50, 150  # opportunity cost of a US/EU consultant or developer
DEV_RATE = 100  # what a freelancer charges to build an integration
PRICE_MIN, PRICE_MAX = 19, 149
TTV_LIMIT_MIN = 5

BIG_ECOSYSTEMS = {"n8n", "make", "zapier", "openai", "supabase", "airtable", "stripe", "shopify", "google sheets", "gmail",
                  "calendly", "cal.com", "slack", "notion", "hubspot", "woocommerce", "quickbooks", "xero", "whatsapp"}

# ------------------------------------------------------------------ post-level signals
_MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(k)?\b", re.IGNORECASE)
_LOSS_WORDS = re.compile(r"\b(los(e|ing|t)|cost(s|ing)?|wast(e|ing|ed)|leav(e|ing)|missed|unpaid|owed|overdue|refunds?|bleed(ing)?|"
                         r"chargebacks?|forgot|forget)\b", re.IGNORECASE)
_LOSS_PHRASES = re.compile(r"\b(losing (money|revenue|sales)|lost (revenue|sales|money)|leaving money on the table|unpaid invoices?|"
                           r"cash ?flow)\b", re.IGNORECASE)
_TIME = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:hours?|hrs?|h)\b\s*(?:a|an|per|each|every)?\s*(day|week|wk|month|mo)\b", re.IGNORECASE)
_CHURN = re.compile(r"\b(los(e|ing|t)\s+(a\s+|my\s+|our\s+)?(clients?|customers?|deals?|leads?)|churn(ing)?|clients?\s+(leave|leaving|cancel|"
                    r"ghost)|no-?shows?)\b", re.IGNORECASE)
_NICE = re.compile(r"\b(would be (nice|cool|neat)|nice to have|just curious|for fun|side project|someday|not urgent|no rush)\b", re.IGNORECASE)
_FRUSTRATION = re.compile(r"\b(hate|nightmare|painful|sick of|tired of|tedious|broken|nothing works|too expensive|frustrat\w*|"
                          r"drives me crazy|killing me|waste of time)\b", re.IGNORECASE)
_SATISFIED = re.compile(r"\b(works (fine|well|great)|happy with|solved it|no problem|good enough)\b", re.IGNORECASE)
_STEP_SPLIT = re.compile(r"\s*(?:,|;|->|→|\bthen\b|\bafter that\b|\bnext\b|\bfinally\b|\band\b)\s*", re.IGNORECASE)
_MANUAL_VERB = re.compile(r"\b(download|export|upload|import|cop(y|ies)|paste|run|send|email|check|update|enter|type|open|move|sync|"
                          r"re-?enter|merge|format|rename|forward|attach|log|track|reconcile|chase|remind)\w*\b", re.IGNORECASE)
_PAYS = re.compile(r"\b(i pay|we pay|paying for|paid for|subscription|subscribed|worth (every penny|paying|the money)|would (happily |gladly )?pay|"
                   r"take my money|bought|purchased|licen[cs]e|budget for)\b", re.IGNORECASE)
_FREE = re.compile(r"\b(free (alternative|version|tool|option|plan only)|open[- ]source alternative|pirat\w*|crack(ed)?|cheapest|"
                   r"without paying|for free|can'?t afford|no budget)\b", re.IGNORECASE)
_TOOLS = re.compile(r"\b(spreadsheet|excel|google sheets?|csv|stripe|quickbooks|xero|notion|airtable|zapier|make|n8n|gmail|outlook|"
                    r"calendly|cal\.com|hubspot|shopify|slack|whatsapp|python script|script|crm|trello|clickup)\b", re.IGNORECASE)


@dataclass
class PostAssessment:
    urgency: str  # hair_on_fire | important | nice_to_have
    money_loss_usd: float
    hours_per_week: float
    churn_risk: bool
    workaround_steps: int
    tools_mentioned: int
    wtp: float  # 0-10
    importance: float  # 1-10
    satisfaction: float  # 1-10 (current solution)
    opportunity: float  # 0-20
    multi_tool_workaround: bool = False  # 3+ tools glued together by hand
    paying_signal: float = 0.0  # +1 pays for software, -1 hunts free/pirated tools

    def as_dict(self) -> dict[str, Any]:
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in asdict(self).items()}


def _money_loss(text: str) -> float:
    total = 0.0
    for m in _MONEY.finditer(text):
        window = text[max(0, m.start() - 70): m.end() + 70]
        if _LOSS_WORDS.search(window):
            v = float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1)
            total = max(total, v)
    return total


def _hours_per_week(text: str) -> float:
    best = 0.0
    for m in _TIME.finditer(text):
        v, unit = float(m.group(1)), m.group(2).lower()
        weekly = v * 5 if unit == "day" else v / 4.3 if unit in ("month", "mo") else v
        best = max(best, weekly)
    return best


def workaround_steps(text: str) -> int:
    """Manual steps described (clauses with a manual verb). 'download the CSV, upload it to Sheets, run my script, then email...' = 4."""
    return sum(1 for clause in _STEP_SPLIT.split(text) if clause and _MANUAL_VERB.search(clause))


def assess_post(text: str, engagement: int = 0, intent_hits: float = 0.0) -> PostAssessment:
    money = _money_loss(text)
    hours = _hours_per_week(text)
    churn = bool(_CHURN.search(text))
    loss_phrase = bool(_LOSS_PHRASES.search(text))
    steps = workaround_steps(text)
    tools = len({m.group(0).lower() for m in _TOOLS.finditer(text)})
    if money > 0 or loss_phrase or hours >= 5 or churn:
        urgency = "hair_on_fire"
    elif _NICE.search(text) or (steps == 0 and intent_hits == 0 and not _FRUSTRATION.search(text)):
        urgency = "nice_to_have"
    else:
        urgency = "important"
    multi_tool = tools >= 3 and steps >= 3
    wtp = min(10.0, 1.5 * steps + (2.0 if tools >= 2 else 1.0 if tools else 0.0) + (1.0 if money > 0 else 0.0) + (2.0 if multi_tool else 0.0))
    paying = max(-1.0, min(1.0, 0.5 * len(_PAYS.findall(text)) - 0.5 * len(_FREE.findall(text))))
    importance = 3.0 + {"hair_on_fire": 4.0, "important": 2.0, "nice_to_have": 0.0}[urgency]
    importance += min(2.0, math.log10(1 + max(engagement, 0))) + min(1.0, intent_hits / 2)
    importance = min(10.0, importance)
    satisfaction = 5.0 - (2.0 if steps >= 3 else 0.0) - (2.0 if _FRUSTRATION.search(text) else 0.0) - (1.0 if intent_hits else 0.0)
    satisfaction += 3.0 if _SATISFIED.search(text) else 0.0
    satisfaction = max(1.0, min(10.0, satisfaction))
    opportunity = importance + max(importance - satisfaction, 0.0)
    return PostAssessment(urgency, money, hours, churn, steps, tools, wtp, importance, satisfaction, opportunity, multi_tool, paying)


# ------------------------------------------------------------------ pricing and value
def charm_price(usd: float) -> int:
    """Nearest price ending in 7 or 9 inside the $19-149 band ($19, $27, $29, ... $47, $49, ... $97, $99, ... $149)."""
    ladder = [p for p in range(PRICE_MIN, PRICE_MAX + 1) if p % 10 in (7, 9)]
    target = min(max(usd, PRICE_MIN), PRICE_MAX)
    return min(ladder, key=lambda p: (abs(p - target), -p))


def value_anchor(hours_saved_month: float, integrations: int) -> dict[str, float]:
    """Monthly value of the buyer's time and what a freelancer would charge to build the same thing."""
    return {"value_month_low": hours_saved_month * HOURLY_LOW, "value_month_high": hours_saved_month * HOURLY_HIGH,
            "dev_build_cost": max(2, integrations) * 4 * DEV_RATE}


PRICE_ANCHORS = (19, 29, 49, 99, 149)


def value_price(hours_saved_month: float) -> int:
    """Iris's anchor: ~15 % of one month of the buyer's time at $50/h, snapped to the classic $19/29/49/99/149 points."""
    target = 0.15 * hours_saved_month * HOURLY_LOW
    return min(PRICE_ANCHORS, key=lambda p: (abs(p - target), p))


def friction_score(env_vars: int, credentials: int, needs_server: bool) -> float:
    """Cognitive friction of setup (0-10): every key to paste, every OAuth credential, and running a server."""
    return min(10.0, 0.75 * env_vars + 1.5 * credentials + (3.0 if needs_server else 0.0))


def friction_tolerance(technical_audience: bool) -> float:
    """What the buyer accepts before giving up: n8n/dev users tolerate more than non-technical owners."""
    return 8.0 if technical_audience else 5.5


# ------------------------------------------------------------------ idea-level verdict
@dataclass
class IdeaAssessment:
    urgency: str
    opportunity: float
    hair_on_fire_share: float
    wtp: float
    max_workaround_steps: int
    ttv_minutes: int
    distribution: float  # 0-10
    moat: int  # integrated systems
    commodity: bool
    hours_saved_month: float
    value_month_low: float
    value_month_high: float
    dev_build_cost: float
    price_usd: int
    no_brainer_ratio: float  # one month of value / price
    scope_steps: int  # MVP size (workflow steps); > MAX_SCOPE = feature creep
    friction: float  # 0-10
    friction_tolerance: float
    payer_habit: float  # -1..1
    multi_tool_share: float  # share of evidence with 3+ tools glued by hand
    mvp_score: float  # 0-100
    approve: bool
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in asdict(self).items()}


def distribution_score(search_queries: tuple[str, ...] | list[str], ecosystems: tuple[str, ...] | list[str]) -> float:
    eco = {e.lower() for e in ecosystems} & BIG_ECOSYSTEMS
    return min(10.0, min(6.0, 2.0 * len(eco)) + min(4.0, 1.0 * len(search_queries)))


MAX_SCOPE = 10


def assess_idea(signals: list[dict[str, Any]], *, ttv_minutes: int, integrations: tuple[str, ...] | list[str],
                search_queries: tuple[str, ...] | list[str], ecosystems: tuple[str, ...] | list[str], hours_saved_month: float,
                commodity: bool = False, scope_steps: int = 6, env_vars: int = 3, credentials: int = 1,
                needs_server: bool = False, technical_audience: bool = False, payer_prior: float = 0.3) -> IdeaAssessment:
    """``signals``: PostAssessment dicts of the topic's evidence (strongest first)."""
    top = sorted(signals, key=lambda a: -a.get("opportunity", 0))[:10] or [{}]
    opp = sum(a.get("opportunity", 0) for a in top) / len(top)
    hof = sum(1 for a in top if a.get("urgency") == "hair_on_fire") / len(top)
    nice = sum(1 for a in top if a.get("urgency") == "nice_to_have") / len(top)
    urgency = "hair_on_fire" if hof >= 0.3 else "nice_to_have" if nice > 0.5 else "important"
    wtp = max(a.get("wtp", 0) for a in top)
    steps = max(a.get("workaround_steps", 0) for a in top)
    dist = distribution_score(search_queries, ecosystems)
    moat = 0 if commodity else len(set(integrations))
    anchor = value_anchor(hours_saved_month, moat)
    price = value_price(hours_saved_month)
    ratio = anchor["value_month_low"] / price if price else 0.0
    multi = sum(1 for a in top if a.get("multi_tool_workaround")) / len(top)
    payer = 0.5 * payer_prior + 0.5 * (sum(a.get("paying_signal", 0.0) for a in top) / len(top))
    friction = friction_score(env_vars, credentials, needs_server)
    tolerance = friction_tolerance(technical_audience)
    ttv_factor = 1.0 if ttv_minutes <= TTV_LIMIT_MIN else 0.5 if ttv_minutes <= 10 else 0.2
    score = (30 * opp / 20 + 15 * wtp / 10 + 15 * dist / 10 + 15 * min(moat, 4) / 4
             + {"hair_on_fire": 25, "important": 12, "nice_to_have": 0}[urgency]) * ttv_factor
    score += 5 * multi + 5 * payer - max(0.0, friction - tolerance) * 3
    score = max(0.0, min(100.0, score))
    if commodity:
        score *= 0.3
    reasons = []
    if urgency == "nice_to_have":
        reasons.append("nice-to-have pain: no financial loss, critical time or churn risk")
    if ttv_minutes > TTV_LIMIT_MIN:
        reasons.append(f"time-to-value {ttv_minutes} min > {TTV_LIMIT_MIN} min")
    if commodity:
        reasons.append("commoditized AI idea (saturated, zero perceived value)")
    elif moat < 2:
        reasons.append(f"moat too thin: {moat} integrated system(s), need >= 2")
    if scope_steps > MAX_SCOPE:
        reasons.append(f"feature creep: {scope_steps} steps > {MAX_SCOPE} for one job")
    if friction > tolerance:
        reasons.append(f"entry barrier: friction {friction:.1f} > {tolerance:.1f} this audience tolerates")
    if payer < -0.3:
        reasons.append(f"audience avoids paying for software (payer habit {payer:.2f})")
    if score < 45:
        reasons.append(f"MVP score {score:.0f} < 45")
    return IdeaAssessment(urgency, opp, hof, wtp, steps, ttv_minutes, dist, moat, commodity, hours_saved_month,
                          anchor["value_month_low"], anchor["value_month_high"], anchor["dev_build_cost"], price, ratio,
                          scope_steps, friction, tolerance, round(payer, 2), multi, round(score, 1), not reasons, reasons)


# ------------------------------------------------------------------ CFO helpers
def contribution_margin(gross_cents: int, fee_cents: int) -> float:
    return (gross_cents - fee_cents) / gross_cents if gross_cents else 0.0
