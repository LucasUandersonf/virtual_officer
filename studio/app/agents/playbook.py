"""Built-in expert knowledge: recurring solopreneur pains and how to package a fix for each.

Two uses:
1. Research: classify mined posts into a topic (keyword match) so evidence adds up per pain.
2. Fallback: when no free LLM answers, the agents still produce a coherent idea, workflow, service
   and sales copy from these entries. The office keeps working at zero cost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Step:
    kind: str  # webhook | schedule | http | set | if | code | email | slack | sheets | wait | respond
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Topic:
    key: str
    keywords: tuple[str, ...]
    niche: str
    problem: str
    solution_type: str  # n8n | python
    product: str
    tagline: str
    mvp: str
    price_usd: int
    steps: tuple[Step, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    logic: str = ""  # python services: body of logic.py


_LEAD_LOGIC = '''"""Score inbound leads 0-100 and route hot ones."""

HOT_WORDS = ("budget", "asap", "urgent", "this week", "ready to", "quote", "pricing")
FREE_DOMAINS = ("gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com")


def process(payload: dict) -> dict:
    email = str(payload.get("email", "")).lower()
    message = str(payload.get("message", "")).lower()
    company = str(payload.get("company", "")).strip()
    score = 20
    if email and email.split("@")[-1] not in FREE_DOMAINS:
        score += 25
    if company:
        score += 15
    score += min(30, 10 * sum(w in message for w in HOT_WORDS))
    if len(message) > 140:
        score += 10
    score = min(score, 100)
    tier = "hot" if score >= 70 else "warm" if score >= 45 else "cold"
    return {"email": email, "score": score, "tier": tier, "notify": tier == "hot"}
'''

_INVOICE_LOGIC = '''"""Decide which overdue invoices get which reminder today."""

from datetime import date

STAGES = ((1, "friendly"), (7, "firm"), (14, "final"))


def process(payload: dict) -> dict:
    today = date.fromisoformat(payload.get("today") or date.today().isoformat())
    out = []
    for inv in payload.get("invoices", []):
        if inv.get("paid"):
            continue
        overdue = (today - date.fromisoformat(inv["due_date"])).days
        stage = None
        for days, name in STAGES:
            if overdue >= days:
                stage = name
        if stage and stage != inv.get("last_stage"):
            out.append({"invoice": inv.get("number"), "email": inv.get("email"), "stage": stage,
                        "days_overdue": overdue, "amount": inv.get("amount")})
    return {"reminders": out, "count": len(out)}
'''

_REVIEW_LOGIC = '''"""Classify a customer review and draft a reply."""

NEG = ("bad", "terrible", "late", "rude", "refund", "broken", "worst", "disappointed", "never")
POS = ("great", "amazing", "love", "excellent", "perfect", "recommend", "friendly", "fast")


def process(payload: dict) -> dict:
    text = str(payload.get("review", "")).lower()
    name = str(payload.get("name", "there")).split(" ")[0] or "there"
    stars = int(payload.get("stars") or 0)
    neg = sum(w in text for w in NEG) + (2 if 0 < stars <= 2 else 0)
    pos = sum(w in text for w in POS) + (2 if stars >= 4 else 0)
    sentiment = "negative" if neg > pos else "positive" if pos > neg else "neutral"
    if sentiment == "negative":
        reply = (f"Hi {name}, thank you for telling us. We're sorry we fell short. "
                 "Could you reply with your order details so we can make this right today?")
    elif sentiment == "positive":
        reply = f"Thank you so much, {name}! We're thrilled you had a great experience and hope to see you again soon."
    else:
        reply = f"Thanks for the feedback, {name}! If there's anything we can do better, just let us know."
    return {"sentiment": sentiment, "escalate": sentiment == "negative", "reply": reply}
'''

TOPICS: tuple[Topic, ...] = (
    Topic(
        "no_show", ("no-show", "no show", "noshow", "missed appointment", "appointment reminder", "cancellations"),
        "Appointment-based solo businesses (coaches, therapists, salons, consultants)",
        "Clients book and don't show up; reminders are sent by hand or not at all, and every no-show is lost revenue.",
        "n8n", "No-Show Recovery Autopilot", "Cut no-shows with smart reminders and 1-click rebooking.",
        "Calendly/Cal.com webhook -> reminders 24h and 2h before -> if no-show, rebooking email with link.", 39,
        (Step("webhook", "New Booking"), Step("set", "Normalize Booking", {"fields": ["name", "email", "start_time"]}),
         Step("wait", "Wait Until 24h Before", {"until": "={{ DateTime.fromISO($json.start_time).minus({hours: 24}).toISO() }}"}), Step("email", "Reminder 24h", {"subject": "See you tomorrow!"}),
         Step("wait", "Wait Until 2h Before", {"until": "={{ DateTime.fromISO($('Normalize Booking').item.json.start_time).minus({hours: 2}).toISO() }}"}), Step("email", "Reminder 2h", {"subject": "Starting in 2 hours"}),
         Step("http", "Check Attendance", {"url": "={{$env.BOOKING_API_URL}}/bookings"}),
         Step("if", "Was No-Show?", {"field": "status", "equals": "no_show"}),
         Step("email", "Rebooking Offer", {"subject": "We missed you, pick a new time",
                                                "text": "We missed you today! Pick a new time here: {{$env.REBOOK_LINK}}"})),
        (("BOOKING_API_URL", "Your booking tool API base URL (Cal.com / Calendly)"),
         ("SMTP_FROM", "Sender address for reminders"), ("REBOOK_LINK", "Your public booking link"))),
    Topic(
        "lead_capture", ("lead gen", "lead generation", "leads from", "capture leads", "lead magnet", "inbound leads", "crm"),
        "Freelancers and agencies with a contact form",
        "Form leads sit in an inbox for hours; hot leads go cold before anyone replies or logs them in a CRM.",
        "n8n", "AI Lead Capture & Instant Reply", "Every form lead logged, scored and answered in under 60 seconds.",
        "Form webhook -> score lead -> append to Google Sheets -> instant personalized reply -> Slack alert for hot leads.", 49,
        (Step("webhook", "Form Submission"),
         Step("code", "Score Lead", {"js": "const m=($json.message||'').toLowerCase();let s=20;['budget','asap','quote','pricing'].forEach(w=>{if(m.includes(w))s+=15});return [{json:{...$json,score:Math.min(s,100)}}];"}),
         Step("sheets", "Log to CRM Sheet"), Step("email", "Instant Reply", {"subject": "Got your message, here's what happens next"}),
         Step("if", "Hot Lead?", {"field": "score", "gte": 60}), Step("slack", "Alert Hot Lead"),
         Step("respond", "Respond OK")),
        (("GOOGLE_SHEET_ID", "Spreadsheet that acts as your lead CRM"), ("SLACK_CHANNEL", "Channel for hot-lead alerts"),
         ("SMTP_FROM", "Sender address for replies"))),
    Topic(
        "client_onboarding", ("client onboarding", "onboarding new clients", "onboard clients", "welcome packet", "intake form"),
        "Service freelancers and small agencies",
        "Every new client means the same 10 manual steps: contract, invoice, folder, welcome email, kickoff call.",
        "n8n", "Client Onboarding Autopilot", "From 'signed' to 'kickoff booked' without touching a thing.",
        "Payment/contract webhook -> create client folder -> welcome email with intake form -> kickoff link -> task in PM tool.", 59,
        (Step("webhook", "Deal Won"), Step("set", "Client Profile", {"fields": ["client_name", "email", "package"]}),
         Step("http", "Create Drive Folder", {"url": "https://www.googleapis.com/drive/v3/files"}),
         Step("email", "Welcome Email", {"subject": "Welcome aboard! Your next 3 steps"}),
         Step("http", "Create Kickoff Task", {"url": "={{$env.PM_WEBHOOK_URL}}"}),
         Step("wait", "Wait 2 Days", {"amount": 2, "unit": "days"}), Step("if", "Intake Done?", {"field": "intake_completed", "equals": True}),
         Step("email", "Intake Nudge", {"subject": "Quick reminder: your intake form"})),
        (("PM_WEBHOOK_URL", "Webhook of your project tool (Trello/ClickUp/Notion via n8n)"),
         ("INTAKE_FORM_URL", "Link to your intake form"), ("SMTP_FROM", "Sender address"))),
    Topic(
        "invoice_chasing", ("invoice", "late payment", "unpaid", "overdue", "chasing payments", "accounts receivable"),
        "Freelancers and consultants who invoice clients",
        "Chasing overdue invoices is awkward and manual; cash flow suffers because reminders go out late or never.",
        "python", "Invoice Chaser API", "Polite-to-firm payment reminders that run themselves.",
        "FastAPI service: daily endpoint reads open invoices, picks the right reminder stage, returns what to send.", 49,
        logic=_INVOICE_LOGIC),
    Topic(
        "review_replies", ("google reviews", "review replies", "reply to reviews", "reputation", "yelp", "trustpilot"),
        "Local businesses and e-commerce stores",
        "Reviews pile up unanswered; negative ones hurt the rating and nobody notices them in time.",
        "python", "Review Responder API", "Every review answered on-brand, angry ones escalated instantly.",
        "FastAPI webhook: classify review sentiment, draft reply, flag negative ones for a human.", 39,
        logic=_REVIEW_LOGIC),
    Topic(
        "lead_scoring", ("lead scoring", "qualify leads", "lead qualification", "which leads"),
        "B2B solopreneurs and small sales teams",
        "All leads look the same; time goes to tire-kickers while serious buyers wait.",
        "python", "Lead Scoring Microservice", "Know in one second which lead deserves your call.",
        "FastAPI webhook: score 0-100 from email domain, company and intent words; route hot leads.", 59,
        logic=_LEAD_LOGIC),
    Topic(
        "social_repurpose", ("repurpose content", "social media posting", "schedule posts", "content calendar", "linkedin posts"),
        "Creators, coaches and consultants who sell via content",
        "Posting consistently on 3 networks takes hours a week; good content is written once and never reused.",
        "n8n", "Content Repurposing Engine", "One blog post in, a week of social posts out.",
        "RSS/blog webhook -> AI rewrites into X thread, LinkedIn post and newsletter blurb -> queue in Google Sheets.", 45,
        (Step("schedule", "Every Morning"), Step("http", "Fetch Latest Post", {"url": "={{$env.RSS_URL}}"}),
         Step("code", "Split Into Ideas", {"js": "return ($json.items||[$json]).slice(0,1).map(i=>({json:{title:i.title,link:i.link}}));"}),
         Step("http", "Rewrite With AI", {"url": "https://api.groq.com/openai/v1/chat/completions"}),
         Step("sheets", "Add To Content Queue")),
        (("RSS_URL", "Feed of your blog/newsletter"), ("GROQ_API_KEY", "Free Groq key for rewriting"),
         ("GOOGLE_SHEET_ID", "Sheet used as the posting queue"))),
    Topic(
        "email_triage", ("inbox zero", "email overload", "too many emails", "sort emails", "email triage"),
        "Busy solo founders",
        "The inbox mixes sales, support and noise; important emails get buried for days.",
        "n8n", "AI Inbox Triage", "Your inbox, sorted and summarized before you open it.",
        "Gmail trigger -> AI labels (sales/support/billing/noise) -> daily digest -> urgent ones to Slack.", 39,
        (Step("schedule", "Every Hour", {"hours": 1}), Step("http", "Fetch Unread", {"url": "https://gmail.googleapis.com/gmail/v1/users/me/messages"}),
         Step("http", "Classify With AI", {"url": "https://api.groq.com/openai/v1/chat/completions"}),
         Step("if", "Urgent?", {"field": "label", "equals": "urgent"}), Step("slack", "Ping Me")),
        (("GROQ_API_KEY", "Free Groq key used for classification"), ("SLACK_CHANNEL", "Channel for urgent mail"))),
    Topic(
        "weekly_report", ("weekly report", "kpi report", "client reporting", "reporting dashboard", "monthly report"),
        "Agencies and freelancers reporting to clients",
        "Building the weekly client report by copy-pasting numbers takes hours that nobody pays for.",
        "n8n", "Auto Client Report", "Client KPIs collected and emailed every Monday, hands-free.",
        "Monday schedule -> pull metrics (GA4/Stripe/Ads APIs) -> format summary -> email to each client.", 49,
        (Step("schedule", "Every Week", {"hours": 168}), Step("http", "Pull Metrics", {"url": "={{$env.METRICS_API_URL}}"}),
         Step("code", "Build Summary", {"js": "const d=$json;return [{json:{summary:`Visits: ${d.visits||0} | Leads: ${d.leads||0} | Revenue: ${d.revenue||0}`}}];"}),
         Step("email", "Send Report", {"subject": "Your weekly results"})),
        (("METRICS_API_URL", "Endpoint that returns the client's metrics"), ("SMTP_FROM", "Sender address"))),
    Topic(
        "abandoned_cart", ("abandoned cart", "cart recovery", "checkout abandon", "shopify abandoned"),
        "Small e-commerce stores (Shopify, WooCommerce)",
        "Most carts are abandoned and the built-in recovery email is generic and sent once.",
        "n8n", "Cart Recovery Sequence", "A 3-touch recovery sequence that brings carts back.",
        "Store webhook -> wait 1h -> reminder -> wait 24h -> discount code -> stop when order placed.", 45,
        (Step("webhook", "Cart Abandoned"), Step("wait", "Wait 1 Hour"),
         Step("http", "Order Placed?", {"url": "={{$env.STORE_API_URL}}/orders"}),
         Step("if", "Still Abandoned?", {"field": "ordered", "equals": False}),
         Step("email", "Reminder", {"subject": "You left something behind"}), Step("wait", "Wait 24 Hours", {"amount": 24, "unit": "hours"}),
         Step("email", "Discount Offer", {"subject": "10% off to finish your order"})),
        (("STORE_API_URL", "Your store API base URL"), ("DISCOUNT_CODE", "Code offered in the last email"))),
    Topic(
        "testimonials", ("testimonial", "collect reviews", "case study", "social proof"),
        "Coaches, course creators and freelancers",
        "Happy clients would leave a testimonial, but nobody asks at the right moment.",
        "n8n", "Testimonial Collector", "Ask at the perfect moment, collect proof on autopilot.",
        "Project-completed webhook -> wait 3 days -> ask for testimonial -> save answers to a sheet.", 29,
        (Step("webhook", "Project Completed"), Step("wait", "Wait 3 Days", {"amount": 3, "unit": "days"}),
         Step("email", "Ask For Testimonial", {"subject": "Could you share 2 sentences about working together?"}),
         Step("sheets", "Save Testimonial")),
        (("TESTIMONIAL_FORM_URL", "Your testimonial form link"), ("GOOGLE_SHEET_ID", "Where testimonials are stored"))),
    Topic(
        "meeting_notes", ("meeting notes", "call summary", "transcript", "follow-up email after call"),
        "Consultants and sales-led solopreneurs",
        "After every call the follow-up email and CRM notes are written by hand, or forgotten.",
        "n8n", "Call-to-Follow-Up Autopilot", "Transcript in, summary + follow-up email + CRM note out.",
        "Recording webhook (Zoom/Fireflies) -> AI summary + action items -> draft follow-up email -> log in sheet.", 49,
        (Step("webhook", "Transcript Ready"), Step("http", "Summarize With AI", {"url": "https://api.groq.com/openai/v1/chat/completions"}),
         Step("email", "Follow-Up Draft", {"subject": "Recap + next steps from our call"}), Step("sheets", "Log Call Notes")),
        (("GROQ_API_KEY", "Free Groq key used for summaries"), ("GOOGLE_SHEET_ID", "Call log sheet"))),
)

BY_KEY = {t.key: t for t in TOPICS}

INTENT_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\bis there (a|any) (tool|app|way|template|workflow)", r"\bhow (do|can) (i|you|we) automate", r"\blooking for (a|an) ",
    r"\bi spend (hours|too much time)", r"\bmanually\b", r"\bwish there (was|were)", r"\bany (template|workflow|recommendation)s?\b",
    r"\banyone (automated|automate|built)", r"\bwould pay\b", r"\bsick of\b", r"\bpain\b", r"\bstruggl", r"\btedious\b",
    r"\btakes (me )?forever\b", r"\bwaste of time\b",
))


def classify(text: str) -> str:
    """Best matching topic key, or '' when the post matches no known pain."""
    low = text.lower()
    best, hits = "", 0
    for t in TOPICS:
        n = sum(low.count(k) for k in t.keywords)
        if n > hits:
            best, hits = t.key, n
    return best


def intent_score(text: str) -> float:
    return float(sum(1 for p in INTENT_PATTERNS if p.search(text)))
