"""Agent 2 — Theo, Senior Dev & Builder: build the solution, QA it, package the .zip."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.agents import playbook
from app.agents.base import Agent
from app.models import Idea, Product
from app.services import n8n, packaging


class StepSpec(BaseModel):
    kind: str
    name: str = Field(max_length=60)
    params: dict[str, Any] = Field(default_factory=dict)


class EnvVar(BaseModel):
    name: str
    description: str


class WorkflowSpec(BaseModel):
    name: str
    tagline: str
    steps: list[StepSpec] = Field(min_length=2, max_length=14)
    env: list[EnvVar] = Field(default_factory=list)
    guide: str


class ServiceSpec(BaseModel):
    name: str
    tagline: str
    logic: str
    sample_payload: dict[str, Any]
    env: list[EnvVar] = Field(default_factory=list)
    guide: str


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "product"


def effort_hours(idea: Idea) -> float:
    t = playbook.BY_KEY.get(idea.topic)
    if idea.solution_type == "python":
        return 3.0
    return 0.5 * len(t.steps) if t and t.steps else 4.0


QUICKSTART = {
    "n8n": ("**Import** `src/workflow.json` in n8n (*Workflows → Import from File*).",
            "**Connect** the credentials flagged in red and add the variables from `.env.example`.",
            "**Activate.** Done."),
    "python": ("**Configure:** `cp .env.example .env` and set `WEBHOOK_SECRET`.",
               "**Run:** `./install.sh` (Docker) — or deploy the `src/` folder on Railway/Render.",
               "**Connect:** point your tool's webhook to `POST /webhook` with header `X-Webhook-Secret`."),
}
INSTALL_STEPS = {
    "n8n": ("In n8n (cloud or self-hosted, v1.30+): **Workflows → Import from File** → `src/workflow.json` "
            "(self-hosted shortcut: `./install.sh`).",
            "Open each node with a red warning and pick or create its credential (SMTP, Google, Slack).",
            "Add the variables from the table below in **Settings → Variables** (self-hosted: container env + "
            "`N8N_BLOCK_ENV_ACCESS_IN_NODE=false`).",
            "Click **Test workflow** with the sample data, check the result.",
            "Toggle **Active**. Duplicates and unauthenticated calls are filtered by the Guard node."),
    "python": ("`cp .env.example .env` and set `WEBHOOK_SECRET` (any long random string) plus the values below.",
               "Run `./install.sh` (builds and starts the Docker container), or deploy the `src/` folder on Railway/Render: "
               "it reads `$PORT` and idles at ~50 MB RAM.",
               "Check `GET /health` returns `{\"ok\": true}`.",
               "Point your tool's webhook to `POST /webhook` with header `X-Webhook-Secret`.",
               "Optional: `cd src && pip install -r requirements.txt && pytest` runs the included tests."),
}
LAYOUT = {
    "n8n": "src/workflow.json   the workflow to import\ninstall.sh          self-hosted import\n.env.example        every variable it reads\n"
           "README.md           this guide\nLICENSE.txt",
    "python": "src/main.py         API: auth, validation, idempotency, rate limit\nsrc/logic.py        your business rules\n"
              "src/test_*.py       tests\nsrc/Dockerfile      lean container\ninstall.sh          build + run\n"
              ".env.example        every variable it reads\nREADME.md           this guide\nLICENSE.txt",
}
TROUBLESHOOT = {
    "n8n": "- An execution failed? Open **Executions**: the red *Fail With Readable Error* step names the step that broke and what "
           "to check. Fix the credential/variable and click **Retry** — duplicates are ignored, so retrying is safe.\n"
           "- Calls to other services already retry 3 times with backoff before failing.",
    "python": "- `GET /health` shows `errors` and a plain-English `last_error`.\n- `401`: the `X-Webhook-Secret` header does not match "
              "`WEBHOOK_SECRET`. `422`: a field is missing or malformed. `429`: slow down (see `RATE_LIMIT_PER_MIN`).\n"
              "- Retrying a request with the same `Idempotency-Key` (or payload `id`) never processes it twice.",
}


def _numbered(items: tuple[str, ...]) -> str:
    return "\n".join(f"{i}. {step}" for i, step in enumerate(items, 1))


def readme(name: str, tagline: str, problem: str, kind: str, guide: str, env: list[tuple[str, str]]) -> str:
    env_rows = "\n".join(f"| `{k}` | {d} |" for k, d in env) or "| — | No variables needed |"
    safety = ("Built defensively: webhook calls are authenticated with `WEBHOOK_SECRET`, payloads are validated, duplicate "
              "deliveries are ignored (idempotency), and calls to other services retry with backoff. No secret is stored in "
              "the files: everything comes from your environment variables.")
    return (f"# {name}\n\n> {tagline}\n\n## 3-Step Quickstart\n\n{_numbered(QUICKSTART[kind])}\n\n"
            f"## The problem it solves\n\n{problem}\n\n"
            f"## Step-by-step installation in under 5 minutes\n\n{_numbered(INSTALL_STEPS[kind])}\n\n"
            f"## Configuration\n\n| Variable | What it is |\n|---|---|\n{env_rows}\n\n## How it works\n\n{guide}\n\n"
            f"## What's in the package\n\n```\n{LAYOUT[kind]}\n```\n\n"
            f"## If something goes wrong\n\n{TROUBLESHOOT[kind]}\n\n"
            f"## Reliability & security\n\n{safety}\n\n"
            "## Support\n\nReply to your purchase email with any question. Updates are free for this version.\n")


def _section_steps(text: str, heading: str) -> int:
    part = text.split(f"## {heading}", 1)[-1].split("\n## ", 1)[0] if f"## {heading}" in text else ""
    return sum(1 for line in part.splitlines() if line[:1].isdigit() and ". " in line[:4])


def check_ttv(readme_text: str, env: list[tuple[str, str]]) -> dict[str, object]:
    """Time-to-value / DX gate: 3-step quickstart, <= 5 detailed install steps, <= 5 variables to fill in."""
    quick = _section_steps(readme_text, "3-Step Quickstart")
    steps = _section_steps(readme_text, "Step-by-step installation in under 5 minutes")
    return {"check": "time-to-value: 3-step quickstart, <= 5 steps, <= 5 variables", "ok": quick == 3 and 0 < steps <= 5 and len(env) <= 5,
            "detail": f"quickstart {quick}, {steps} steps, {len(env)} variables"}


def check_env_documented(used: list[str], env: list[tuple[str, str]]) -> dict[str, object]:
    missing = sorted(set(used) - {k for k, _ in env})
    return {"check": ".env.example documents every variable the code reads", "ok": not missing, "detail": ", ".join(missing)}


class Builder(Agent):
    key = "builder"

    async def build(self, idea: Idea) -> Product:
        await self.status("working", f"Building {idea.title}")
        t = playbook.BY_KEY.get(idea.topic)
        if idea.solution_type == "n8n":
            files, name, tagline, env, qa = await self._n8n(idea, t)
        else:
            files, name, tagline, env, qa = await self._python(idea, t)
        name, tagline = name[:200], tagline[:300]  # LLM output: keep within the column sizes
        slug = slugify(name)
        async with self.ctx.db.session() as ses:
            dup = await ses.scalar(select(func.count(Product.id)).where(Product.slug.like(f"{slug}%")))
        if dup:
            slug = f"{slug}-{dup + 1}"
        files["README.md"] = readme(name, tagline, idea.problem, idea.solution_type, files.pop("_guide"), env)
        qa["checks"].append(check_ttv(files["README.md"], env))
        qa["ok"] = all(c["ok"] for c in qa["checks"])
        files[".env.example"] = packaging.env_example(env)
        pkg = packaging.build_zip(self.ctx.settings.artifacts_dir, slug, files)
        status = "built" if qa["ok"] else "qa_failed"
        async with self.ctx.db.session() as ses:
            product = Product(idea_id=idea.id, name=name, slug=slug, tagline=tagline, solution_type=idea.solution_type,
                              price_cents=idea.price_cents, status=status, artifact_path=pkg.path, artifact_sha256=pkg.sha256,
                              size_bytes=pkg.size, files=pkg.files, qa=qa)
            ses.add(product)
            db_idea = await ses.get(Idea, idea.id)
            if db_idea:
                db_idea.status = "built" if qa["ok"] else "failed"
        passed = sum(c["ok"] for c in qa["checks"])
        await self.log("build", f"{name}: {len(pkg.files)} files, {pkg.size / 1024:.1f} KB, QA {passed}/{len(qa['checks'])}",
                       product_id=product.id, provider=qa.get("provider"))
        await self.status("idle", f"Shipped {name} to QA" if qa["ok"] else f"{name} failed QA")
        return product

    async def _n8n(self, idea: Idea, t: playbook.Topic | None) -> tuple[dict[str, str], str, str, list[tuple[str, str]], dict[str, Any]]:
        def fallback() -> WorkflowSpec:
            steps = [StepSpec(kind=s.kind, name=s.name, params=dict(s.params)) for s in (t.steps if t else ())] or [
                StepSpec(kind="webhook", name="Trigger"), StepSpec(kind="respond", name="Respond")]
            return WorkflowSpec(name=t.product if t else idea.title, tagline=t.tagline if t else idea.mvp[:120], steps=steps,
                                env=[EnvVar(name=k, description=d) for k, d in (t.env if t else ())],
                                guide=f"{idea.mvp}\n\nEach node is named after what it does; open them in order to follow the flow.")

        spec, provider = await self.ctx.llm.json(
            self.key, self.persona.system,
            f"Product idea: {idea.title}\nProblem: {idea.problem}\nMVP: {idea.mvp}\n\n"
            f"Design an n8n workflow as an ordered list of steps. Allowed kinds: {sorted(n8n.KINDS)}. The first step must be "
            "'webhook' or 'schedule'. params by kind: http {method,url}, set {fields:[...]}, if {field, equals|gte}, "
            "code {js: n8n Code node JavaScript returning items}, email {subject,text}, wait {amount,unit}, sheets {sheet}. "
            "Use ={{$env.NAME}} for every URL/ID the buyer must configure and list those in env (max 4: setup must take under "
            "5 minutes). Never put a key or token in params. Code nodes must handle missing fields (optional chaining, defaults). "
            "A guard node (auth, validation, idempotency) and retries on external calls are added for you. guide: 4-8 sentences.",
            WorkflowSpec, fallback, max_tokens=2500)
        wf = n8n.build_workflow(spec.name, slugify(spec.name), [s.model_dump() for s in spec.steps], spec.guide)
        qa = n8n.validate_workflow(wf)
        declared = {e.name: e.description for e in spec.env}
        declared.setdefault("WEBHOOK_SECRET", "Shared secret your tool sends in the X-Webhook-Secret header (empty = no auth check)")
        env = [(k, declared.get(k, "Set this value (used by the workflow)")) for k in n8n.env_vars(wf)]
        env += [(k, d) for k, d in declared.items() if k not in dict(env)]
        qa["checks"].append(check_env_documented(n8n.env_vars(wf), env))
        scope = len([n for n in wf["nodes"] if n["type"] != "n8n-nodes-base.stickyNote"])
        qa["checks"].append({"check": "MVP scope: one job, no feature creep", "ok": len(spec.steps) <= 14, "detail": f"{scope} nodes"})
        qa["provider"] = provider
        files = {**packaging.n8n_files(json.dumps(wf, indent=2)), "_guide": spec.guide}
        return files, spec.name, spec.tagline, env, qa

    async def _python(self, idea: Idea, t: playbook.Topic | None) -> tuple[dict[str, str], str, str, list[tuple[str, str]], dict[str, Any]]:
        def fallback() -> ServiceSpec:
            logic = t.logic if t and t.logic else "def process(payload: dict) -> dict:\n    return {'received': payload}\n"
            return ServiceSpec(name=t.product if t else idea.title, tagline=t.tagline if t else idea.mvp[:120], logic=logic,
                               sample_payload={"email": "jane@acme.io", "message": "Need a quote asap"},
                               env=[EnvVar(name="WEBHOOK_SECRET", description="Shared secret sent in X-Webhook-Secret")],
                               guide=f"{idea.mvp}\n\nAll business rules live in `src/logic.py`, in plain Python you can edit.")

        spec, provider = await self.ctx.llm.json(
            self.key, self.persona.system,
            f"Product idea: {idea.title}\nProblem: {idea.problem}\nMVP: {idea.mvp}\n\n"
            "Write logic.py for a FastAPI webhook service: pure Python standard library only, one top-level function "
            "`process(payload: dict) -> dict` with the business rules, docstring, no network calls, no file or OS access, no "
            "eval/exec, no secrets. Validate inputs defensively: use payload.get with defaults, coerce types, and raise "
            "ValueError with a clear message for unusable input (the API turns it into HTTP 422). Same input -> same output "
            "(idempotent, no side effects). "
            "Give a realistic sample_payload for a test. guide: 4-8 sentences.",
            ServiceSpec, fallback, max_tokens=2500, temperature=0.3)
        qa = packaging.check_python(spec.logic)
        if not qa["ok"] and provider != "playbook":
            spec, provider = fallback(), "playbook"  # never ship code that failed the gate
            qa = packaging.check_python(spec.logic)
        qa["provider"] = provider
        env = [(e.name, e.description) for e in spec.env]
        if "WEBHOOK_SECRET" not in dict(env):
            env.insert(0, ("WEBHOOK_SECRET", "Shared secret sent in X-Webhook-Secret"))
        files = {**packaging.python_service_files(spec.name, spec.logic, repr(spec.sample_payload), slugify(spec.name)), "_guide": spec.guide}
        # RATE_LIMIT_PER_MIN is optional (default 120) and documented in the README troubleshooting section
        used = [v for v in packaging.python_env_vars(files["src/main.py"], spec.logic) if v != "RATE_LIMIT_PER_MIN"]
        for v in used:
            if v not in dict(env):
                env.append((v, "Read by src/logic.py: set this value"))
        qa["checks"].append(check_env_documented(used, env))
        qa["ok"] = all(c["ok"] for c in qa["checks"])
        return files, spec.name, spec.tagline, env, qa
