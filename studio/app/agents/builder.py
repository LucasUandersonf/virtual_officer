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


def readme(name: str, tagline: str, problem: str, kind: str, guide: str, env: list[tuple[str, str]]) -> str:
    env_rows = "\n".join(f"| `{k}` | {d} |" for k, d in env) or "| — | No variables needed |"
    if kind == "n8n":
        install = ("1. Open your n8n (cloud or self-hosted, v1.30+).\n2. **Workflows → Import from File** and pick `workflow.json` "
                   "(or run `./import.sh` on the n8n host).\n3. Create the credentials the nodes ask for (SMTP, Google, Slack).\n"
                   "4. Set the variables below in n8n (Settings → Variables) or in the n8n container environment. "
                   "Self-hosted n8n needs `N8N_BLOCK_ENV_ACCESS_IN_NODE=false` to read `$env`.\n5. Run once with test data, then **Activate**.")
    else:
        install = ("1. `cp .env.example .env` and fill in the values.\n2. Local: `./start.sh` (Python 3.11+).\n"
                   "3. Docker: `docker build -t svc service && docker run -p 8080:8080 --env-file .env svc`.\n"
                   "4. Railway/Render: deploy the `service/` folder; it reads `$PORT`.\n"
                   "5. Send events to `POST /webhook` with header `X-Webhook-Secret`. Health check: `GET /health`.\n"
                   "6. Tests: `cd service && pytest`.")
    return (f"# {name}\n\n> {tagline}\n\n## The problem it solves\n\n{problem}\n\n## Install (10 minutes)\n\n{install}\n\n"
            f"## Configuration\n\n| Variable | What it is |\n|---|---|\n{env_rows}\n\n## How it works\n\n{guide}\n\n"
            "## Support\n\nReply to your purchase email with any question. Updates are free for this version.\n")


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
            "Use ={{$env.NAME}} for every URL/ID the buyer must configure and list those in env. guide: 4-8 sentences.",
            WorkflowSpec, fallback, max_tokens=2500)
        wf = n8n.build_workflow(spec.name, slugify(spec.name), [s.model_dump() for s in spec.steps], spec.guide)
        qa = n8n.validate_workflow(wf)
        declared = {e.name: e.description for e in spec.env}
        env = [(k, declared.get(k, "Set this value (used by the workflow)")) for k in n8n.env_vars(wf)]
        env += [(k, d) for k, d in declared.items() if k not in dict(env)]
        qa["checks"].append({"check": "env vars documented", "ok": True, "detail": ", ".join(k for k, _ in env)})
        qa["provider"] = provider
        files = {**packaging.n8n_files(json.dumps(wf, indent=2)), "_guide": spec.guide}
        return files, spec.name, spec.tagline, env, qa

    async def _python(self, idea: Idea, t: playbook.Topic | None) -> tuple[dict[str, str], str, str, list[tuple[str, str]], dict[str, Any]]:
        def fallback() -> ServiceSpec:
            logic = t.logic if t and t.logic else "def process(payload: dict) -> dict:\n    return {'received': payload}\n"
            return ServiceSpec(name=t.product if t else idea.title, tagline=t.tagline if t else idea.mvp[:120], logic=logic,
                               sample_payload={"email": "jane@acme.io", "message": "Need a quote asap"},
                               env=[EnvVar(name="WEBHOOK_SECRET", description="Shared secret sent in X-Webhook-Secret")],
                               guide=f"{idea.mvp}\n\nAll business rules live in `service/logic.py`, in plain Python you can edit.")

        spec, provider = await self.ctx.llm.json(
            self.key, self.persona.system,
            f"Product idea: {idea.title}\nProblem: {idea.problem}\nMVP: {idea.mvp}\n\n"
            "Write logic.py for a FastAPI webhook service: pure Python standard library only, one top-level function "
            "`process(payload: dict) -> dict` with the business rules, docstring, no network calls, no file or OS access. "
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
        files = {**packaging.python_service_files(spec.name, spec.logic, repr(spec.sample_payload)), "_guide": spec.guide}
        return files, spec.name, spec.tagline, env, qa
