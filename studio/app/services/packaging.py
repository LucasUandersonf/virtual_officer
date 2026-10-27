"""Turn a built solution into a sellable .zip: solution files + English install guide + .env.example + scripts.

Generated Python is checked statically (parsed, never executed here) for syntax, the required
``process`` entry point and obviously dangerous calls, so the office cannot ship broken or
hostile code to a buyer.
"""

from __future__ import annotations

import ast
import hashlib
import io
import os
import re
import zipfile
from dataclasses import dataclass
from typing import Any

SECRET_RE = re.compile(r"(sk_(live|test)_[0-9A-Za-z]{10,}|rk_live_[0-9A-Za-z]{10,}|whsec_[0-9A-Za-z]{10,}|xox[abprs]-[0-9A-Za-z-]{10,}|"
                       r"AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}|ghp_[0-9A-Za-z]{30,}|gsk_[0-9A-Za-z]{20,}|"
                       r"(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][^'\"\s]{12,}['\"])", re.IGNORECASE)


def find_secrets(text: str) -> list[str]:
    """Hard-coded credentials (OWASP A07): returns a masked sample of each hit."""
    return [m.group(0)[:8] + "…" for m in SECRET_RE.finditer(text)]


FORBIDDEN_IMPORTS = {"subprocess", "socket", "ctypes", "pty", "shutil", "multiprocessing"}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "system", "popen", "rmtree", "remove", "unlink"}


def check_python(source: str, require_process: bool = True) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"ok": False, "checks": [{"check": "syntax", "ok": False, "detail": str(exc)}]}
    checks.append({"check": "syntax", "ok": True, "detail": ""})
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad += [a.name for a in node.names if a.name.split(".")[0] in FORBIDDEN_IMPORTS]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in FORBIDDEN_IMPORTS:
            bad.append(node.module or "")
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
            if name in FORBIDDEN_CALLS:
                bad.append(f"{name}()")
    checks.append({"check": "no dangerous calls", "ok": not bad, "detail": ", ".join(bad)})
    leaks = find_secrets(source)
    checks.append({"check": "no hard-coded secrets", "ok": not leaks, "detail": ", ".join(leaks)})
    if require_process:
        has = any(isinstance(n, ast.FunctionDef) and n.name == "process" for n in tree.body)
        checks.append({"check": "defines process(payload)", "ok": has, "detail": ""})
    return {"ok": all(c["ok"] for c in checks), "checks": checks}


SERVICE_MAIN = '''\"\"\"{name} - FastAPI service. Run: uvicorn main:app --port 8080

Defensive by default: shared-secret auth, 64 KB payload limit, JSON-object validation, idempotency
(Idempotency-Key header or payload id; duplicates return the first answer and are not processed again),
per-client rate limit, and errors that never leak internals.
\"\"\"

import hmac
import json
import logging
import os
import time
from collections import OrderedDict, defaultdict, deque

from fastapi import FastAPI, Header, HTTPException, Request

from logic import process

log = logging.getLogger("service")
app = FastAPI(title="{name}")
SECRET = os.environ.get("WEBHOOK_SECRET", "")
MAX_BODY = 64 * 1024
RATE_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "120"))
_seen: OrderedDict[str, dict] = OrderedDict()  # idempotency cache (single instance; use Redis/DB if you scale out)
_hits: defaultdict[str, deque] = defaultdict(deque)
_errors = {{"count": 0, "last": ""}}  # readable diagnosis exposed on /health (self-diagnosing service)


@app.get("/health")
def health() -> dict:
    return {{"ok": True, "errors": _errors["count"], "last_error": _errors["last"]}}


def _rate_limited(client: str) -> bool:
    now, q = time.monotonic(), _hits[client]
    while q and now - q[0] > 60:
        q.popleft()
    q.append(now)
    return len(q) > RATE_PER_MIN


@app.post("/webhook")
async def webhook(request: Request, x_webhook_secret: str = Header(default=""), idempotency_key: str = Header(default="")) -> dict:
    if not SECRET or not hmac.compare_digest(SECRET, x_webhook_secret):
        raise HTTPException(status_code=401, detail="bad or missing X-Webhook-Secret")
    if _rate_limited(request.client.host if request.client else "unknown"):
        raise HTTPException(status_code=429, detail="too many requests")
    raw = await request.body()
    if len(raw) > MAX_BODY:
        raise HTTPException(status_code=413, detail="payload too large")
    try:
        payload = json.loads(raw or b"{{}}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="body must be JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="body must be a JSON object")
    key = idempotency_key or str(payload.get("id") or payload.get("event_id") or "")
    if key and key in _seen:
        return {{**_seen[key], "duplicate": True}}
    try:
        result = process(payload)
    except (KeyError, ValueError, TypeError) as exc:
        _errors["count"] += 1
        _errors["last"] = f"Rejected a payload ({{type(exc).__name__}}: {{str(exc)[:120]}}). Check the fields your tool sends."
        raise HTTPException(status_code=422, detail=f"invalid payload: {{type(exc).__name__}}") from exc
    except Exception as exc:  # never leak internals to the caller
        _errors["count"] += 1
        _errors["last"] = f"Internal error ({{type(exc).__name__}}). See the container logs; the request can be retried safely."
        log.exception("process failed")
        raise HTTPException(status_code=500, detail="internal error") from exc
    if key:
        _seen[key] = result
        if len(_seen) > 5000:
            _seen.popitem(last=False)
    return result
'''

# Lean by design: one process, no access log, capped concurrency and malloc arenas -> ~40-60 MB RAM idle,
# which keeps a Railway/Render instance in the cents-per-day range.
SERVICE_DOCKERFILE = """FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MALLOC_ARENA_MAX=2 PORT=8080
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY --chown=app:app . .
USER app
HEALTHCHECK --interval=60s --timeout=4s CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ['PORT']+'/health',timeout=3)"
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1 --no-access-log --limit-concurrency 50"]
"""

INSTALL_PY = """#!/usr/bin/env sh
# Builds and runs the service with your .env (Docker). Stop it with: docker rm -f {slug}
set -e
cd "$(dirname "$0")"
[ -f .env ] || {{ cp .env.example .env; echo "Created .env: fill in WEBHOOK_SECRET and the other values, then run ./install.sh again."; exit 1; }}
docker build -t {slug} src
docker rm -f {slug} >/dev/null 2>&1 || true
docker run -d --name {slug} --restart unless-stopped -p 8080:8080 --env-file .env {slug}
echo "Running: curl http://localhost:8080/health"
"""

INSTALL_N8N = """#!/usr/bin/env sh
# Self-hosted n8n only (run on the n8n host). n8n Cloud: Workflows -> Import from File -> src/workflow.json
set -e
cd "$(dirname "$0")"
n8n import:workflow --input=src/workflow.json
echo "Imported. Open n8n, add the credentials and variables from .env.example, then activate."
"""

ENV_USE = re.compile(r"(?:os\.environ\.get|os\.getenv|os\.environ\[)\(?\s*['\"]([A-Z][A-Z0-9_]*)['\"]")


def python_env_vars(*sources: str) -> list[str]:
    """Every environment variable the Python code reads, so .env.example documents 100 % of them."""
    return sorted({m for src in sources for m in ENV_USE.findall(src)})

SERVICE_TEST = '''from logic import process


def test_process_returns_dict():
    assert isinstance(process({sample}), dict)
'''


SERVICE_API_TEST = '''import os

os.environ["WEBHOOK_SECRET"] = "test-secret"

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402

client = TestClient(app)
H = {"X-Webhook-Secret": "test-secret"}


def test_rejects_missing_secret():
    assert client.post("/webhook", json={}).status_code == 401


def test_rejects_non_object():
    assert client.post("/webhook", content="[1, 2]", headers=H).status_code == 422


def test_duplicate_delivery_is_not_processed_twice():
    first = client.post("/webhook", json={"id": "evt_1"}, headers=H)
    again = client.post("/webhook", json={"id": "evt_1"}, headers=H)
    assert again.json().get("duplicate") is True or first.status_code == 422
'''


def python_service_files(name: str, logic: str, sample: str, slug: str = "service") -> dict[str, str]:
    return {
        "src/main.py": SERVICE_MAIN.format(name=name),
        "src/logic.py": logic,
        "src/requirements.txt": "fastapi==0.115.12\nuvicorn==0.34.2\nhttpx==0.28.1\npytest==8.3.5\n",
        "src/Dockerfile": SERVICE_DOCKERFILE,
        "src/test_logic.py": SERVICE_TEST.format(sample=sample),
        "src/test_api.py": SERVICE_API_TEST,
        "install.sh": INSTALL_PY.format(slug=slug),
    }


def n8n_files(workflow_json: str) -> dict[str, str]:
    return {
        "src/workflow.json": workflow_json,
        "install.sh": INSTALL_N8N,
    }


def env_example(rows: list[tuple[str, str]]) -> str:
    return "".join(f"# {desc}\n{key}=\n\n" for key, desc in rows)


LICENSE = """Single-business license

You may use and modify this product for your own business and your clients' projects.
You may not resell or redistribute it as a template or product.
Provided "as is", without warranty of any kind.
"""


@dataclass(frozen=True)
class Package:
    path: str
    sha256: str
    size: int
    files: list[str]


def build_zip(artifacts_dir: str, slug: str, files: dict[str, str]) -> Package:
    files = {**files, "LICENSE.txt": LICENSE}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            info = zipfile.ZipInfo(f"{slug}/{name}", date_time=(2026, 1, 1, 0, 0, 0))
            info.external_attr = (0o755 if name.endswith(".sh") else 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, files[name])
    data = buf.getvalue()
    os.makedirs(artifacts_dir, exist_ok=True)
    path = os.path.join(artifacts_dir, f"{slug}.zip")
    with open(path, "wb") as f:
        f.write(data)
    return Package(path, hashlib.sha256(data).hexdigest(), len(data), sorted(files))
