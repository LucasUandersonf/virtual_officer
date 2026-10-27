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
import zipfile
from dataclasses import dataclass
from typing import Any

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
    if require_process:
        has = any(isinstance(n, ast.FunctionDef) and n.name == "process" for n in tree.body)
        checks.append({"check": "defines process(payload)", "ok": has, "detail": ""})
    return {"ok": all(c["ok"] for c in checks), "checks": checks}


SERVICE_MAIN = '''"""{name} - FastAPI service. Run: uvicorn main:app --port 8080"""

import hmac
import os

from fastapi import FastAPI, Header, HTTPException

from logic import process

app = FastAPI(title="{name}")
SECRET = os.environ.get("WEBHOOK_SECRET", "")


@app.get("/health")
def health() -> dict:
    return {{"ok": True}}


@app.post("/webhook")
def webhook(payload: dict, x_webhook_secret: str = Header(default="")) -> dict:
    if SECRET and not hmac.compare_digest(SECRET, x_webhook_secret):
        raise HTTPException(status_code=401, detail="bad secret")
    return process(payload)
'''

SERVICE_DOCKERFILE = """FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
"""

SERVICE_TEST = '''from logic import process


def test_process_returns_dict():
    assert isinstance(process({sample}), dict)
'''


def python_service_files(name: str, logic: str, sample: str) -> dict[str, str]:
    return {
        "service/main.py": SERVICE_MAIN.format(name=name),
        "service/logic.py": logic,
        "service/requirements.txt": "fastapi==0.115.12\nuvicorn==0.34.2\npytest==8.3.5\n",
        "service/Dockerfile": SERVICE_DOCKERFILE,
        "service/test_logic.py": SERVICE_TEST.format(sample=sample),
        "start.sh": "#!/usr/bin/env sh\ncd service && pip install -r requirements.txt && uvicorn main:app --port 8080\n",
    }


def n8n_files(workflow_json: str) -> dict[str, str]:
    return {
        "workflow.json": workflow_json,
        "import.sh": "#!/usr/bin/env sh\n# Imports the workflow into a self-hosted n8n (run on the n8n host)\nn8n import:workflow --input=workflow.json\n",
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
