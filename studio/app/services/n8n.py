"""Build importable n8n workflow JSON from a short step spec, and check it before it is sold.

The LLM (or the playbook) only chooses *which* steps and their names/params. The JSON itself is
assembled here from known node templates with real typeVersions, so every workflow imports cleanly
(``n8n import:workflow --input=workflow.json`` or paste into the editor).
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from app.services.packaging import find_secrets

TRIGGERS = {"webhook", "schedule"}
EXTERNAL = {"http", "email", "slack", "sheets"}  # nodes that call another system: retried with backoff
RETRY = {"retryOnFail": True, "maxTries": 3, "waitBetweenTries": 2000}
GUARD_NAME = "Guard: Auth, Validate, Dedupe"
ERROR_SET, ERROR_STOP = "Explain Failure", "Fail With Readable Error"
# Self-healing: external nodes retry 3x with backoff; if a third-party API is still down, the item goes to an
# error branch that writes a plain-English diagnosis and fails the execution visibly (other items keep going),
# so the buyer sees *what* broke and *what to check* in the Executions list instead of a stack trace.
ERROR_MESSAGE = ("={{ 'Step \"' + $json.failed_step + '\" could not complete: ' + ($json.error?.message || $json.error || 'unknown error') + "
                 "'. Check its credential and the variables in .env.example, then retry this execution. Other items were processed.' }}")

# Runs right after the webhook. Defensive by design: rejects calls without the shared secret (when
# WEBHOOK_SECRET is set), drops payloads missing required fields, and drops deliveries already seen
# (idempotency key = id/event_id or the payload itself, kept in workflow static data), so a provider
# retrying a webhook never sends a second email or creates a second record.
GUARD_JS = """const REQUIRED = %s;
const store = $getWorkflowStaticData('global');
store.seen = store.seen || [];
const out = [];
$input.all().forEach((item, i) => {
  const headers = item.json.headers || {};
  const secret = $env.WEBHOOK_SECRET;
  if (secret && headers['x-webhook-secret'] !== secret) return;
  const body = item.json.body || {};
  const missing = REQUIRED.filter((f) => body[f] === undefined || body[f] === null || body[f] === '');
  if (missing.length) return;
  const key = String(body.id || body.event_id || body.uid || JSON.stringify(body));
  if (store.seen.includes(key)) return;
  store.seen.push(key);
  if (store.seen.length > 1000) store.seen.shift();
  out.push({ json: { ...item.json, idempotency_key: key }, pairedItem: i });
});
return out;"""
KINDS = TRIGGERS | {"http", "set", "if", "code", "email", "slack", "sheets", "wait", "respond"}
ENV_RE = re.compile(r"\$env\.([A-Z][A-Z0-9_]*)")


def _uid() -> str:
    return str(uuid.uuid4())


def _node(kind: str, name: str, params: dict[str, Any], slug: str, has_respond: bool, src: str = "$json") -> dict[str, Any]:
    """``src`` is the expression that holds the customer record (e.g. ``$('Normalize Booking').item.json``)."""
    p = params or {}
    if kind == "webhook":
        return {"type": "n8n-nodes-base.webhook", "typeVersion": 2, "webhookId": _uid(), "parameters": {
            "httpMethod": "POST", "path": slug, "responseMode": "responseNode" if has_respond else "onReceived", "options": {}}}
    if kind == "schedule":
        return {"type": "n8n-nodes-base.scheduleTrigger", "typeVersion": 1.2, "parameters": {
            "rule": {"interval": [{"field": "hours", "hoursInterval": int(p.get("hours", 24))}]}}}
    if kind == "http":
        return {"type": "n8n-nodes-base.httpRequest", "typeVersion": 4.2, "parameters": {
            "method": str(p.get("method", "GET")).upper(), "url": str(p.get("url", "={{$env.API_URL}}")), "options": {"timeout": 15000}}}
    if kind == "set":
        fields = p.get("fields") or ["email"]
        return {"type": "n8n-nodes-base.set", "typeVersion": 3.4, "parameters": {"assignments": {"assignments": [
            {"id": _uid(), "name": str(f), "value": f"={{{{ {src}.{f} }}}}", "type": "string"} for f in fields]}, "options": {}}}
    if kind == "if":
        field = str(p.get("field", "status"))
        if "gte" in p:
            op, right = {"type": "number", "operation": "gte"}, p["gte"]
        elif isinstance(p.get("equals"), bool):
            op, right = {"type": "boolean", "operation": "true" if p["equals"] else "false", "singleValue": True}, ""
        else:
            op, right = {"type": "string", "operation": "equals"}, str(p.get("equals", ""))
        return {"type": "n8n-nodes-base.if", "typeVersion": 2, "parameters": {"conditions": {
            "options": {"caseSensitive": True, "leftValue": "", "typeValidation": "loose"},
            "conditions": [{"id": _uid(), "leftValue": f"={{{{ $json.{field} }}}}", "rightValue": right, "operator": op}],
            "combinator": "and"}, "options": {}}}
    if kind == "code":
        return {"type": "n8n-nodes-base.code", "typeVersion": 2, "parameters": {
            "jsCode": str(p.get("js") or "return $input.all();")}}
    if kind == "email":
        return {"type": "n8n-nodes-base.emailSend", "typeVersion": 2.1, "parameters": {
            "fromEmail": "={{$env.SMTP_FROM}}", "toEmail": f"={{{{ {src}.email }}}}", "subject": str(p.get("subject", name)),
            "emailFormat": "text",
            "text": "=" + str(p.get("text") or f"Hi {{{{ {src}.name || 'there' }}}},\n\n{p.get('subject', name)}\n").lstrip("="),
            "options": {}}}
    if kind == "slack":
        return {"type": "n8n-nodes-base.slack", "typeVersion": 2.2, "parameters": {
            "select": "channel", "channelId": {"__rl": True, "value": "={{$env.SLACK_CHANNEL}}", "mode": "id"},
            "text": str(p.get("text") or f"{name}: {{{{ JSON.stringify($json) }}}}"), "otherOptions": {}}}
    if kind == "sheets":
        return {"type": "n8n-nodes-base.googleSheets", "typeVersion": 4.5, "parameters": {
            "operation": "append", "documentId": {"__rl": True, "value": "={{$env.GOOGLE_SHEET_ID}}", "mode": "id"},
            "sheetName": {"__rl": True, "value": str(p.get("sheet", "Sheet1")), "mode": "name"},
            "columns": {"mappingMode": "autoMapInputData", "value": {}}, "options": {}}}
    if kind == "wait":
        if p.get("until"):  # n8n expression giving an ISO date, e.g. 24 h before the appointment
            return {"type": "n8n-nodes-base.wait", "typeVersion": 1.1, "webhookId": _uid(), "parameters": {
                "resume": "specificTime", "dateTime": str(p["until"])}}
        return {"type": "n8n-nodes-base.wait", "typeVersion": 1.1, "webhookId": _uid(), "parameters": {
            "amount": float(p.get("amount", 1)), "unit": str(p.get("unit", "hours"))}}
    if kind == "respond":
        return {"type": "n8n-nodes-base.respondToWebhook", "typeVersion": 1.1, "parameters": {
            "respondWith": "json", "responseBody": '={ "ok": true }', "options": {}}}
    raise ValueError(f"unknown step kind {kind}")


def build_workflow(name: str, slug: str, steps: list[dict[str, Any]], guide: str) -> dict[str, Any]:
    steps = [s for s in steps if s.get("kind") in KINDS]
    if not steps or steps[0]["kind"] not in TRIGGERS:
        steps.insert(0, {"kind": "webhook", "name": "Trigger"})
    # Webhooks are acknowledged on receipt (fast 200, providers stop retrying); the guard then filters.
    steps = [steps[0]] + [s for s in steps[1:] if s["kind"] not in TRIGGERS | {"respond"}]
    has_respond = False
    if steps[0]["kind"] == "webhook":
        sets = [s for s in steps if s["kind"] == "set"]
        required = list((sets[0].get("params") or {}).get("fields") or ["email"]) if sets else ["email"]
        steps.insert(1, {"kind": "code", "name": GUARD_NAME, "params": {"js": GUARD_JS % json.dumps(required)}})
    nodes: list[dict[str, Any]] = [{
        "id": _uid(), "name": "Setup Guide", "type": "n8n-nodes-base.stickyNote", "typeVersion": 1,
        "position": [-260, -260], "parameters": {"content": guide[:2000], "height": 300, "width": 420}}]
    seen: set[str] = set()
    names: list[str] = []
    for i, st in enumerate(steps):
        nm = str(st.get("name") or st["kind"].title())[:60]
        while nm in seen:
            nm += " 2"
        seen.add(nm)
        # Where the customer record lives: the webhook body until a set/code node reshapes it, then that node.
        if i == 0:
            src = "$json.body" if st["kind"] == "webhook" else "$json"
        node = _node(st["kind"], nm, dict(st.get("params") or {}), slug, has_respond, src)
        nodes.append({"id": _uid(), "name": nm, "position": [i * 240, 0], **node, **(RETRY if st["kind"] in EXTERNAL else {})})
        names.append(nm)
        if nm == GUARD_NAME:
            src = f"$('{nm}').item.json.body"
        elif st["kind"] in ("set", "code"):
            src = f"$('{nm}').item.json"
    connections: dict[str, Any] = {a: {"main": [[{"node": b, "type": "main", "index": 0}]]} for a, b in zip(names, names[1:], strict=False)}
    external = [n for n in nodes if n.get("retryOnFail")]
    if external:
        y = 260
        nodes.append({"id": _uid(), "name": ERROR_SET, "type": "n8n-nodes-base.set", "typeVersion": 3.4, "position": [480, y],
                      "parameters": {"assignments": {"assignments": [
                          {"id": _uid(), "name": "failed_step", "value": "={{ $prevNode.name }}", "type": "string"}]},
                          "includeOtherFields": True, "options": {}}})
        nodes.append({"id": _uid(), "name": ERROR_STOP, "type": "n8n-nodes-base.stopAndError", "typeVersion": 1, "position": [720, y],
                      "parameters": {"errorMessage": ERROR_MESSAGE}})
        connections[ERROR_SET] = {"main": [[{"node": ERROR_STOP, "type": "main", "index": 0}]]}
        for n in external:
            n["onError"] = "continueErrorOutput"
            out = connections.setdefault(n["name"], {"main": [[]]})["main"]
            out.append([{"node": ERROR_SET, "type": "main", "index": 0}])
    return {"name": name, "nodes": nodes, "connections": connections, "active": False,
            "settings": {"executionOrder": "v1"}, "pinData": {}, "meta": {"templateCredsSetupCompleted": False}}


def env_vars(workflow: dict[str, Any]) -> list[str]:
    return sorted(set(ENV_RE.findall(json.dumps(workflow))))


def validate_workflow(wf: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    nodes = wf.get("nodes") or []
    names = [n.get("name") for n in nodes]
    check("json round-trip", json.loads(json.dumps(wf)) == wf)
    check("unique node names", len(names) == len(set(names)))
    check("has trigger", any(n.get("type") in ("n8n-nodes-base.webhook", "n8n-nodes-base.scheduleTrigger") for n in nodes))
    check("every node typed", all(n.get("type", "").startswith("n8n-nodes-base.") and n.get("typeVersion") for n in nodes))
    targets = [c["node"] for v in wf.get("connections", {}).values() for out in v.get("main", []) for c in out]
    check("connections resolve", all(t in names for t in targets) and all(k in names for k in wf.get("connections", {})))
    triggers = ("n8n-nodes-base.webhook", "n8n-nodes-base.scheduleTrigger", "n8n-nodes-base.stickyNote")
    orphans = [n["name"] for n in nodes if n.get("type") not in triggers and n["name"] not in targets]
    check("all steps connected", not orphans, ", ".join(orphans))
    ext_nodes = [n for n in nodes if n.get("retryOnFail")]
    check("self-healing: failures go to a readable error branch",
          all(n.get("onError") == "continueErrorOutput" and len(wf["connections"].get(n["name"], {}).get("main", [])) == 2
              for n in ext_nodes), f"{len(ext_nodes)} external nodes")
    codes = [n["parameters"].get("jsCode", "") for n in nodes if n.get("type") == "n8n-nodes-base.code"]
    check("code nodes balanced", all(c.count("{") == c.count("}") and c.count("(") == c.count(")") for c in codes))
    has_webhook = any(n.get("type") == "n8n-nodes-base.webhook" for n in nodes)
    check("webhook guarded (auth + validation + idempotency)", not has_webhook or GUARD_NAME in names)
    ext = [n for n in nodes if n.get("type") in ("n8n-nodes-base.httpRequest", "n8n-nodes-base.emailSend",
                                                  "n8n-nodes-base.slack", "n8n-nodes-base.googleSheets")]
    check("external calls retry with backoff", all(n.get("retryOnFail") and n.get("maxTries", 0) >= 2 for n in ext), f"{len(ext)} nodes")
    leaks = find_secrets(json.dumps(wf))
    check("no hard-coded secrets", not leaks, ", ".join(leaks))
    return {"ok": all(c["ok"] for c in checks), "checks": checks}
