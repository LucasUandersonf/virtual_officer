"""Railway usage (optional): what the services actually consume, priced with Railway's public rates.

Rates (docs.railway.com, pricing): $20 per vCPU-month, $10 per GB of RAM per month, $0.05 per GB of egress;
volumes are configurable. The ``estimatedUsage`` query below follows Railway's public GraphQL API, which
is not fully documented: on any error the caller keeps the RAILWAY_*_USD estimates, so nothing breaks.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import Settings, secret

log = logging.getLogger(__name__)
API = "https://backboard.railway.com/graphql/v2"
MIN_PER_MONTH = 30 * 24 * 60
# usage is reported per minute: vCPU-minutes and GB-minutes; egress in GB
PRICE = {"CPU_USAGE": 20 / MIN_PER_MONTH, "MEMORY_USAGE_GB": 10 / MIN_PER_MONTH, "NETWORK_TX_GB": 0.05,
         "DISK_USAGE_GB": 0.15 / MIN_PER_MONTH}
QUERY = """query($projectId: String!, $measurements: [MetricMeasurement!]!) {
  estimatedUsage(projectId: $projectId, measurements: $measurements) { measurement estimatedValue }
}"""


async def estimated_month_usd(s: Settings, client: httpx.AsyncClient) -> dict[str, Any] | None:
    if not (s.railway_api_token and s.railway_project_id):
        return None
    try:
        r = await client.post(API, json={"query": QUERY, "variables": {"projectId": s.railway_project_id,
                                                                        "measurements": list(PRICE)}},
                              headers={"Authorization": f"Bearer {secret(s.railway_api_token)}"})
        data = r.json()
        rows = (data.get("data") or {}).get("estimatedUsage")
        if r.status_code != 200 or data.get("errors") or not rows:
            log.warning("railway usage unavailable: %s", str(data.get("errors") or r.status_code)[:200])
            return None
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("railway usage failed: %s", exc)
        return None
    by = {row["measurement"]: float(row.get("estimatedValue") or 0) for row in rows}
    cost = {k: round(v * PRICE.get(k, 0), 2) for k, v in by.items()}
    return {"estimated_month_usd": round(sum(cost.values()), 2), "by_measurement_usd": cost, "raw": by}
