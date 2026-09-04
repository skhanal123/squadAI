"""Observability fixtures for on-call SRE triage — multi-incident noise + scenario profiles."""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Incident profiles (ground truth for scoring — not shown to agents)
# ---------------------------------------------------------------------------

INCIDENT_PROFILES: dict[str, dict[str, Any]] = {
    "checkout-latency-db-pool": {
        "service": "checkout-service",
        "region": "us-east-1",
        "alert_time": "2026-08-26T14:12:00Z",
        "symptom": "p99 latency > 2000ms (threshold 500ms)",
        "ground_truth": {
            "incident_type": "DB_POOL_EXHAUSTION",
            "severity": "SEV2",
            "metrics_spike_start": "2026-08-26T14:12:04Z",
            "primary_log_snippet": "ConnectionPool exhausted",
            "runbook_marker": "Confirm pool exhaustion",
            "red_herring_deploy_version": "v2.4.1",
        },
    },
}

DEFAULT_INCIDENT_ID = "checkout-latency-db-pool"
DEFAULT_INCIDENT = {
    **{k: v for k, v in INCIDENT_PROFILES[DEFAULT_INCIDENT_ID].items() if k != "ground_truth"},
    "incident_id": DEFAULT_INCIDENT_ID,
}


def get_incident_profile(incident_id: str | None = None) -> dict[str, Any]:
    """Return full profile including ground_truth for eval/scoring."""
    key = incident_id or DEFAULT_INCIDENT_ID
    if key not in INCIDENT_PROFILES:
        raise ValueError(
            f"Unknown incident_id={key!r}. Known: {list(INCIDENT_PROFILES)}"
        )
    return INCIDENT_PROFILES[key]


def list_incident_ids() -> list[str]:
    return list(INCIDENT_PROFILES)


# ---------------------------------------------------------------------------
# Metrics — primary signal + ambient noise from other services/time windows
# ---------------------------------------------------------------------------

METRICS_DATA: dict[tuple[str, str], dict[str, Any]] = {
    ("checkout-service", "us-east-1"): {
        "window": "30m",
        "latency": {
            "p50_ms": {"baseline": 45, "current": 890},
            "p95_ms": {"baseline": 120, "current": 2100},
            "p99_ms": {"baseline": 180, "current": 2840},
            "spike_start": "2026-08-26T14:12:04Z",
        },
        "traffic": {
            "rps": {"baseline": 415, "current": 420},
            "concurrent_sessions": {"baseline": 980, "current": 995},
        },
        "errors": {
            "rate_pct": {"baseline": 0.05, "current": 0.2},
            "timeout_pct": {"baseline": 0.01, "current": 0.15},
        },
        "saturation": {
            "cpu_pct": {"baseline": 38, "current": 52},
            "memory_pct": {"baseline": 61, "current": 64},
            "db_pool_utilization_pct": {"baseline": 45, "current": 100},
            "db_pool_waiting": {"baseline": 0, "current": 47},
        },
        "affected_endpoints": [
            {"path": "/api/checkout", "p99_ms": 2650, "error_pct": 0.1},
            {"path": "/api/payment", "p99_ms": 2840, "error_pct": 0.35},
            {"path": "/api/health", "p99_ms": 12, "error_pct": 0.0},
        ],
        "notes": [
            "Spike aligns with payment path latency; health endpoint unaffected.",
            "RPS flat — unlikely pure traffic surge.",
        ],
    },
    # Noise: different service/region — unrelated traffic surge from last week
    ("auth-service", "us-east-1"): {
        "window": "30m",
        "latency": {
            "p99_ms": {"baseline": 90, "current": 95},
        },
        "traffic": {
            "rps": {"baseline": 1200, "current": 3400},
        },
        "notes": ["Historical note: auth-service traffic surge on 2026-08-19 — resolved by autoscale."],
    },
}

# ---------------------------------------------------------------------------
# Logs — chronological stream with unrelated noise mixed in
# ---------------------------------------------------------------------------

LOG_STREAMS: dict[str, list[dict[str, str]]] = {
    "checkout-service": [
        # --- noise: earlier unrelated incidents / routine ops ---
        {
            "line": "2026-08-26T13:45:12Z INFO  cron/inventory-sync completed in 42s",
            "tags": "routine",
        },
        {
            "line": "2026-08-26T13:52:01Z WARN  auth-service upstream 401 spike (resolved — bad API key rotation in staging)",
            "tags": "noise-other-service",
        },
        {
            "line": "2026-08-26T14:02:30Z ERROR redis-cache timeout after 200ms key=session:abc123",
            "tags": "noise-transient",
        },
        {
            "line": "2026-08-26T14:08:00Z INFO  deployment-controller canary analysis passed for v2.4.0",
            "tags": "routine",
        },
        # --- incident window ---
        {
            "line": "2026-08-26T14:11:58Z INFO  payment-db health check ok pool=20 active=18",
            "tags": "incident",
        },
        {
            "line": "2026-08-26T14:12:05Z ERROR payment-db ConnectionPool exhausted (max=20, active=20, waiting=47)",
            "tags": "incident-primary",
        },
        {
            "line": "2026-08-26T14:12:06Z WARN  checkout-service retry attempt 3/3 for payment-db",
            "tags": "incident",
        },
        {
            "line": "2026-08-26T14:12:08Z ERROR /api/payment upstream timeout after 3000ms trace_id=8f3a2c",
            "tags": "incident",
        },
        {
            "line": "2026-08-26T14:12:10Z WARN  /api/checkout latency degraded p99=2650ms",
            "tags": "incident",
        },
        {
            "line": "2026-08-26T14:12:15Z INFO  autoscaler evaluation: CPU 52% — no scale action",
            "tags": "incident",
        },
        # --- noise: after incident starts ---
        {
            "line": "2026-08-26T14:15:00Z WARN  marketing-service/email-queue backlog (unrelated batch job)",
            "tags": "noise-other-service",
        },
    ],
    # Noise bucket: logs from a past unrelated incident (agents querying wrong service)
    "auth-service": [
        {
            "line": "2026-08-19T18:00:01Z ERROR rate limit exceeded client_id=mobile-app",
            "tags": "noise-historical",
        },
        {
            "line": "2026-08-19T18:00:05Z WARN  autoscale triggered +3 pods",
            "tags": "noise-historical",
        },
    ],
}

# Backward-compatible flat list for tools that iterate raw lines
LOG_DATA: dict[str, list[str]] = {
    service: [entry["line"] for entry in entries]
    for service, entries in LOG_STREAMS.items()
}

# ---------------------------------------------------------------------------
# Deploy / change history — includes rollbacks, config, feature flags
# ---------------------------------------------------------------------------

CHANGE_HISTORY: dict[str, list[dict[str, str]]] = {
    "checkout-service": [
        {
            "kind": "deploy",
            "version": "v2.4.0",
            "time": "2026-08-26T09:00:00Z",
            "author": "ci-bot",
            "summary": "Routine release — cart bugfix",
        },
        {
            "kind": "config",
            "version": "payment-db.pool.max=20",
            "time": "2026-08-26T11:30:00Z",
            "author": "platform-team",
            "summary": "Config push — unchanged pool size",
        },
        {
            "kind": "feature_flag",
            "version": "new-checkout-ui=10%",
            "time": "2026-08-26T13:00:00Z",
            "author": "growth-team",
            "summary": "Gradual rollout — low traffic slice",
        },
        {
            "kind": "deploy",
            "version": "v2.4.1",
            "time": "2026-08-26T14:00:00Z",
            "author": "ci-bot",
            "summary": "Hotfix — logging change in payment client",
        },
    ],
}

DEPLOY_DATA: dict[str, list[dict[str, str]]] = {
    service: [c for c in changes if c["kind"] == "deploy"]
    for service, changes in CHANGE_HISTORY.items()
}

# ---------------------------------------------------------------------------
# Runbooks
# ---------------------------------------------------------------------------

RUNBOOKS: dict[str, list[str]] = {
    "DB_POOL_EXHAUSTION": [
        "1. Confirm pool exhaustion in logs (ConnectionPool exhausted or pool waiting > 0).",
        "2. Check db_pool_utilization in metrics — values near 100% confirm saturation.",
        "3. Scale payment-db connection pool or restart pods.",
        "4. If deploy occurred within 30 minutes, treat as correlation until proven — consider rollback.",
        "5. Monitor p99 latency and error rate for 15 minutes after mitigation.",
    ],
    "TRAFFIC_SURGE": [
        "1. Confirm elevated RPS or concurrent sessions vs baseline in metrics.",
        "2. Verify CPU/memory saturation aligns with traffic increase.",
        "3. Scale service horizontally.",
        "4. Enable rate limiting or queue shedding if latency SLO still breached.",
    ],
    "DEPENDENCY_FAILURE": [
        "1. Identify failing upstream dependency from ERROR logs and trace IDs.",
        "2. Check whether dependency health checks failed before user-facing errors.",
        "3. Fail over to secondary region/instance if available.",
        "4. Engage dependency owner; preserve logs/metrics snapshot.",
    ],
    "UNKNOWN": [
        "1. Document timeline, blast radius, and hypotheses with confidence levels.",
        "2. Escalate to platform on-call.",
        "3. Preserve logs and metrics snapshots.",
        "4. Open a SEV bridge if customer impact is confirmed.",
    ],
}

VALID_INCIDENT_TYPES = frozenset(RUNBOOKS)
VALID_SEVERITIES = frozenset({"SEV1", "SEV2", "SEV3", "SEV4"})
