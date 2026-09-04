"""Mock observability tools for incident triage."""

from squadAI.tools import tool_wrapper

from useCases.on_Call_SRE.fixtures import (
    CHANGE_HISTORY,
    LOG_STREAMS,
    METRICS_DATA,
    RUNBOOKS,
    VALID_INCIDENT_TYPES,
)


def _format_metrics_block(service: str, region: str, data: dict) -> str:
    latency = data["latency"]
    traffic = data.get("traffic", {})
    errors = data.get("errors", {})
    saturation = data.get("saturation", {})
    endpoints = data.get("affected_endpoints", [])
    notes = data.get("notes", [])

    lines = [
        f"Metrics ({data.get('window', '30m')} window) for {service} in {region}:",
        "",
        "Latency:",
    ]
    for key, values in latency.items():
        if isinstance(values, dict) and "baseline" in values:
            lines.append(f"  - {key}: current={values['current']} baseline={values['baseline']}")
        else:
            lines.append(f"  - {key}: {values}")

    if traffic:
        lines.extend(["", "Traffic:"])
        for key, values in traffic.items():
            if isinstance(values, dict):
                lines.append(f"  - {key}: current={values['current']} baseline={values['baseline']}")

    if errors:
        lines.extend(["", "Errors:"])
        for key, values in errors.items():
            if isinstance(values, dict):
                lines.append(f"  - {key}: current={values['current']} baseline={values['baseline']}")

    if saturation:
        lines.extend(["", "Saturation:"])
        for key, values in saturation.items():
            if isinstance(values, dict):
                lines.append(f"  - {key}: current={values['current']} baseline={values['baseline']}")

    if endpoints:
        lines.extend(["", "Affected endpoints:"])
        for ep in endpoints:
            lines.append(
                f"  - {ep['path']}: p99={ep['p99_ms']}ms errors={ep['error_pct']}%"
            )

    if notes:
        lines.extend(["", "Analyst notes:"])
        for note in notes:
            lines.append(f"  - {note}")

    return "\n".join(lines)


@tool_wrapper
def query_metrics(service: str, region: str, window_minutes: int = 30) -> str:
    """Query latency, traffic, error, and saturation metrics for a service in a region.

    Args:
        service: Service name (e.g. checkout-service).
        region: AWS region (e.g. us-east-1).
        window_minutes: Lookback window in minutes.
    """
    key = (service.strip(), region.strip())
    data = METRICS_DATA.get(key)
    if data is None:
        known = [f"{s}/{r}" for s, r in METRICS_DATA]
        raise ValueError(
            f"No metrics for service={service!r} region={region!r}. Known: {known}"
        )

    return _format_metrics_block(service, region, data)


@tool_wrapper
def search_logs(
    service: str,
    severity: str = "ERROR",
    limit: int = 20,
    include_info: bool = False,
) -> str:
    """Search recent logs for a service. Results are chronological and may include unrelated noise.

    Args:
        service: Service name.
        severity: Minimum log level filter — INFO, WARN, or ERROR.
        limit: Maximum number of log lines to return.
        include_info: When False, filters out routine INFO-only noise where possible.
    """
    key = service.strip()
    entries = LOG_STREAMS.get(key)
    if entries is None:
        raise ValueError(f"No logs for service={service!r}. Known: {list(LOG_STREAMS)}")

    level_rank = {"INFO": 0, "WARN": 1, "ERROR": 2}
    min_rank = level_rank.get(severity.strip().upper(), 2)

    filtered: list[dict[str, str]] = []
    for entry in entries:
        line = entry["line"]
        tags = entry.get("tags", "")
        for level, rank in level_rank.items():
            if f" {level} " in line:
                if rank < min_rank:
                    break
                if not include_info and tags == "routine" and level == "INFO":
                    break
                filtered.append(entry)
                break

    selected = filtered[-limit:] if limit else filtered
    if not selected:
        return f"No {severity}+ logs found for {service}."

    body_lines = []
    for entry in selected:
        tag_hint = f" [{entry.get('tags', 'untagged')}]" if entry.get("tags") else ""
        body_lines.append(f"- {entry['line']}{tag_hint}")

    return (
        f"Log search ({severity}+) for {service} — {len(selected)} lines, chronological:\n"
        + "\n".join(body_lines)
        + "\n\nNote: Streams may contain unrelated errors from other services or transient blips. "
        "Correlate timestamps with the alert window."
    )


@tool_wrapper
def get_recent_changes(service: str, hours: int = 24) -> str:
    """List recent deploys, config pushes, and feature-flag changes for a service.

    Args:
        service: Service name.
        hours: Lookback window in hours.
    """
    key = service.strip()
    changes = CHANGE_HISTORY.get(key)
    if changes is None:
        raise ValueError(
            f"No change history for service={service!r}. Known: {list(CHANGE_HISTORY)}"
        )

    lines = [
        f"Change history for {service} (last {hours}h):",
        "",
    ]
    for change in changes:
        lines.append(
            f"- [{change['kind'].upper()}] {change['version']} at {change['time']} "
            f"by {change['author']} — {change['summary']}"
        )

    lines.append(
        "\nNote: Temporal proximity to an alert does not prove causation — correlate with logs/metrics."
    )
    return "\n".join(lines)


@tool_wrapper
def lookup_runbook(incident_type: str) -> str:
    """Fetch runbook steps for an incident classification.

    Args:
        incident_type: One of DB_POOL_EXHAUSTION, TRAFFIC_SURGE, DEPENDENCY_FAILURE, UNKNOWN.
    """
    key = incident_type.strip().upper().replace(" ", "_")
    if key not in VALID_INCIDENT_TYPES:
        raise ValueError(
            f"Unknown incident_type={incident_type!r}. "
            f"Valid: {', '.join(sorted(VALID_INCIDENT_TYPES))}"
        )

    steps = RUNBOOKS[key]
    numbered = "\n".join(steps)
    return f"Runbook {key}:\n{numbered}"
