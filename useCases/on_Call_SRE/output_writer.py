"""Write incident-focused triage results to useCases/on_Call_SRE/output/."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from squadAI.squadAgent import SquadResult

from useCases.on_Call_SRE.workflow import IncidentTriageBundle

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def _run_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _run_directory_name(service: str, stamp: str | None = None) -> str:
    stamp = stamp or _run_stamp()
    safe_service = re.sub(r"[^\w.-]", "-", service.strip().lower())
    return f"{stamp}_{safe_service}"


def build_incident_payload(
    *,
    result: SquadResult,
    bundle: IncidentTriageBundle,
    incident: dict[str, Any],
) -> dict[str, Any]:
    """Build JSON payload containing only incident-related content."""
    return {
        "incident": {
            "service": incident["service"],
            "region": incident["region"],
            "alert_time": incident["alert_time"],
            "symptom": incident["symptom"],
        },
        "investigations": {
            "metrics": result.get(bundle.task_metrics) or "",
            "logs": result.get(bundle.task_logs) or "",
            "changes": result.get(bundle.task_changes) or "",
        },
        "assessment": result.get(bundle.task_commander) or "",
        "runbook": result.final or result.get(bundle.task_runbook) or "",
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    incident = payload["incident"]
    investigations = payload["investigations"]
    lines = [
        "# Incident Triage Report",
        "",
        "## Alert",
        "",
        f"- **Service:** {incident['service']}",
        f"- **Region:** {incident['region']}",
        f"- **Alert time:** {incident['alert_time']}",
        f"- **Symptom:** {incident['symptom']}",
        "",
        "## Metrics investigation",
        "",
        investigations["metrics"] or "(no findings)",
        "",
        "## Logs investigation",
        "",
        investigations["logs"] or "(no findings)",
        "",
        "## Change history investigation",
        "",
        investigations["changes"] or "(no findings)",
        "",
        "## Incident assessment",
        "",
        payload["assessment"] or "(no assessment)",
        "",
        "## Recommended actions (runbook)",
        "",
        payload["runbook"] or "(no runbook)",
        "",
    ]
    return "\n".join(lines)


def write_triage_output(
    *,
    result: SquadResult,
    bundle: IncidentTriageBundle,
    incident: dict[str, Any],
    output_dir: Path | None = None,
    run_id: str | None = None,
) -> Path:
    """Persist incident files directly under ``output_dir``.

    Writes:
        - ``incident_report.md`` — human-readable incident summary
        - ``incident.json`` — structured incident data only
        - ``execution_trace.json`` — ReAct steps, tool calls, validation retries
        - ``investigations/`` — metrics, logs, changes findings
        - ``assessment.txt`` — commander assessment
        - ``runbook.txt`` — recommended actions

    Args:
        output_dir: Destination directory (created if needed). When omitted,
            uses ``on_Call_SRE/output/``.
        run_id: Deprecated; kept for tests. When set, writes under
            ``output_dir / run_id`` instead of ``output_dir`` directly.

    Returns:
        Path to the directory containing the incident files.
    """
    base_dir = output_dir or DEFAULT_OUTPUT_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = base_dir / run_id if run_id else base_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = build_incident_payload(result=result, bundle=bundle, incident=incident)

    (run_dir / "incident.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    result.write_trace(run_dir)
    (run_dir / "incident_report.md").write_text(
        _render_markdown(payload),
        encoding="utf-8",
    )
    (run_dir / "assessment.txt").write_text(payload["assessment"], encoding="utf-8")
    (run_dir / "runbook.txt").write_text(payload["runbook"], encoding="utf-8")

    investigations_dir = run_dir / "investigations"
    investigations_dir.mkdir(exist_ok=True)
    for name, content in payload["investigations"].items():
        (investigations_dir / f"{name}.txt").write_text(content, encoding="utf-8")

    return run_dir
