"""Heuristic scoring for incident triage runs — driven by incident profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from squadAI.squadAgent import SquadResult
from squadAI.task import Task

from useCases.on_Call_SRE.fixtures import DEFAULT_INCIDENT_ID, get_incident_profile


@dataclass
class IncidentScore:
    """Scores for a single triage run (each dimension 0 = fail, 1 = partial, 2 = pass)."""

    metrics_signal: int = 0
    logs_signal: int = 0
    correct_incident_type: int = 0
    correct_severity: int = 0
    reasoning_hygiene: int = 0
    runbook_match: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return (
            self.metrics_signal
            + self.logs_signal
            + self.correct_incident_type
            + self.correct_severity
            + self.reasoning_hygiene
            + self.runbook_match
        )

    @property
    def max_score(self) -> int:
        return 12

    def as_dict(self) -> dict[str, int]:
        return {
            "metrics_signal": self.metrics_signal,
            "logs_signal": self.logs_signal,
            "correct_incident_type": self.correct_incident_type,
            "correct_severity": self.correct_severity,
            "reasoning_hygiene": self.reasoning_hygiene,
            "runbook_match": self.runbook_match,
            "total": self.total,
            "max_score": self.max_score,
        }


def score_incident_run(
    result: SquadResult,
    *,
    incident_id: str = DEFAULT_INCIDENT_ID,
    task_metrics: Task,
    task_logs: Task,
    task_commander: Task,
    task_runbook: Task,
) -> IncidentScore:
    """Score a triage run against the incident profile ground truth."""
    profile = get_incident_profile(incident_id)
    truth = profile["ground_truth"]

    metrics_out = result.get(task_metrics) or ""
    logs_out = result.get(task_logs) or ""
    commander_out = result.get(task_commander) or ""
    runbook_out = result.get(task_runbook) or ""

    score = IncidentScore()
    spike_time = truth["metrics_spike_start"]
    log_snippet = truth["primary_log_snippet"]
    incident_type = truth["incident_type"]
    severity = truth["severity"]
    runbook_marker = truth["runbook_marker"]
    red_herring_version = truth.get("red_herring_deploy_version")

    if spike_time in metrics_out:
        score.metrics_signal = 2
    elif "spike" in metrics_out.lower() or "p99" in metrics_out.lower():
        score.metrics_signal = 1

    if log_snippet in logs_out:
        score.logs_signal = 2
    elif log_snippet.lower() in logs_out.lower():
        score.logs_signal = 1

    type_field = f"INCIDENT_TYPE: {incident_type}"
    if type_field in commander_out or f"INCIDENT_TYPE:{incident_type}" in commander_out.replace(" ", ""):
        score.correct_incident_type = 2
    elif incident_type in commander_out:
        score.correct_incident_type = 1

    if severity in commander_out:
        score.correct_severity = 2
    elif "SEV" in commander_out:
        score.correct_severity = 1

    lower_cmd = commander_out.lower()
    bad_causation = (
        red_herring_version
        and "caused by" in lower_cmd
        and red_herring_version in commander_out
    )
    if not bad_causation:
        score.reasoning_hygiene = 2
    elif any(word in lower_cmd for word in ("red_herring", "correlation", "coincident", "unconfirmed")):
        score.reasoning_hygiene = 1

    if runbook_marker in runbook_out:
        score.runbook_match = 2
    elif incident_type in runbook_out:
        score.runbook_match = 1

    score.details = {
        "incident_id": incident_id,
        "commander_has_root_cause": "ROOT_CAUSE:" in commander_out,
        "commander_has_evidence": "EVIDENCE:" in commander_out,
        "commander_has_confidence": "CONFIDENCE:" in commander_out,
        "expected_incident_type": incident_type,
        "expected_severity": severity,
    }
    return score
