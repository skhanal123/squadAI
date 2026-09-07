"""Generic validation gates for incident commander assessments."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from squadAI.output_schema import is_json_task_output
from squadAI.validation import ValidationResult

from useCases.on_Call_SRE.fixtures import VALID_INCIDENT_TYPES, VALID_SEVERITIES
from useCases.on_Call_SRE.schemas import IncidentAssessment

REQUIRED_SECTIONS = (
    "ROOT_CAUSE:",
    "SEVERITY:",
    "INCIDENT_TYPE:",
    "EVIDENCE:",
    "RED_HERRINGS:",
    "CONFIDENCE:",
)

EVIDENCE_SOURCE_KEYWORDS = (
    "metric",
    "log",
    "deploy",
    "trace",
    "change",
    "config",
    "feature",
    "rps",
    "p99",
    "latency",
    "error",
    "pool",
    "version",
)

CAUTIOUS_LANGUAGE = (
    "correlation",
    "correlated",
    "possible",
    "may",
    "might",
    "unclear",
    "unconfirmed",
    "red_herring",
    "red herring",
    "coincident",
    "not proven",
    "cannot confirm",
)

STRONG_CAUSATION_PATTERNS = (
    re.compile(r"\bcaused by\b", re.IGNORECASE),
    re.compile(r"\broot cause is the deploy\b", re.IGNORECASE),
    re.compile(r"\bdue to the deploy\b", re.IGNORECASE),
)

VERSION_PATTERN = re.compile(r"\bv?\d+\.\d+\.\d+\b", re.IGNORECASE)
EVIDENCE_BULLET = re.compile(r"^\s*[-*]\s+\S", re.MULTILINE)


def parse_commander_output(raw: str) -> IncidentAssessment | None:
    """Parse structured JSON commander output, or return ``None`` for legacy prose."""
    if not is_json_task_output(raw):
        return None
    try:
        return IncidentAssessment.model_validate_json(raw)
    except (ValidationError, json.JSONDecodeError):
        return None


def _validate_structured_assessment(assessment: IncidentAssessment) -> ValidationResult:
    if len(assessment.evidence) < 2:
        return ValidationResult(
            approved=False,
            feedback="evidence must include at least two entries citing upstream investigations.",
        )

    evidence_block = "\n".join(f"- {item}" for item in assessment.evidence)
    if not _evidence_cites_sources(evidence_block):
        return ValidationResult(
            approved=False,
            feedback=(
                "Each evidence entry should cite a source type (metrics, logs, deploys, "
                "traces, or config changes) — not unsupported assertions."
            ),
        )

    legacy_text = assessment_to_legacy_text(assessment)
    causation_issue = _check_causation_vs_correlation(legacy_text)
    if causation_issue is not None:
        return causation_issue

    return ValidationResult(approved=True)


def assessment_to_legacy_text(assessment: IncidentAssessment) -> str:
    """Render structured assessment as legacy sectioned text for scoring/helpers."""
    evidence = "\n".join(f"- {item}" for item in assessment.evidence)
    red_herrings = "\n".join(f"- {item}" for item in assessment.red_herrings)
    return (
        f"ROOT_CAUSE: {assessment.root_cause}\n"
        f"SEVERITY: {assessment.severity}\n"
        f"INCIDENT_TYPE: {assessment.incident_type}\n"
        f"CONFIDENCE: {assessment.confidence}\n"
        f"EVIDENCE:\n{evidence}\n"
        f"RED_HERRINGS:\n{red_herrings}"
    )


def commander_output_text(raw: str) -> str:
    """Normalize commander output to legacy sectioned text when possible."""
    assessment = parse_commander_output(raw)
    if assessment is not None:
        return assessment_to_legacy_text(assessment)
    return raw


def _extract_field(text: str, field: str) -> str:
    """Return content after ``FIELD:`` until the next ALL-CAPS field or EOF."""
    marker = f"{field}:"
    start = text.find(marker)
    if start == -1:
        return ""

    start += len(marker)
    rest = text[start:]
    next_field = re.search(r"\n[A-Z_]+:", rest)
    if next_field:
        return rest[: next_field.start()].strip()
    return rest.strip()


def _missing_sections(text: str) -> list[str]:
    return [section for section in REQUIRED_SECTIONS if section not in text]


def _evidence_bullets(text: str) -> list[str]:
    evidence_block = _extract_field(text, "EVIDENCE")
    if not evidence_block:
        return []
    return EVIDENCE_BULLET.findall(evidence_block)


def _evidence_cites_sources(evidence_block: str) -> bool:
    lower = evidence_block.lower()
    return any(keyword in lower for keyword in EVIDENCE_SOURCE_KEYWORDS)


def _severity_valid(text: str) -> tuple[bool, str]:
    severity = _extract_field(text, "SEVERITY").upper()
    if not severity:
        return False, "SEVERITY section is empty."

    for level in VALID_SEVERITIES:
        if level in severity.split():
            return True, level

    token = severity.split()[0] if severity.split() else severity
    if token in VALID_SEVERITIES:
        return True, token

    return False, f"SEVERITY must be one of {', '.join(sorted(VALID_SEVERITIES))}."


def _incident_type_valid(text: str) -> tuple[bool, str]:
    incident_type = _extract_field(text, "INCIDENT_TYPE").upper().replace(" ", "_")
    if not incident_type:
        return False, "INCIDENT_TYPE section is empty."

    for known in VALID_INCIDENT_TYPES:
        if known in incident_type:
            return True, known

    return False, (
        f"INCIDENT_TYPE must be one of: {', '.join(sorted(VALID_INCIDENT_TYPES))}."
    )


def _confidence_valid(text: str) -> tuple[bool, str]:
    confidence = _extract_field(text, "CONFIDENCE").lower()
    if not confidence:
        return False, "CONFIDENCE section is empty."

    if any(word in confidence for word in ("high", "medium", "low")):
        return True, confidence

    return False, "CONFIDENCE must state high, medium, or low with brief justification."


def _check_causation_vs_correlation(text: str) -> ValidationResult | None:
    """Flag strong deploy/version causation without cautious language in ROOT_CAUSE."""
    root_cause = _extract_field(text, "ROOT_CAUSE")
    red_herrings = _extract_field(text, "RED_HERRINGS")
    lower_rc = root_cause.lower()

    mentions_version = bool(VERSION_PATTERN.search(root_cause))
    rc_has_cautious = any(phrase in lower_rc for phrase in CAUTIOUS_LANGUAGE)
    strong_causation = any(pattern.search(root_cause) for pattern in STRONG_CAUSATION_PATTERNS)

    if strong_causation and mentions_version and not rc_has_cautious:
        return ValidationResult(
            approved=False,
            feedback=(
                "ROOT_CAUSE uses strong causation language for a version/deploy without "
                "cautious wording (correlation, may, unconfirmed). Move unproven deploy "
                "links to RED_HERRINGS unless logs/metrics prove causation."
            ),
        )

    if mentions_version and not red_herrings.strip():
        return ValidationResult(
            approved=False,
            feedback=(
                "A version/deploy appears in ROOT_CAUSE but RED_HERRINGS is empty — "
                "note coincidental changes or explain why the deploy is causal."
            ),
        )

    if mentions_version and red_herrings.strip().lower() in {"- none", "- none identified", "none"}:
        rh_has_cautious = any(phrase in red_herrings.lower() for phrase in CAUTIOUS_LANGUAGE)
        if not rh_has_cautious and strong_causation:
            return ValidationResult(
                approved=False,
                feedback=(
                    "RED_HERRINGS dismisses coincidental deploys too briefly — name the "
                    "change and why it is correlation-only."
                ),
            )

    return None


def commander_gate_validator(
    critic_output: str,
    *,
    upstream_output: str,
) -> ValidationResult:
    """Validate commander output structure, evidence quality, and reasoning hygiene."""
    del critic_output  # gate validates upstream (commander) output

    assessment = parse_commander_output(upstream_output)
    if assessment is not None:
        return _validate_structured_assessment(assessment)

    missing = _missing_sections(upstream_output)
    if missing:
        return ValidationResult(
            approved=False,
            feedback=f"Missing required sections: {', '.join(missing)}",
        )

    root_cause = _extract_field(upstream_output, "ROOT_CAUSE")
    if len(root_cause) < 15:
        return ValidationResult(
            approved=False,
            feedback="ROOT_CAUSE must be a substantive, evidence-based sentence (not empty or vague).",
        )

    ok, severity_feedback = _severity_valid(upstream_output)
    if not ok:
        return ValidationResult(approved=False, feedback=severity_feedback)

    ok, type_feedback = _incident_type_valid(upstream_output)
    if not ok:
        return ValidationResult(approved=False, feedback=type_feedback)

    ok, confidence_feedback = _confidence_valid(upstream_output)
    if not ok:
        return ValidationResult(approved=False, feedback=confidence_feedback)

    bullets = _evidence_bullets(upstream_output)
    if len(bullets) < 2:
        return ValidationResult(
            approved=False,
            feedback="EVIDENCE must include at least two bullet points citing upstream investigations.",
        )

    evidence_block = _extract_field(upstream_output, "EVIDENCE")
    if not _evidence_cites_sources(evidence_block):
        return ValidationResult(
            approved=False,
            feedback=(
                "Each EVIDENCE bullet should cite a source type (metrics, logs, deploys, "
                "traces, or config changes) — not unsupported assertions."
            ),
        )

    causation_issue = _check_causation_vs_correlation(upstream_output)
    if causation_issue is not None:
        return causation_issue

    return ValidationResult(approved=True)
