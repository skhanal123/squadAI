"""Structured output schemas for the on-call SRE incident triage workflow."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from squadAI.output_schema import TaskOutputBase

InvestigationSource = Literal["metrics", "logs", "changes"]
ConfidenceLevel = Literal["high", "medium", "low"]
SeverityLevel = Literal["SEV1", "SEV2", "SEV3", "SEV4"]
IncidentType = Literal[
    "DB_POOL_EXHAUSTION",
    "TRAFFIC_SURGE",
    "DEPENDENCY_FAILURE",
    "UNKNOWN",
]


class Finding(TaskOutputBase):
    """One evidence-backed finding from an investigation task."""

    summary: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    confidence: ConfidenceLevel


class InvestigationReport(TaskOutputBase):
    """Shared structured output for metrics, logs, and changes investigators."""

    source: InvestigationSource
    timeline: str = Field(min_length=1)
    findings: list[Finding] = Field(min_length=1)
    noise: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel


class IncidentAssessment(TaskOutputBase):
    """Structured incident commander assessment."""

    root_cause: str = Field(min_length=15)
    severity: SeverityLevel
    incident_type: IncidentType
    confidence: str = Field(min_length=1)
    evidence: list[str] = Field(min_length=2)
    red_herrings: list[str] = Field(default_factory=list)
