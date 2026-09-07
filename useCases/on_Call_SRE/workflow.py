"""Build and run the on-call SRE incident triage squad."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from squadAI.createAgent import Agent
from squadAI.squadAgent import SquadAgents, SquadResult
from squadAI.task import Task

from useCases.on_Call_SRE.fixtures import DEFAULT_INCIDENT, DEFAULT_INCIDENT_ID, get_incident_profile
from useCases.on_Call_SRE.scoring import IncidentScore, score_incident_run
from useCases.on_Call_SRE.tools import (
    get_recent_changes,
    lookup_runbook,
    query_metrics,
    search_logs,
)
from useCases.on_Call_SRE.schemas import IncidentAssessment, InvestigationReport
from useCases.on_Call_SRE.validators import commander_gate_validator

Variant = Literal["baseline", "qa"]

TOOL_RULES = (
    "You MUST call your tools for every factual lookup — never invent timestamps, "
    "versions, metric values, or log lines. After tools return, summarize findings "
    "with explicit timestamps and values."
)

INVESTIGATOR_FRAMEWORK = (
    "Follow this triage method:\n"
    "1. Establish timeline — what changed first relative to the alert window?\n"
    "2. Separate symptoms (timeouts, latency) from likely causes (saturation, dependency failure).\n"
    "3. Flag noise — unrelated services, routine jobs, or transient blips that share keywords.\n"
    "4. State confidence (high/medium/low) for each finding."
)

COMMANDER_OUTPUT_FORMAT = """\
ROOT_CAUSE: <one sentence hypothesis; evidence-based, not speculative>
SEVERITY: SEV1|SEV2|SEV3|SEV4
INCIDENT_TYPE: DB_POOL_EXHAUSTION|TRAFFIC_SURGE|DEPENDENCY_FAILURE|UNKNOWN
CONFIDENCE: <high|medium|low — one sentence why>
EVIDENCE:
- <bullet citing metrics investigation>
- <bullet citing logs investigation>
- <bullet citing change/deploy investigation>
RED_HERRINGS:
- <coincidental signals ruled out or marked unconfirmed>
"""


@dataclass(frozen=True)
class IncidentTriageBundle:
    """Tasks and agents for scoring after a run."""

    squad: SquadAgents
    task_metrics: Task
    task_logs: Task
    task_changes: Task
    task_commander: Task
    task_qa: Task | None
    task_runbook: Task
    variant: Variant
    incident_id: str


def build_incident_squad(*, variant: Variant = "baseline") -> IncidentTriageBundle:
    """Wire agents and tasks for the incident triage pipeline."""
    metrics_agent = Agent(
        backstory=(
            f"You are a senior SRE specializing in observability and SLO analysis. {TOOL_RULES}\n\n"
            f"{INVESTIGATOR_FRAMEWORK}\n\n"
            "For latency alerts, always report: p50/p95/p99 vs baseline, spike start time, "
            "affected endpoints, error and timeout rates, RPS vs baseline, and resource saturation "
            "(CPU, memory, connection pools). Explicitly note whether traffic increased — "
            "flat RPS argues against a pure traffic surge."
        ),
        tools=[query_metrics],
        max_iterations=8,
    )
    logs_agent = Agent(
        backstory=(
            f"You are a senior SRE specializing in distributed systems log analysis. {TOOL_RULES}\n\n"
            f"{INVESTIGATOR_FRAMEWORK}\n\n"
            "Log streams include noise from other incidents and routine jobs. Filter chronologically "
            "around the alert window. Identify the earliest ERROR that plausibly explains downstream "
            "symptoms. Quote exact log lines. Distinguish primary errors from cascading timeouts."
        ),
        tools=[search_logs],
        max_iterations=8,
    )
    changes_agent = Agent(
        backstory=(
            f"You are a release engineer and change-management specialist. {TOOL_RULES}\n\n"
            f"{INVESTIGATOR_FRAMEWORK}\n\n"
            "Review deploys, config pushes, and feature-flag changes. Note timing relative to "
            "the alert but treat temporal proximity as correlation until logs/metrics prove causation. "
            "Summarize each change with version/id, timestamp, author, and kind."
        ),
        tools=[get_recent_changes],
        max_iterations=8,
    )
    commander_agent = Agent(
        backstory=(
            "You are an incident commander with 10+ years of production operations experience.\n\n"
            "Synthesize upstream investigations into a structured assessment:\n"
            "- Build a mental timeline: first failure → cascading symptoms.\n"
            "- Rank hypotheses; pick the best-supported root cause.\n"
            "- Assign severity by customer impact (SEV1=outage, SEV2=major degradation, "
            "SEV3=minor, SEV4=informational).\n"
            "- Classify INCIDENT_TYPE using evidence patterns, not hunches.\n"
            "- List RED_HERRINGS for signals that looked suspicious but lack supporting evidence.\n"
            "- Never claim a deploy caused an incident unless logs/metrics prove it — "
            "coincidental timing belongs in RED_HERRINGS.\n"
            "- State CONFIDENCE honestly when evidence is circumstantial."
        ),
        max_iterations=6,
    )
    qa_agent = Agent(
        backstory=(
            "You are an incident review lead. Your job is to enforce quality standards on "
            "commander assessments before runbook execution:\n"
            "- All required sections present and substantive.\n"
            "- Evidence cites metrics, logs, and changes — not unsupported assertions.\n"
            "- Severity and incident type align with cited evidence.\n"
            "- Deploy/version claims use cautious language unless causation is proven.\n"
            "- RED_HERRINGS populated when coincidental changes exist.\n"
            "Approve only assessments that a production on-call team could act on safely."
        ),
        max_iterations=4,
    )
    runbook_agent = Agent(
        backstory=(
            f"You are an operations executor. {TOOL_RULES}\n\n"
            "Read incident_type from the commander assessment JSON in <context>. "
            "Call lookup_runbook with the exact type string (e.g. DB_POOL_EXHAUSTION). "
            "Return the runbook steps verbatim — do not paraphrase or skip steps."
        ),
        tools=[lookup_runbook],
        max_iterations=6,
    )

    task_metrics = Task(
        task_description=(
            "Investigate metrics for service '{service}' in region '{region}' "
            "for alert at {alert_time} (symptom: {symptom}). "
            "Set source to \"metrics\" in the structured output."
        ),
        agent=metrics_agent,
        output_schema=InvestigationReport,
    )
    task_logs = Task(
        task_description=(
            "Search logs for '{service}' around alert time {alert_time}. "
            "Filter noise; identify the earliest ERROR that explains the symptom. "
            "Set source to \"logs\" in the structured output."
        ),
        agent=logs_agent,
        output_schema=InvestigationReport,
    )
    task_changes = Task(
        task_description=(
            "Review recent changes (deploys, config, feature flags) for '{service}' "
            "within 24h of alert {alert_time}. Note timing vs the alert — correlation only. "
            "Set source to \"changes\" in the structured output."
        ),
        agent=changes_agent,
        output_schema=InvestigationReport,
    )
    task_commander = Task(
        task_description=(
            "Produce a structured incident assessment for '{service}' "
            "(alert: {alert_time}, symptom: {symptom}). "
            "Use upstream JSON investigation reports as evidence sources."
        ),
        dependency=[task_metrics, task_logs, task_changes],
        agent=commander_agent,
        task_output=COMMANDER_OUTPUT_FORMAT,
        output_schema=IncidentAssessment,
    )

    task_qa: Task | None = None
    runbook_dependency: list[Task] = [task_commander]

    if variant == "qa":
        task_qa = Task(
            task_description=(
                "Review the commander assessment against on-call quality standards "
                "before runbook execution."
            ),
            dependency=[task_commander],
            validates=task_commander,
            validator=commander_gate_validator,
            max_retries=2,
            agent=qa_agent,
        )
        runbook_dependency = [task_qa]

    task_runbook = Task(
        task_description="Fetch and return runbook steps for the classified incident type.",
        dependency=runbook_dependency,
        agent=runbook_agent,
    )

    agents = [
        metrics_agent,
        logs_agent,
        changes_agent,
        commander_agent,
        runbook_agent,
    ]
    if task_qa is not None:
        agents.append(qa_agent)

    tasks = [
        task_metrics,
        task_logs,
        task_changes,
        task_commander,
        task_runbook,
    ]
    if task_qa is not None:
        tasks.insert(-1, task_qa)

    squad = SquadAgents(agents=agents, tasks=tasks)

    return IncidentTriageBundle(
        squad=squad,
        task_metrics=task_metrics,
        task_logs=task_logs,
        task_changes=task_changes,
        task_commander=task_commander,
        task_qa=task_qa,
        task_runbook=task_runbook,
        variant=variant,
        incident_id=DEFAULT_INCIDENT_ID,
    )


def run_incident_triage(
    *,
    variant: Variant = "baseline",
    include_usage_in_result: bool = False,
    incident_id: str | None = None,
    **incident_kwargs,
) -> tuple[SquadResult, IncidentScore, IncidentTriageBundle]:
    """Run the triage pipeline and return result, score, and task bundle."""
    profile = get_incident_profile(incident_id)
    incident = {
        "incident_id": incident_id or DEFAULT_INCIDENT_ID,
        "service": profile["service"],
        "region": profile["region"],
        "alert_time": profile["alert_time"],
        "symptom": profile["symptom"],
        **incident_kwargs,
    }
    bundle = build_incident_squad(variant=variant)

    result = bundle.squad.run(
        include_usage_in_result=include_usage_in_result,
        **{k: v for k, v in incident.items() if k != "incident_id"},
    )
    score = score_incident_run(
        result,
        incident_id=incident["incident_id"],
        task_metrics=bundle.task_metrics,
        task_logs=bundle.task_logs,
        task_commander=bundle.task_commander,
        task_runbook=bundle.task_runbook,
    )
    return result, score, bundle
