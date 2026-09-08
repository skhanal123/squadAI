"""Offline tests for on-call SRE scenario (no LLM API key required)."""

import json
import unittest

from squadAI.createAgent import Agent
from squadAI.squadAgent import SquadResult, TaskResult
from squadAI.task import Task
from squadAI.usage import TaskUsage, TokenUsage

from useCases.on_Call_SRE.scoring import score_incident_run
from useCases.on_Call_SRE.schemas import IncidentAssessment
from useCases.on_Call_SRE.validators import (
    assessment_to_legacy_text,
    commander_gate_validator,
)
from useCases.on_Call_SRE.workflow import build_incident_squad


def _valid_commander_json(**overrides) -> str:
    base = IncidentAssessment(
        root_cause="payment-db connection pool saturation caused upstream payment timeouts",
        severity="SEV2",
        incident_type="DB_POOL_EXHAUSTION",
        confidence="high — pool exhaustion log precedes payment timeouts",
        evidence=[
            "metrics: p99 spiked at 14:12:04 with flat RPS",
            "logs: ConnectionPool exhausted at 14:12:05",
            "changes: v2.4.1 deploy at 14:00 noted for correlation",
        ],
        red_herrings=[
            "v2.4.1 deploy timing is correlation only, not proven causation",
        ],
    )
    payload = base.model_dump()
    payload.update(overrides)
    return json.dumps(IncidentAssessment.model_validate(payload).model_dump())


def _valid_commander_output(**overrides: str) -> str:
    base = {
        "ROOT_CAUSE": "payment-db connection pool saturation caused upstream payment timeouts",
        "SEVERITY": "SEV2",
        "INCIDENT_TYPE": "DB_POOL_EXHAUSTION",
        "CONFIDENCE": "high — pool exhaustion log precedes payment timeouts",
        "EVIDENCE": (
            "- metrics: p99 spiked at 14:12:04 with flat RPS\n"
            "- logs: ConnectionPool exhausted at 14:12:05\n"
            "- changes: v2.4.1 deploy at 14:00 noted for correlation"
        ),
        "RED_HERRINGS": "- v2.4.1 deploy timing is correlation only, not proven causation",
    }
    base.update(overrides)
    return "\n".join(f"{key}: {value}" for key, value in base.items())


class TestCommanderValidator(unittest.TestCase):
    def test_rejects_missing_sections(self):
        result = commander_gate_validator(
            "review",
            upstream_output="ROOT_CAUSE: db issue",
        )
        self.assertFalse(result.approved)
        self.assertIn("Missing required sections", result.feedback)

    def test_rejects_deploy_causation_without_caveat(self):
        output = _valid_commander_output(
            ROOT_CAUSE="caused by v2.4.1 deploy rolling out bad code",
            RED_HERRINGS="- none",
        )
        result = commander_gate_validator("review", upstream_output=output)
        self.assertFalse(result.approved)

    def test_rejects_insufficient_evidence(self):
        output = _valid_commander_output(EVIDENCE="- only one bullet about logs")
        result = commander_gate_validator("review", upstream_output=output)
        self.assertFalse(result.approved)

    def test_accepts_complete_evidence_based_output(self):
        result = commander_gate_validator(
            "review",
            upstream_output=_valid_commander_output(),
        )
        self.assertTrue(result.approved)

    def test_accepts_traffic_surge_incident_type(self):
        output = _valid_commander_output(
            ROOT_CAUSE="RPS doubled causing CPU saturation and latency breach",
            INCIDENT_TYPE="TRAFFIC_SURGE",
            EVIDENCE=(
                "- metrics: RPS 3400 vs baseline 1200\n"
                "- logs: autoscale triggered\n"
                "- changes: no deploy in window"
            ),
            RED_HERRINGS="- none identified",
        )
        result = commander_gate_validator("review", upstream_output=output)
        self.assertTrue(result.approved)


    def test_accepts_structured_json_output(self):
        result = commander_gate_validator(
            "review",
            upstream_output=_valid_commander_json(),
        )
        self.assertTrue(result.approved)

    def test_structured_output_renders_legacy_text(self):
        legacy = assessment_to_legacy_text(IncidentAssessment.model_validate_json(_valid_commander_json()))
        self.assertIn("ROOT_CAUSE:", legacy)
        self.assertIn("INCIDENT_TYPE: DB_POOL_EXHAUSTION", legacy)


class TestIncidentScoring(unittest.TestCase):
    def setUp(self):
        agent = Agent(backstory="test")
        self.task_metrics = Task(task_description="metrics", agent=agent)
        self.task_logs = Task(task_description="logs", agent=agent)
        self.task_commander = Task(task_description="commander", agent=agent)
        self.task_runbook = Task(task_description="runbook", agent=agent)

    def test_perfect_score(self):
        result = SquadResult(
            task_results=[
                TaskResult(
                    task_id=self.task_metrics.id,
                    description="metrics",
                    output="spike at 2026-08-26T14:12:04Z p99 elevated",
                    usage=TaskUsage.from_tokens("mock", TokenUsage()),
                ),
                TaskResult(
                    task_id=self.task_logs.id,
                    description="logs",
                    output="ConnectionPool exhausted (max=20)",
                    usage=TaskUsage.from_tokens("mock", TokenUsage()),
                ),
                TaskResult(
                    task_id=self.task_commander.id,
                    description="commander",
                    output=_valid_commander_output(),
                    usage=TaskUsage.from_tokens("mock", TokenUsage()),
                ),
                TaskResult(
                    task_id=self.task_runbook.id,
                    description="runbook",
                    output="1. Confirm pool exhaustion in logs",
                    usage=TaskUsage.from_tokens("mock", TokenUsage()),
                ),
            ],
            final="runbook steps",
        )

        score = score_incident_run(
            result,
            task_metrics=self.task_metrics,
            task_logs=self.task_logs,
            task_commander=self.task_commander,
            task_runbook=self.task_runbook,
        )
        self.assertEqual(score.total, score.max_score)


class TestWorkflowBuild(unittest.TestCase):
    def test_baseline_squad_has_five_tasks(self):
        bundle = build_incident_squad(variant="baseline")
        self.assertEqual(len(bundle.squad.tasks), 5)
        self.assertIsNone(bundle.task_qa)

    def test_qa_squad_has_six_tasks(self):
        bundle = build_incident_squad(variant="qa")
        self.assertEqual(len(bundle.squad.tasks), 6)
        self.assertIsNotNone(bundle.task_qa)
        self.assertIsNone(bundle.task_qa.agent)

    def test_baseline_and_qa_construct_without_error(self):
        build_incident_squad(variant="baseline")
        build_incident_squad(variant="qa")

    def test_tasks_use_structured_output_schemas(self):
        bundle = build_incident_squad(variant="baseline")
        self.assertIsNotNone(bundle.task_metrics.output_schema)
        self.assertIsNotNone(bundle.task_commander.output_schema)


if __name__ == "__main__":
    unittest.main()
