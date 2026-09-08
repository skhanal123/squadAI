"""Tests for incident output writer."""

import json
import tempfile
import unittest
from pathlib import Path

from squadAI.createAgent import Agent
from squadAI.providers.base import LLMResponse
from squadAI.providers.mock import MockProvider
from squadAI.squadAgent import SquadAgents, SquadResult, TaskResult
from squadAI.usage import TaskUsage, TokenUsage

from useCases.on_Call_SRE.output_writer import write_triage_output
from useCases.on_Call_SRE.workflow import build_incident_squad


class TestTriageOutputWriter(unittest.TestCase):
    def test_writes_incident_only_files(self):
        bundle = build_incident_squad(variant="baseline")

        result = SquadResult(
            task_results=[
                TaskResult(
                    task_id=bundle.task_metrics.id,
                    description="Investigate metrics",
                    output="p99 spiked at 14:12:04",
                    usage=TaskUsage.from_tokens("mock", TokenUsage()),
                ),
                TaskResult(
                    task_id=bundle.task_logs.id,
                    description="Search logs",
                    output="ConnectionPool exhausted",
                    usage=TaskUsage.from_tokens("mock", TokenUsage()),
                ),
                TaskResult(
                    task_id=bundle.task_changes.id,
                    description="Review changes",
                    output="v2.4.1 deployed at 14:00",
                    usage=TaskUsage.from_tokens("mock", TokenUsage()),
                ),
                TaskResult(
                    task_id=bundle.task_commander.id,
                    description="Commander assessment",
                    output="ROOT_CAUSE: pool exhausted\nSEVERITY: SEV2",
                    usage=TaskUsage.from_tokens("mock", TokenUsage()),
                ),
                TaskResult(
                    task_id=bundle.task_runbook.id,
                    description="Runbook",
                    output="1. Confirm pool exhaustion",
                    usage=TaskUsage.from_tokens("mock", TokenUsage()),
                ),
            ],
            final="1. Confirm pool exhaustion",
        )

        incident = {
            "service": "checkout-service",
            "region": "us-east-1",
            "alert_time": "2026-08-26T14:12:00Z",
            "symptom": "p99 latency > 2000ms",
        }

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = write_triage_output(
                result=result,
                bundle=bundle,
                incident=incident,
                output_dir=Path(tmp),
                run_id="test_checkout_service",
            )

            self.assertTrue((run_dir / "incident_report.md").is_file())
            self.assertTrue((run_dir / "incident.json").is_file())
            self.assertTrue((run_dir / "execution_trace.json").is_file())
            self.assertTrue((run_dir / "assessment.txt").is_file())
            self.assertTrue((run_dir / "runbook.txt").is_file())
            self.assertTrue((run_dir / "investigations" / "metrics.txt").is_file())

            payload = json.loads((run_dir / "incident.json").read_text(encoding="utf-8"))
            self.assertEqual(set(payload.keys()), {"incident", "investigations", "assessment", "runbook"})
            self.assertNotIn("score", payload)
            self.assertNotIn("usage", payload)
            self.assertNotIn("pipeline", payload)
            self.assertEqual(payload["investigations"]["metrics"], "p99 spiked at 14:12:04")
            self.assertEqual(payload["assessment"], "ROOT_CAUSE: pool exhausted\nSEVERITY: SEV2")

            trace_payload = json.loads(
                (run_dir / "execution_trace.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(trace_payload["tasks"]), 5)
            self.assertEqual(trace_payload["tasks"][0]["description"], "Investigate metrics")
            self.assertIn("trace", trace_payload["tasks"][0])

            report = (run_dir / "incident_report.md").read_text(encoding="utf-8")
            self.assertIn("## Alert", report)
            self.assertIn("## Metrics investigation", report)
            self.assertNotIn("Evaluation score", report)
            self.assertNotIn("Token usage", report)


if __name__ == "__main__":
    unittest.main()
