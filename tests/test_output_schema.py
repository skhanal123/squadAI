"""Tests for structured task output schemas."""

import json
import unittest

from pydantic import BaseModel, Field, ValidationError

from squadAI.createAgent import Agent
from squadAI.output_schema import (
    TaskOutputBase,
    format_context_block,
    normalize_task_output,
)
from squadAI.providers.base import LLMResponse
from squadAI.providers.mock import MockProvider
from squadAI.squadAgent import SquadAgents, build_task_context
from squadAI.task import Task


class SampleReport(TaskOutputBase):
    summary: str = Field(min_length=1)
    confidence: str


class TestTaskOutputBase(unittest.TestCase):
    def test_rejects_too_many_extras(self):
        with self.assertRaises(ValidationError):
            SampleReport(
                summary="ok",
                confidence="high",
                extras={"one": "a", "two": "b", "three": "c", "four": "d"},
            )

    def test_rejects_long_extra_values(self):
        with self.assertRaises(ValidationError):
            SampleReport(
                summary="ok",
                confidence="high",
                extras={"note": "x" * 301},
            )


class TestNormalizeTaskOutput(unittest.TestCase):
    def test_normalizes_pydantic_model_output(self):
        payload = SampleReport(summary="latency spike", confidence="high")
        normalized = normalize_task_output(
            payload.model_dump_json(),
            model=SampleReport,
        )
        self.assertEqual(json.loads(normalized)["summary"], "latency spike")


class TestStructuredSquadRun(unittest.TestCase):
    def test_task_with_output_schema_returns_json(self):
        report = SampleReport(summary="done", confidence="high")
        provider = MockProvider([LLMResponse(content=report.model_dump_json())])
        agent = Agent(backstory="Reporter.", provider=provider)
        task = Task(
            task_description="Produce a report.",
            agent=agent,
            output_schema=SampleReport,
        )
        squad = SquadAgents(tasks=[task])

        result = squad.run()

        parsed = json.loads(result.final)
        self.assertEqual(parsed["summary"], "done")
        self.assertIn("response_format", provider.calls[0])
        self.assertIsNotNone(provider.calls[0]["response_format"])

    def test_json_upstream_context_is_labeled(self):
        upstream_json = json.dumps({"summary": "price", "confidence": "high"})
        provider = MockProvider(
            [
                LLMResponse(content=upstream_json),
                LLMResponse(content='{"summary":"merged","confidence":"high"}'),
            ]
        )
        agent = Agent(backstory="Helper.", provider=provider)
        task1 = Task(
            task_description="First",
            agent=agent,
            output_schema=SampleReport,
        )
        task2 = Task(
            task_description="Second",
            agent=agent,
            dependency=[task1],
            output_schema=SampleReport,
        )
        squad = SquadAgents(tasks=[task1, task2])
        squad.run()

        context = build_task_context([task1], {task1.id: upstream_json})
        self.assertIn('format="application/json"', context)
        self.assertIn('"summary": "price"', context)

        second_prompt = provider.calls[1]["messages"][-1]["content"]
        self.assertIn('format="application/json"', second_prompt)


class TestFormatContextBlock(unittest.TestCase):
    def test_plain_text_block_has_no_json_format_attribute(self):
        block = format_context_block(
            index=1,
            description="Plain task",
            output="hello",
        )
        self.assertNotIn("application/json", block)
        self.assertIn("<output>hello</output>", block)


if __name__ == "__main__":
    unittest.main()
