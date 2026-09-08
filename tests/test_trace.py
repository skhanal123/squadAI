import json
import tempfile
import unittest
from pathlib import Path

from squadAI.createAgent import Agent
from squadAI.providers.base import LLMResponse, ToolCall
from squadAI.providers.mock import MockProvider
from squadAI.reactAgent import ReactAgent, ReactAgentMaxIterationsError
from squadAI.squadAgent import SquadAgents
from squadAI.task import Task
from squadAI.tools import tool_wrapper
from squadAI.trace import DEFAULT_TRACE_FILENAME
from squadAI.validation import ValidationResult


@tool_wrapper
def echo_value(value: str) -> str:
    """Echo a string value."""
    return value


class TestReactAgentTrace(unittest.TestCase):
    def test_tool_calls_recorded_in_trace(self):
        provider = MockProvider(
            [
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="echo_value",
                            arguments={"value": "metrics data"},
                        )
                    ],
                ),
                LLMResponse(content="Done investigating."),
            ]
        )
        agent = ReactAgent(
            tools=[echo_value],
            prompt="Investigator.",
            provider=provider,
        )

        result = agent.invoke("Investigate")

        self.assertEqual(result.output, "Done investigating.")
        self.assertIsNotNone(result.trace)
        self.assertEqual(result.trace.status, "success")
        self.assertEqual(len(result.trace.react_steps), 2)
        tool_step = result.trace.react_steps[0]
        self.assertEqual(len(tool_step.tool_calls), 1)
        self.assertEqual(tool_step.tool_calls[0].name, "echo_value")
        self.assertEqual(tool_step.tool_calls[0].arguments, {"value": "metrics data"})
        self.assertFalse(tool_step.tool_calls[0].error)

    def test_max_iterations_trace_on_exception(self):
        provider = MockProvider(
            [
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="echo_value",
                            arguments={"value": "x"},
                        )
                    ],
                )
            ]
        )
        agent = ReactAgent(
            tools=[echo_value],
            prompt="Investigator.",
            provider=provider,
            max_iterations=1,
        )

        with self.assertRaises(ReactAgentMaxIterationsError) as ctx:
            agent.invoke("Investigate")

        self.assertIsNotNone(ctx.exception.trace)
        self.assertEqual(ctx.exception.trace.status, "react_exhausted")
        self.assertEqual(len(ctx.exception.trace.react_steps), 1)


class TestSquadTrace(unittest.TestCase):
    def test_validation_retries_recorded_on_task_result(self):
        provider = MockProvider(
            [
                LLMResponse(content="short"),
                LLMResponse(content="long enough"),
            ]
        )
        agent = Agent(backstory="Writer.", provider=provider)

        def min_length_validator(output: str) -> ValidationResult:
            if len(output) >= 10:
                return ValidationResult(approved=True)
            return ValidationResult(
                approved=False,
                feedback="Response is too short.",
            )

        task = Task(
            task_description="Write a sentence.",
            agent=agent,
            validator=min_length_validator,
            max_retries=2,
        )
        result = SquadAgents(tasks=[task]).run()

        trace = result.task_results[0].trace
        self.assertEqual(trace.status, "success")
        self.assertEqual(len(trace.attempts), 2)
        self.assertFalse(trace.attempts[0].validation.approved)
        self.assertEqual(trace.attempts[0].validation.feedback, "Response is too short.")
        self.assertTrue(trace.attempts[1].validation.approved)

    def test_dag_level_and_parallel_on_task_result(self):
        provider = MockProvider(
            [
                LLMResponse(content="Price is $999."),
                LLMResponse(content="Tax rate is 8.75%."),
                LLMResponse(content="Total is $3,259.24."),
            ]
        )
        agent = Agent(backstory="Billing helper.", provider=provider)

        task_price = Task(task_description="Look up price", agent=agent)
        task_tax = Task(task_description="Look up tax rate", agent=agent)
        task_total = Task(
            task_description="Calculate checkout total",
            dependency=[task_price, task_tax],
            agent=agent,
        )
        result = SquadAgents(tasks=[task_total, task_price, task_tax]).run()

        price_trace = result.get_trace(task_price)
        tax_trace = result.get_trace(task_tax)
        total_trace = result.get_trace(task_total)

        self.assertEqual(price_trace.dag_level, 0)
        self.assertEqual(tax_trace.dag_level, 0)
        self.assertTrue(price_trace.parallel)
        self.assertTrue(tax_trace.parallel)
        self.assertEqual(total_trace.dag_level, 1)
        self.assertFalse(total_trace.parallel)

    def test_trace_payload_matches_task_results(self):
        provider = MockProvider([LLMResponse(content="Done.")])
        agent = Agent(backstory="Writer.", provider=provider)
        task = Task(task_description="Write.", agent=agent)
        result = SquadAgents(tasks=[task]).run()

        payload = result.trace_payload()
        self.assertEqual(len(payload["tasks"]), 1)
        self.assertEqual(payload["tasks"][0]["description"], "Write.")
        self.assertEqual(payload["tasks"][0]["trace"], result.task_results[0].trace.model_dump())

    def test_write_trace_to_directory(self):
        provider = MockProvider([LLMResponse(content="Done.")])
        agent = Agent(backstory="Writer.", provider=provider)
        task = Task(task_description="Write.", agent=agent)
        result = SquadAgents(tasks=[task]).run()

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = result.write_trace(Path(tmp))
            self.assertEqual(trace_path, Path(tmp) / DEFAULT_TRACE_FILENAME)
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["tasks"]), 1)

    def test_run_writes_trace_when_trace_output_dir_set(self):
        provider = MockProvider([LLMResponse(content="Done.")])
        agent = Agent(backstory="Writer.", provider=provider)
        task = Task(task_description="Write.", agent=agent)

        with tempfile.TemporaryDirectory() as tmp:
            result = SquadAgents(tasks=[task]).run(trace_output_dir=Path(tmp))
            trace_path = Path(tmp) / DEFAULT_TRACE_FILENAME
            self.assertTrue(trace_path.is_file())
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(result.get(task), "Done.")
            self.assertEqual(len(payload["tasks"]), 1)


if __name__ == "__main__":
    unittest.main()
