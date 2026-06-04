import unittest
from types import SimpleNamespace

from squadAI.providers.base import LLMResponse
from squadAI.providers.mock import MockProvider
from squadAI.providers.openai_compatible import parse_openai_message
from squadAI.providers.anthropic import anthropic_response_to_llm_response
from squadAI.reactAgent import ReactAgent
from squadAI.createAgent import Agent
from squadAI.squadAgent import SquadAgents
from squadAI.task import Task
from squadAI.usage import (
    TaskUsage,
    TokenUsage,
    merge_usage_by_model,
    usage_from_anthropic_response,
    usage_from_openai_response,
)


class TestUsageParsers(unittest.TestCase):
    def test_usage_from_openai_response(self):
        raw = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50)
        )
        usage = usage_from_openai_response(raw)
        self.assertEqual(usage.input_tokens, 100)
        self.assertEqual(usage.output_tokens, 50)
        self.assertEqual(usage.total_tokens, 150)

    def test_usage_from_openai_response_missing(self):
        self.assertEqual(usage_from_openai_response(SimpleNamespace()), TokenUsage())

    def test_usage_from_anthropic_response(self):
        raw = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=200, output_tokens=75)
        )
        usage = usage_from_anthropic_response(raw)
        self.assertEqual(usage.input_tokens, 200)
        self.assertEqual(usage.output_tokens, 75)

    def test_parse_openai_message_includes_usage(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Hi", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )
        message = response.choices[0].message
        llm_response = parse_openai_message(response, message)
        self.assertEqual(llm_response.usage.input_tokens, 10)
        self.assertEqual(llm_response.usage.output_tokens, 5)

    def test_anthropic_response_includes_usage(self):
        response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Hello")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=30, output_tokens=12),
        )
        llm_response = anthropic_response_to_llm_response(response)
        self.assertEqual(llm_response.usage.input_tokens, 30)
        self.assertEqual(llm_response.usage.output_tokens, 12)


class TestTokenUsageMath(unittest.TestCase):
    def test_task_usage_add_same_model(self):
        a = TaskUsage(model="gpt-4o", input_tokens=10, output_tokens=5)
        b = TaskUsage(model="gpt-4o", input_tokens=20, output_tokens=8)
        combined = a + b
        self.assertEqual(combined.input_tokens, 30)
        self.assertEqual(combined.output_tokens, 13)

    def test_task_usage_add_different_model_raises(self):
        with self.assertRaises(ValueError):
            TaskUsage(model="a") + TaskUsage(model="b")


class TestReactAgentUsage(unittest.TestCase):
    def test_react_loop_accumulates_usage(self):
        from squadAI.providers.base import ToolCall
        from squadAI.tools import tool_wrapper

        @tool_wrapper
        def echo(value: str) -> str:
            """Echo a value."""
            return value

        provider = MockProvider(
            [
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(id="c1", name="echo", arguments={"value": "x"})
                    ],
                    usage=TokenUsage(10, 5),
                ),
                LLMResponse(content="Final.", usage=TokenUsage(20, 8)),
            ],
            model="test-model",
        )
        agent = ReactAgent(tools=[echo], prompt="Helper.", provider=provider)
        result = agent.invoke("Hello")
        self.assertEqual(result.output, "Final.")
        self.assertEqual(result.model, "test-model")
        self.assertEqual(result.usage.input_tokens, 30)
        self.assertEqual(result.usage.output_tokens, 13)


class TestSquadUsage(unittest.TestCase):
    def test_per_task_and_squad_totals(self):
        provider = MockProvider(
            [
                LLMResponse(content="First.", usage=TokenUsage(10, 5)),
                LLMResponse(content="Second.", usage=TokenUsage(20, 8)),
            ],
            model="test-model",
        )
        agent = Agent(backstory="Helper.", provider=provider)
        task1 = Task(task_description="First task", agent=agent)
        task2 = Task(
            task_description="Second task",
            dependency=[task1],
            agent=agent,
        )
        result = SquadAgents(tasks=[task2, task1]).run()

        self.assertEqual(result.get_usage(task1).model, "test-model")
        self.assertEqual(result.get_usage(task1).total_tokens, 15)
        self.assertEqual(result.get_usage(task2).total_tokens, 28)
        self.assertEqual(result.usage.total_tokens, 43)
        self.assertEqual(result.usage_by_model["test-model"].total_tokens, 43)

    def test_usage_display_only_when_flag_set(self):
        provider = MockProvider(
            [LLMResponse(content="Done.", usage=TokenUsage(5, 2))],
            model="mock-model",
        )
        agent = Agent(backstory="Helper.", provider=provider)
        task = Task(task_description="Task", agent=agent)

        without = SquadAgents(tasks=[task]).run()
        self.assertIsNone(without.usage_display)
        self.assertEqual(without.usage.total_tokens, 7)

        with_display = SquadAgents(tasks=[task]).run(include_usage_in_result=True)
        self.assertIsNotNone(with_display.usage_display)
        self.assertIn("mock-model", with_display.usage_display)

    def test_validation_retry_accumulates_usage(self):
        attempts = {"count": 0}

        def validator(output: str):
            from squadAI.validation import ValidationResult

            attempts["count"] += 1
            if len(output) >= 10:
                return ValidationResult(approved=True)
            return ValidationResult(approved=False, feedback="Too short.")

        provider = MockProvider(
            [
                LLMResponse(content="short", usage=TokenUsage(1, 1)),
                LLMResponse(content="long enough", usage=TokenUsage(10, 5)),
            ],
            model="retry-model",
        )
        agent = Agent(backstory="Writer.", provider=provider)
        task = Task(
            task_description="Write.",
            agent=agent,
            validator=validator,
            max_retries=2,
        )
        result = SquadAgents(tasks=[task]).run()

        self.assertEqual(attempts["count"], 2)
        self.assertEqual(result.get_usage(task).total_tokens, 17)
