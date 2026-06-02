import unittest

from squadAI.providers.base import LLMResponse, ToolCall
from squadAI.providers.mock import MockProvider
from squadAI.reactAgent import ReactAgent, ReactAgentMaxIterationsError
from squadAI.tools import tool_wrapper
from squadAI.utils import get_fn_signature
import json


@tool_wrapper
def add_two_numbers(first_number: float, second_number: float) -> float:
    """Add two numbers."""
    return first_number + second_number


class TestUtils(unittest.TestCase):
    def test_fn_signature_includes_required_fields(self):
        schema = json.loads(get_fn_signature(add_two_numbers.fn))
        self.assertEqual(schema["name"], "add_two_numbers")
        self.assertIn("first_number", schema["parameters"]["required"])
        self.assertIn("second_number", schema["parameters"]["required"])
        self.assertEqual(
            schema["parameters"]["properties"]["first_number"]["type"], "number"
        )


class TestToolSchema(unittest.TestCase):
    def test_to_openai_schema(self):
        schema = add_two_numbers.to_openai_schema()
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "add_two_numbers")
        self.assertIn("parameters", schema["function"])


class TestReactAgent(unittest.TestCase):
    def test_tool_call_loop(self):
        provider = MockProvider(
            [
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="add_two_numbers",
                            arguments={"first_number": 2, "second_number": 3},
                        )
                    ],
                ),
                LLMResponse(content="The sum is 5."),
            ]
        )

        agent = ReactAgent(
            tools=[add_two_numbers],
            prompt="You are a math assistant.",
            provider=provider,
        )

        result = agent.invoke("Add 2 and 3")
        self.assertEqual(result, "The sum is 5.")
        self.assertEqual(len(provider.calls), 2)
        self.assertIsNotNone(provider.calls[0]["tools"])
        tool_message = provider.calls[1]["messages"][-1]
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(tool_message["tool_call_id"], "call_1")

    def test_max_iterations(self):
        provider = MockProvider(
            [
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="add_two_numbers",
                            arguments={"first_number": 1, "second_number": 1},
                        )
                    ],
                )
            ]
        )

        agent = ReactAgent(
            tools=[add_two_numbers],
            prompt="You are a math assistant.",
            provider=provider,
            max_iterations=1,
        )

        with self.assertRaises(ReactAgentMaxIterationsError):
            agent.invoke("Add numbers")


if __name__ == "__main__":
    unittest.main()
