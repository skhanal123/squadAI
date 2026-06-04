import unittest
from types import SimpleNamespace
from unittest import mock

from squadAI.providers.anthropic import (
    anthropic_response_to_llm_response,
    openai_messages_to_anthropic,
    openai_tools_to_anthropic,
)
from squadAI.providers.base import ToolCall


class TestAnthropicMessageTranslation(unittest.TestCase):
    def test_openai_tools_to_anthropic(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]

        result = openai_tools_to_anthropic(tools)

        self.assertEqual(result[0]["name"], "get_weather")
        self.assertEqual(result[0]["description"], "Get weather")
        self.assertIn("city", result[0]["input_schema"]["properties"])

    def test_system_message_extraction(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]

        system, anthropic_messages = openai_messages_to_anthropic(messages)

        self.assertEqual(system, "You are helpful.")
        self.assertEqual(len(anthropic_messages), 1)
        self.assertEqual(anthropic_messages[0]["role"], "user")

    def test_assistant_tool_calls_to_tool_use_blocks(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "test"}',
                        },
                    }
                ],
            }
        ]

        _, anthropic_messages = openai_messages_to_anthropic(messages)

        blocks = anthropic_messages[0]["content"]
        tool_use = next(b for b in blocks if b["type"] == "tool_use")
        self.assertEqual(tool_use["id"], "call_1")
        self.assertEqual(tool_use["name"], "search")
        self.assertEqual(tool_use["input"], {"query": "test"})

    def test_consecutive_tool_messages_merge(self):
        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": "result one"},
            {"role": "tool", "tool_call_id": "call_2", "content": "result two"},
        ]

        _, anthropic_messages = openai_messages_to_anthropic(messages)

        self.assertEqual(len(anthropic_messages), 1)
        self.assertEqual(anthropic_messages[0]["role"], "user")
        self.assertEqual(len(anthropic_messages[0]["content"]), 2)
        self.assertEqual(
            anthropic_messages[0]["content"][0]["tool_use_id"], "call_1"
        )
        self.assertEqual(
            anthropic_messages[0]["content"][1]["tool_use_id"], "call_2"
        )

    def test_anthropic_response_to_llm_response(self):
        response = SimpleNamespace(
            stop_reason="tool_use",
            content=[
                SimpleNamespace(type="text", text="Let me check."),
                SimpleNamespace(
                    type="tool_use",
                    id="toolu_01",
                    name="search",
                    input={"q": "weather"},
                ),
            ],
        )

        llm_response = anthropic_response_to_llm_response(response)

        self.assertEqual(llm_response.content, "Let me check.")
        self.assertEqual(len(llm_response.tool_calls), 1)
        self.assertEqual(llm_response.tool_calls[0], ToolCall(
            id="toolu_01",
            name="search",
            arguments={"q": "weather"},
        ))


class TestAnthropicProvider(unittest.TestCase):
    def test_complete_calls_messages_api(self):
        from squadAI.providers.anthropic import AnthropicProvider

        mock_client = mock.MagicMock()
        mock_response = SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="Hello.")],
        )
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider(
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            client=mock_client,
        )

        result = provider.complete(
            [{"role": "user", "content": "Hi"}],
            tools=None,
        )

        mock_client.messages.create.assert_called_once()
        self.assertEqual(result.content, "Hello.")


if __name__ == "__main__":
    unittest.main()
