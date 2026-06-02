import json
from typing import Any

from groq import Groq

from squadAI.providers.base import LLMProvider, LLMResponse, ToolCall
from squadAI.providers.openai_compatible import _parse_message


class GroqProvider:
    """Provider for the Groq SDK."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        client: Any | None = None,
    ):
        self.model = model
        self.client = client or Groq(api_key=api_key)

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        return _parse_message(response, message)
