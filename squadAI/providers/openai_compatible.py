import json
from typing import Any

from openai import OpenAI

from squadAI.providers.base import LLMProvider, LLMResponse, ToolCall


class OpenAICompatibleProvider:
    """Provider for OpenAI-compatible APIs (DeepSeek, Ollama, etc.)."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ):
        self.model = model
        self.client = client or OpenAI(api_key=api_key, base_url=base_url)

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


def _parse_message(response: Any, message: Any) -> LLMResponse:
    tool_calls: list[ToolCall] = []
    if message.tool_calls:
        for tool_call in message.tool_calls:
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}

            tool_calls.append(
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=arguments,
                )
            )

    return LLMResponse(
        content=message.content,
        tool_calls=tool_calls,
        finish_reason=response.choices[0].finish_reason,
        raw=response,
    )
