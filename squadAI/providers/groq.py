import json
from typing import Any

from groq import AsyncGroq, Groq

from squadAI.providers.base import LLMProvider, LLMResponse, ToolCall
from squadAI.providers.openai_compatible import _parse_message


class GroqProvider:
    """Provider for the Groq SDK."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        client: Any | None = None,
        async_client: Any | None = None,
    ):
        self.model = model
        self.client = client or Groq(api_key=api_key)
        self.async_client = async_client or AsyncGroq(api_key=api_key)

    def _completion_kwargs(
        self,
        messages: list[dict],
        tools: list[dict] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return kwargs

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        response = self.client.chat.completions.create(
            **self._completion_kwargs(messages, tools)
        )
        message = response.choices[0].message
        return _parse_message(response, message)

    async def complete_async(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        response = await self.async_client.chat.completions.create(
            **self._completion_kwargs(messages, tools)
        )
        message = response.choices[0].message
        return _parse_message(response, message)