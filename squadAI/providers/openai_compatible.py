import json
from typing import Any

from openai import AsyncOpenAI, OpenAI

from squadAI.providers.base import LLMResponse, ToolCall
from squadAI.usage import usage_from_openai_response


class OpenAIChatProvider:
    """Shared OpenAI SDK chat completions backend for OpenAI-shaped APIs."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        async_client: Any | None = None,
    ):
        self.model = model
        self.client = client or OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = async_client or AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def _completion_kwargs(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        *,
        response_format: dict | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        elif response_format is not None:
            kwargs["response_format"] = response_format
        return kwargs

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        response_format: dict | None = None,
    ) -> LLMResponse:
        response = self.client.chat.completions.create(
            **self._completion_kwargs(
                messages,
                tools,
                response_format=response_format,
            )
        )
        message = response.choices[0].message
        return parse_openai_message(response, message)

    async def complete_async(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        response_format: dict | None = None,
    ) -> LLMResponse:
        response = await self.async_client.chat.completions.create(
            **self._completion_kwargs(
                messages,
                tools,
                response_format=response_format,
            )
        )
        message = response.choices[0].message
        return parse_openai_message(response, message)


class OpenAICompatibleProvider(OpenAIChatProvider):
    """Generic OpenAI-compatible API (Ollama, custom ``LLM_BASE_URL``, etc.)."""


def parse_openai_message(response: Any, message: Any) -> LLMResponse:
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
        usage=usage_from_openai_response(response),
    )


# Backward-compatible alias used by Groq provider
_parse_message = parse_openai_message
