import json
from typing import Any
from anthropic import AsyncAnthropic, Anthropic
from squadAI.providers.base import LLMResponse, ToolCall

DEFAULT_MAX_TOKENS = 4096


def openai_tools_to_anthropic(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None

    anthropic_tools: list[dict] = []
    for tool in tools:
        fn = tool.get("function", tool)
        anthropic_tools.append(
            {
                "name": fn["name"],
                "description": fn.get("description") or "",
                "input_schema": fn.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
        )
    return anthropic_tools


def openai_messages_to_anthropic(
    messages: list[dict],
) -> tuple[str | None, list[dict]]:
    system_parts: list[str] = []
    anthropic_messages: list[dict] = []

    for message in messages:
        role = message.get("role")
        if role == "system":
            content = message.get("content")
            if content:
                system_parts.append(str(content))
            continue

        if role == "tool":
            tool_result = {
                "type": "tool_result",
                "tool_use_id": message["tool_call_id"],
                "content": str(message.get("content", "")),
            }
            if (
                anthropic_messages
                and anthropic_messages[-1]["role"] == "user"
                and isinstance(anthropic_messages[-1].get("content"), list)
                and anthropic_messages[-1]["content"]
                and anthropic_messages[-1]["content"][0].get("type") == "tool_result"
            ):
                anthropic_messages[-1]["content"].append(tool_result)
            else:
                anthropic_messages.append({"role": "user", "content": [tool_result]})
            continue

        if role == "assistant":
            content_blocks: list[dict] = []
            text = message.get("content")
            if text:
                content_blocks.append({"type": "text", "text": str(text)})

            for tool_call in message.get("tool_calls") or []:
                fn = tool_call.get("function", {})
                raw_args = fn.get("arguments", "{}")
                try:
                    tool_input = (
                        json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    )
                except json.JSONDecodeError:
                    tool_input = {}
                if not isinstance(tool_input, dict):
                    tool_input = {}

                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": fn["name"],
                        "input": tool_input,
                    }
                )

            if not content_blocks:
                content_blocks.append({"type": "text", "text": ""})

            anthropic_messages.append({"role": "assistant", "content": content_blocks})
            continue

        if role == "user":
            content = message.get("content")
            if content is None:
                content = ""
            if isinstance(content, str):
                anthropic_messages.append({"role": "user", "content": content})
            else:
                anthropic_messages.append({"role": "user", "content": content})
            continue

    system = "\n\n".join(system_parts) if system_parts else None
    return system, anthropic_messages


def anthropic_response_to_llm_response(response: Any) -> LLMResponse:
    tool_calls: list[ToolCall] = []
    text_parts: list[str] = []

    for block in response.content:
        block_type = getattr(block, "type", None) or block.get("type")
        if block_type == "text":
            text = getattr(block, "text", None) or block.get("text", "")
            if text:
                text_parts.append(text)
        elif block_type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=getattr(block, "id", None) or block["id"],
                    name=getattr(block, "name", None) or block["name"],
                    arguments=getattr(block, "input", None) or block.get("input", {}),
                )
            )

    content = "\n".join(text_parts) if text_parts else None
    if not content and not tool_calls:
        content = None

    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason=response.stop_reason,
        raw=response,
    )


class AnthropicProvider:
    """Provider for the Anthropic Messages API (Claude)."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: Any | None = None,
        async_client: Any | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.client = client or Anthropic(api_key=api_key)
        self.async_client = async_client or AsyncAnthropic(api_key=api_key)

    def _request_kwargs(
        self,
        messages: list[dict],
        tools: list[dict] | None,
    ) -> dict[str, Any]:
        system, anthropic_messages = openai_messages_to_anthropic(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": anthropic_messages,
        }
        if system:
            kwargs["system"] = system

        converted_tools = openai_tools_to_anthropic(tools)
        if converted_tools:
            kwargs["tools"] = converted_tools

        return kwargs

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        response = self.client.messages.create(**self._request_kwargs(messages, tools))
        return anthropic_response_to_llm_response(response)

    async def complete_async(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        response = await self.async_client.messages.create(
            **self._request_kwargs(messages, tools)
        )
        return anthropic_response_to_llm_response(response)
