from dataclasses import dataclass, field
from typing import Any, Protocol

from squadAI.usage import TokenUsage


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    raw: Any = None
    usage: TokenUsage | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLMProvider(Protocol):
    """Normalized interface for LLM backends used by ReactAgent."""

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """Send messages (and optional tool schemas) to the LLM."""

    async def complete_async(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """Async variant of :meth:`complete`."""
