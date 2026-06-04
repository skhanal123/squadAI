import asyncio

from squadAI.providers.base import LLMResponse, ToolCall
from squadAI.usage import TokenUsage


class MockProvider:
    """In-memory provider for tests and offline demos."""

    def __init__(
        self,
        responses: list[LLMResponse] | None = None,
        *,
        model: str = "mock-model",
    ):
        self.model = model
        self.responses = list(responses or [])
        self.calls: list[dict] = []
        self._lock = asyncio.Lock()

    def queue(self, *responses: LLMResponse) -> None:
        self.responses.extend(responses)

    def _next_response(
        self,
        messages: list[dict],
        tools: list[dict] | None,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools})
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(
            content="Mock response with no queued replies.",
            usage=TokenUsage(),
        )

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        return self._next_response(messages, tools)

    async def complete_async(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        async with self._lock:
            await asyncio.sleep(0)
            return self._next_response(messages, tools)