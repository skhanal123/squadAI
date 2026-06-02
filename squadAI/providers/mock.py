from squadAI.providers.base import LLMProvider, LLMResponse, ToolCall


class MockProvider:
    """In-memory provider for tests and offline demos."""

    def __init__(self, responses: list[LLMResponse] | None = None):
        self.responses = list(responses or [])
        self.calls: list[dict] = []

    def queue(self, *responses: LLMResponse) -> None:
        self.responses.extend(responses)

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools})
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(content="Mock response with no queued replies.")
