class ChatHistory:
    """
    Chat history helper for LLM conversations.

    Supports plain messages and native tool-calling message shapes
    (assistant messages with tool_calls and tool result messages).
    """

    def __init__(self):
        self.history: list[dict] = []

    def add_chat(self, role: str, prompt: str | None = None, **extra):
        message: dict = {"role": role, "content": prompt, **extra}
        if message.get("content") is None and "tool_calls" not in message:
            message["content"] = ""
        self.history.append(message)

    def add_assistant(
        self,
        content: str | None,
        tool_calls: list[dict] | None = None,
    ) -> None:
        message: dict = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        self.history.append(message)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self.history.append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        )

    def chat(self) -> list[dict]:
        return self.history
