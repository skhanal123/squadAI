import json
from typing import Any

from pydantic import BaseModel, Field

from squadAI.chat import ChatHistory
from squadAI.config import get_settings
from squadAI.llm import create_provider
from squadAI.providers.base import LLMResponse, ToolCall
from squadAI.tools import Tool


def _default_max_iterations() -> int:
    return get_settings().react_max_iterations

TOOL_SYSTEM_PROMPT = """
You have access to tools for completing tasks. Use them when needed instead of guessing.
When a tool is required, call it with the correct arguments. After receiving tool results,
provide a clear final answer to the user.
"""


class ReactAgentError(Exception):
    """Base exception for ReactAgent failures."""


class ReactAgentMaxIterationsError(ReactAgentError):
    """Raised when the ReAct loop exhausts max_iterations without a final response."""


class ReactAgent(BaseModel):
    """
    ReAct agent with native provider tool calling.

    Attributes:
    -----------
    tools: list of tools available to the agent
    prompt: system/backstory prompt for the agent
    max_iterations: maximum ReAct loop iterations
    provider: LLM provider used for completions
    """

    tools: list[Tool] = []
    prompt: str
    max_iterations: int = Field(default_factory=_default_max_iterations)
    provider: Any = Field(default_factory=create_provider)

    model_config = {"arbitrary_types_allowed": True}

    def _create_tool_dict(self) -> dict[str, Tool]:
        return {tool.function_name: tool for tool in self.tools}

    def _tool_schemas(self) -> list[dict]:
        return [tool.to_openai_schema() for tool in self.tools]

    def _create_system_prompt(self) -> str:
        if self.tools:
            return self.prompt + TOOL_SYSTEM_PROMPT
        return self.prompt

    def _create_chat_history(self) -> ChatHistory:
        chat_history = ChatHistory()
        chat_history.add_chat(role="system", prompt=self._create_system_prompt())
        return chat_history

    def _complete(
        self,
        chat_history: ChatHistory,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        response = self.provider.complete(chat_history.chat(), tools=tools)
        if response.content is None and not response.has_tool_calls:
            raise ReactAgentError("LLM returned empty content")
        return response

    async def _complete_async(
        self,
        chat_history: ChatHistory,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        response = await self.provider.complete_async(chat_history.chat(), tools=tools)
        if response.content is None and not response.has_tool_calls:
            raise ReactAgentError("LLM returned empty content")
        return response

    def _tool_calls_to_api(self, tool_calls: list[ToolCall]) -> list[dict]:
        return [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments),
                },
            }
            for tool_call in tool_calls
        ]

    def _execute_tool_call(
        self,
        tools_dict: dict[str, Tool],
        tool_name: str,
        arguments: dict,
    ) -> str:
        if not tool_name:
            return "Tool call missing required 'name' field."

        if tool_name not in tools_dict:
            available = ", ".join(tools_dict) or "none"
            return f"Unknown tool '{tool_name}'. Available tools: {available}."

        if not isinstance(arguments, dict):
            return "Tool call missing required 'arguments' dict."

        try:
            return str(tools_dict[tool_name].run(**arguments))
        except Exception as exc:
            return f"Error executing tool '{tool_name}': {exc}"

    def invoke(self, user_query: str) -> str:
        chat_history = self._create_chat_history()
        chat_history.add_chat(role="user", prompt=user_query)

        if not self.tools:
            response = self._complete(chat_history)
            return response.content or ""

        tools_dict = self._create_tool_dict()
        tool_schemas = self._tool_schemas()

        for _ in range(self.max_iterations):
            response = self._complete(chat_history, tools=tool_schemas)

            if response.has_tool_calls:
                chat_history.add_assistant(
                    content=response.content,
                    tool_calls=self._tool_calls_to_api(response.tool_calls),
                )
                for tool_call in response.tool_calls:
                    result = self._execute_tool_call(
                        tools_dict,
                        tool_call.name,
                        tool_call.arguments,
                    )
                    chat_history.add_tool_result(tool_call.id, result)
                continue

            if response.content:
                return response.content

            chat_history.add_chat(
                role="user",
                prompt="Provide a final answer or call a tool to continue.",
            )

        raise ReactAgentMaxIterationsError(
            f"ReAct loop did not produce a final answer within {self.max_iterations} iterations"
        )

    async def invoke_async(self, user_query: str) -> str:
        chat_history = self._create_chat_history()
        chat_history.add_chat(role="user", prompt=user_query)

        if not self.tools:
            response = await self._complete_async(chat_history)
            return response.content or ""

        tools_dict = self._create_tool_dict()
        tool_schemas = self._tool_schemas()

        for _ in range(self.max_iterations):
            response = await self._complete_async(chat_history, tools=tool_schemas)

            if response.has_tool_calls:
                chat_history.add_assistant(
                    content=response.content,
                    tool_calls=self._tool_calls_to_api(response.tool_calls),
                )
                for tool_call in response.tool_calls:
                    result = self._execute_tool_call(
                        tools_dict,
                        tool_call.name,
                        tool_call.arguments,
                    )
                    chat_history.add_tool_result(tool_call.id, result)
                continue

            if response.content:
                return response.content

            chat_history.add_chat(
                role="user",
                prompt="Provide a final answer or call a tool to continue.",
            )

        raise ReactAgentMaxIterationsError(
            f"ReAct loop did not produce a final answer within {self.max_iterations} iterations"
        )
