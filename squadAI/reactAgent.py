import json
from typing import Any

from pydantic import BaseModel, Field

from squadAI.chat import ChatHistory
from squadAI.config import get_settings
from squadAI.llm import create_provider
from squadAI.output_schema import (
    TaskOutputParseError,
    build_openai_response_format,
    build_openai_response_format_from_schema,
    normalize_task_output,
    provider_supports_structured_output,
    try_normalize_task_output,
)
from squadAI.providers.base import LLMResponse, ToolCall
from squadAI.tools import Tool
from squadAI.usage import AgentRunResult, TokenUsage, resolve_model_name


def _default_max_iterations() -> int:
    return get_settings().react_max_iterations

TOOL_SYSTEM_PROMPT = """
You have access to tools for completing tasks. Use them when needed instead of guessing.
When a tool is required, call it with the correct arguments. After receiving tool results,
provide a clear final answer to the user.
"""

STRUCTURED_FINAL_PROMPT = (
    "Return your final answer as a single JSON object that matches the required "
    "output schema. Do not include markdown fences or explanatory prose."
)

STRUCTURED_RETRY_PROMPT = (
    "Your previous response did not match the required JSON output schema. "
    "Return only a valid JSON object that satisfies the schema."
)


class ReactAgentError(Exception):
    """Base exception for ReactAgent failures."""


class ReactAgentMaxIterationsError(ReactAgentError):
    """Raised when the ReAct loop exhausts max_iterations without a final response."""


class ReactAgentOutputSchemaError(ReactAgentError):
    """Raised when structured task output cannot be validated."""


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

    def _accumulate_usage(
        self,
        accumulator: TokenUsage,
        response: LLMResponse,
    ) -> TokenUsage:
        if response.usage:
            return accumulator + response.usage
        return accumulator

    def _response_format(
        self,
        *,
        output_schema: type[BaseModel] | None,
        output_json_schema: dict | None,
    ) -> dict | None:
        if not provider_supports_structured_output(self.provider):
            return None
        if output_schema is not None:
            return build_openai_response_format(output_schema)
        if output_json_schema is not None:
            return build_openai_response_format_from_schema(output_json_schema)
        return None

    def _complete(
        self,
        chat_history: ChatHistory,
        tools: list[dict] | None = None,
        *,
        response_format: dict | None = None,
    ) -> LLMResponse:
        response = self.provider.complete(
            chat_history.chat(),
            tools=tools,
            response_format=response_format,
        )
        if response.content is None and not response.has_tool_calls:
            raise ReactAgentError("LLM returned empty content")
        return response

    async def _complete_async(
        self,
        chat_history: ChatHistory,
        tools: list[dict] | None = None,
        *,
        response_format: dict | None = None,
    ) -> LLMResponse:
        response = await self.provider.complete_async(
            chat_history.chat(),
            tools=tools,
            response_format=response_format,
        )
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

    def _finalize_structured_output(
        self,
        raw: str,
        *,
        output_schema: type[BaseModel] | None,
        output_json_schema: dict | None,
    ) -> str:
        try:
            return normalize_task_output(
                raw,
                model=output_schema,
                json_schema=output_json_schema,
            )
        except TaskOutputParseError as exc:
            raise ReactAgentOutputSchemaError(str(exc)) from exc

    def _structured_completion(
        self,
        chat_history: ChatHistory,
        usage: TokenUsage,
        *,
        response_format: dict | None,
        output_schema: type[BaseModel] | None,
        output_json_schema: dict | None,
        prompt: str | None = None,
    ) -> tuple[str, TokenUsage]:
        if prompt:
            chat_history.add_chat(role="user", prompt=prompt)
        response = self._complete(
            chat_history,
            tools=None,
            response_format=response_format,
        )
        usage = self._accumulate_usage(usage, response)
        output = self._finalize_structured_output(
            response.content or "",
            output_schema=output_schema,
            output_json_schema=output_json_schema,
        )
        return output, usage

    async def _structured_completion_async(
        self,
        chat_history: ChatHistory,
        usage: TokenUsage,
        *,
        response_format: dict | None,
        output_schema: type[BaseModel] | None,
        output_json_schema: dict | None,
        prompt: str | None = None,
    ) -> tuple[str, TokenUsage]:
        if prompt:
            chat_history.add_chat(role="user", prompt=prompt)
        response = await self._complete_async(
            chat_history,
            tools=None,
            response_format=response_format,
        )
        usage = self._accumulate_usage(usage, response)
        output = self._finalize_structured_output(
            response.content or "",
            output_schema=output_schema,
            output_json_schema=output_json_schema,
        )
        return output, usage

    def _resolve_structured_output(
        self,
        chat_history: ChatHistory,
        usage: TokenUsage,
        draft: str,
        *,
        response_format: dict | None,
        output_schema: type[BaseModel] | None,
        output_json_schema: dict | None,
    ) -> tuple[str, TokenUsage]:
        normalized = try_normalize_task_output(
            draft,
            model=output_schema,
            json_schema=output_json_schema,
        )
        if normalized is not None:
            return normalized, usage

        output, usage = self._structured_completion(
            chat_history,
            usage,
            response_format=response_format,
            output_schema=output_schema,
            output_json_schema=output_json_schema,
            prompt=STRUCTURED_FINAL_PROMPT,
        )
        return output, usage

    async def _resolve_structured_output_async(
        self,
        chat_history: ChatHistory,
        usage: TokenUsage,
        draft: str,
        *,
        response_format: dict | None,
        output_schema: type[BaseModel] | None,
        output_json_schema: dict | None,
    ) -> tuple[str, TokenUsage]:
        normalized = try_normalize_task_output(
            draft,
            model=output_schema,
            json_schema=output_json_schema,
        )
        if normalized is not None:
            return normalized, usage

        output, usage = await self._structured_completion_async(
            chat_history,
            usage,
            response_format=response_format,
            output_schema=output_schema,
            output_json_schema=output_json_schema,
            prompt=STRUCTURED_FINAL_PROMPT,
        )
        return output, usage

    def invoke(
        self,
        user_query: str,
        *,
        output_schema: type[BaseModel] | None = None,
        output_json_schema: dict | None = None,
    ) -> AgentRunResult:
        model = resolve_model_name(self.provider)
        usage = TokenUsage()
        chat_history = self._create_chat_history()
        chat_history.add_chat(role="user", prompt=user_query)
        response_format = self._response_format(
            output_schema=output_schema,
            output_json_schema=output_json_schema,
        )
        structured = output_schema is not None or output_json_schema is not None

        if not self.tools:
            response = self._complete(
                chat_history,
                response_format=response_format if structured else None,
            )
            usage = self._accumulate_usage(usage, response)
            output = response.content or ""
            if structured:
                try:
                    output = self._finalize_structured_output(
                        output,
                        output_schema=output_schema,
                        output_json_schema=output_json_schema,
                    )
                except ReactAgentOutputSchemaError:
                    output, usage = self._structured_completion(
                        chat_history,
                        usage,
                        response_format=response_format,
                        output_schema=output_schema,
                        output_json_schema=output_json_schema,
                        prompt=STRUCTURED_RETRY_PROMPT,
                    )
            return AgentRunResult(output=output, usage=usage, model=model)

        tools_dict = self._create_tool_dict()
        tool_schemas = self._tool_schemas()

        for _ in range(self.max_iterations):
            response = self._complete(chat_history, tools=tool_schemas)
            usage = self._accumulate_usage(usage, response)

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
                output = response.content
                if structured:
                    output, usage = self._resolve_structured_output(
                        chat_history,
                        usage,
                        output,
                        response_format=response_format,
                        output_schema=output_schema,
                        output_json_schema=output_json_schema,
                    )
                return AgentRunResult(output=output, usage=usage, model=model)

            chat_history.add_chat(
                role="user",
                prompt="Provide a final answer or call a tool to continue.",
            )

        raise ReactAgentMaxIterationsError(
            f"ReAct loop did not produce a final answer within {self.max_iterations} iterations"
        )

    async def invoke_async(
        self,
        user_query: str,
        *,
        output_schema: type[BaseModel] | None = None,
        output_json_schema: dict | None = None,
    ) -> AgentRunResult:
        model = resolve_model_name(self.provider)
        usage = TokenUsage()
        chat_history = self._create_chat_history()
        chat_history.add_chat(role="user", prompt=user_query)
        response_format = self._response_format(
            output_schema=output_schema,
            output_json_schema=output_json_schema,
        )
        structured = output_schema is not None or output_json_schema is not None

        if not self.tools:
            response = await self._complete_async(
                chat_history,
                response_format=response_format if structured else None,
            )
            usage = self._accumulate_usage(usage, response)
            output = response.content or ""
            if structured:
                try:
                    output = self._finalize_structured_output(
                        output,
                        output_schema=output_schema,
                        output_json_schema=output_json_schema,
                    )
                except ReactAgentOutputSchemaError:
                    output, usage = await self._structured_completion_async(
                        chat_history,
                        usage,
                        response_format=response_format,
                        output_schema=output_schema,
                        output_json_schema=output_json_schema,
                        prompt=STRUCTURED_RETRY_PROMPT,
                    )
            return AgentRunResult(output=output, usage=usage, model=model)

        tools_dict = self._create_tool_dict()
        tool_schemas = self._tool_schemas()

        for _ in range(self.max_iterations):
            response = await self._complete_async(chat_history, tools=tool_schemas)
            usage = self._accumulate_usage(usage, response)

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
                output = response.content
                if structured:
                    output, usage = await self._resolve_structured_output_async(
                        chat_history,
                        usage,
                        output,
                        response_format=response_format,
                        output_schema=output_schema,
                        output_json_schema=output_json_schema,
                    )
                return AgentRunResult(output=output, usage=usage, model=model)

            chat_history.add_chat(
                role="user",
                prompt="Provide a final answer or call a tool to continue.",
            )

        raise ReactAgentMaxIterationsError(
            f"ReAct loop did not produce a final answer within {self.max_iterations} iterations"
        )
