import os
from pydantic import BaseModel, Field
from typing import Any
import json
import re
from squadAI.tools import Tool
from squadAI.chat import ChatHistory
from squadAI.llm import create_client
from dotenv import load_dotenv

REACT_SYSTEM_PROMPT = """
You operate by running a loop with the following steps: Thought, Action, Observation.
You are provided with function signatures within <tools></tools> XML tags.
You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug
into functions. Pay special attention to the properties 'types'. You should use those types as in a Python dict.

For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:

<tool_call>
{"name": <function-name>,"arguments": <args-dict>, "id": <monotonically-increasing-id>}
</tool_call>

Here are the available tools / actions:

<tools> 
%s
</tools>

Example session:

<question>What's the current temperature in Madrid?</question>
<thought>I need to get the current weather in Madrid</thought>
<tool_call>{"name": "get_current_weather","arguments": {"location": "Madrid", "unit": "celsius"}, "id": 0}</tool_call>

You will be called again with this:

<observation>{0: {"temperature": 25, "unit": "celsius"}}</observation>

You then output:

<response>The current temperature in Madrid is 25 degrees Celsius</response>

ADDITIONAL CONSTRAINTS:

- If the user asks you something unrelated to any of the tools above, answer freely enclosing your answer with <response></response> tags.
"""

AGENT_TOOL_PROMPT = """
"You are an advanced AI assistant that analyzes and interprets outputs from tools or systems to provide accurate, concise, and actionable answers to user queries. Follow these steps:

1. Understand the Query - Carefully read the user's question to determine what they need.

2. Analyze the Tool Output - Examine the provided data, logs, or observations from the tool. Identify key insights, errors, or patterns.

3. Provide a Structured Response -

    If the output answers the query directly, summarize it clearly.

    If the output is ambiguous or incomplete, explain possible interpretations or request clarification.

    If the output indicates an error, diagnose potential causes and suggest fixes.

4. Maintain Context - If follow-up is needed, guide the user on what additional information or actions are required.
"""

load_dotenv()


class ReactAgentError(Exception):
    """Base exception for ReactAgent failures."""


class ReactAgentMaxIterationsError(ReactAgentError):
    """Raised when the ReAct loop exhausts max_iterations without a final response."""


class ReactAgent(BaseModel):
    """
    This is the base class for implementation of react agent.

    Attributes:
    -----------
    tools: list of tool required to execute the task
    prompt: user query provided to perform the task
    max_iterations: maximum number of iterations (re-tries) if the execution fails
    client: client of the llm to execute the task

    Methods:
    _create_tool_dict: returns dictionary to track the details of the tools assigned to agent
    _create_system_prompt: returns system prompt including all the details including tools
    _create_chat_history: maintain and returns the entire chat history during the conversation
    _extract_tag: extracts content from XML-style tags in LLM output
    _call_llm: sends chat history to the LLM and returns the model response
    _add_observation: appends an observation message to chat history for the ReAct loop
    invoke: this methods takes the user query and take action accordingly to generate and return the response
    """

    tools: list[Tool] = []
    prompt: str
    max_iterations: int = 4
    client: Any = Field(default_factory=create_client)

    def _create_tool_dict(self):
        return {tool.function_name: tool for tool in self.tools}

    def _create_system_prompt(self):
        if self.tools:
            function_signatures = "".join([str(i.fn_signature) for i in self.tools])
            system_prompt = self.prompt + REACT_SYSTEM_PROMPT % (function_signatures)
        else:
            system_prompt = self.prompt

        return system_prompt

    def _create_chat_history(self):
        chat_history = ChatHistory()
        system_prompt = self._create_system_prompt()
        chat_history.add_chat(role="system", prompt=system_prompt)
        return chat_history

    def _extract_tag(self, output: str, tag: str) -> str | None:
        """
        Extracts the inner content of an XML-style tag from LLM output.

        Parameters:
        -----------
        output: raw text response from the LLM
        tag: name of the tag to extract (e.g. "response", "tool_call")

        Returns:
        --------
        stripped tag content, or None if the tag is missing or malformed
        """
        pattern = rf"<{tag}>(.*?)</{tag}>"
        matches = re.findall(pattern, output, re.DOTALL)
        if not matches:
            return None
        return matches[0].strip()

    def _call_llm(self, chat_history: ChatHistory) -> str:
        """
        Sends the current chat history to the LLM and returns the model response.

        Parameters:
        -----------
        chat_history: conversation history to pass to the LLM

        Returns:
        --------
        content of the LLM response

        Raises:
        -------
        ReactAgentError: if the LLM returns empty content
        """
        llm_response = (
            self.client.chat.completions.create(
                messages=chat_history.chat(), model=os.getenv("LLM_MODEL")
            )
            .choices[0]
            .message.content
        )
        if llm_response is None:
            raise ReactAgentError("LLM returned empty content")
        return llm_response

    def _add_observation(self, chat_history: ChatHistory, message: str) -> None:
        """
        Appends an observation message to chat history for the ReAct loop.

        Parameters:
        -----------
        chat_history: conversation history to update
        message: error or tool output message to feed back to the LLM
        """
        chat_history.add_chat(role="user", prompt=f"<observation>{message}</observation>")

    def invoke(self, user_query):
        chat_history = self._create_chat_history()
        chat_history.add_chat(role="user", prompt=f"<question>{user_query}</question>")

        if not self.tools:
            return self._call_llm(chat_history)

        tools_dict = self._create_tool_dict()

        for _ in range(self.max_iterations):
            llm_response = self._call_llm(chat_history)

            print(llm_response)

            if "<response>" in llm_response:
                response = self._extract_tag(llm_response, tag="response")
                if response is not None:
                    return response
                self._add_observation(
                    chat_history,
                    "Malformed <response> tag. Enclose your final answer in "
                    "<response>...</response> tags.",
                )
                continue

            chat_history.add_chat(role="assistant", prompt=llm_response)

            tool_call_raw = self._extract_tag(llm_response, tag="tool_call")
            if tool_call_raw is None:
                self._add_observation(
                    chat_history,
                    "No valid <tool_call> found. Call a tool or respond with "
                    "<response>...</response> tags.",
                )
                continue

            try:
                tool_call = json.loads(tool_call_raw)
            except json.JSONDecodeError as exc:
                self._add_observation(
                    chat_history,
                    f"Invalid JSON in <tool_call>: {exc}. "
                    'Expected: {"name": "<tool>", "arguments": {...}, "id": <int>}',
                )
                continue

            tool_name = tool_call.get("name")
            if not tool_name:
                self._add_observation(
                    chat_history,
                    "Tool call missing required 'name' field.",
                )
                continue

            if tool_name not in tools_dict:
                available = ", ".join(tools_dict) or "none"
                self._add_observation(
                    chat_history,
                    f"Unknown tool '{tool_name}'. Available tools: {available}.",
                )
                continue

            arguments = tool_call.get("arguments")
            if not isinstance(arguments, dict):
                self._add_observation(
                    chat_history,
                    "Tool call missing required 'arguments' dict.",
                )
                continue

            try:
                tool_output = tools_dict[tool_name].run(**arguments)
            except Exception as exc:
                self._add_observation(
                    chat_history,
                    f"Error executing tool '{tool_name}': {exc}",
                )
                continue

            self._add_observation(chat_history, str(tool_output))

        raise ReactAgentMaxIterationsError(
            f"ReAct loop did not produce a <response> within {self.max_iterations} iterations"
        )
