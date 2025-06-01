import os
from pydantic import BaseModel
from typing import Any
import json
import re
from openai import OpenAI
from squadAI.tools import Tool
from squadAI.chat import ChatHistory
from groq import Groq
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
    _parse_tool_calling: returns the tools and its argumments for calling based on llm response
    invoke: this methods takes the user query and take action accordingly to generate and return the response
    """

    tools: list[Tool] = []
    prompt: str
    max_iterations: int = 4
    # client: Any = Groq()
    client: Any = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
    )

    def _create_tool_dict(self):
        if self.tools:
            tools_dict = {tool.function_name: tool for tool in self.tools}

        return tools_dict

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

    def _parse_tool_calling(self, output: str, tag: str):
        pattern = rf"<{tag}>(.*?)</{tag}>"
        clean_output = re.findall(pattern, output, re.DOTALL)
        return clean_output[0]

    def invoke(self, user_query):
        chat_history = self._create_chat_history()
        chat_history.add_chat(role="user", prompt=f"<question>{user_query}</question>")

        if self.tools:
            for _ in range(self.max_iterations):

                llm_response = (
                    self.client.chat.completions.create(
                        messages=chat_history.chat(), model=os.getenv("LLM_MODEL")
                    )
                    .choices[0]
                    .message.content
                )

                print(llm_response)
                if "<response>" in llm_response:
                    return self._parse_tool_calling(llm_response, tag="response")

                chat_history.add_chat(role="assistant", prompt=llm_response)
                if self.tools:
                    tools_dict = self._create_tool_dict()
                    print(llm_response)
                    output = self._parse_tool_calling(llm_response, tag="tool_call")
                    print(output)
                    output = json.loads(output)

                    tool_output = tools_dict[output["name"]].run(**output["arguments"])

                    chat_history.add_chat(
                        "user", prompt=f"<observation>{tool_output}</observation>"
                    )
        else:
            llm_response = (
                self.client.chat.completions.create(
                    messages=chat_history.chat(), model=os.getenv("LLM_MODEL")
                )
                .choices[0]
                .message.content
            )

        return llm_response
