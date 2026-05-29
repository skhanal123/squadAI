## SquadAI

SquadAI is a lightweight Python framework for orchestrating multiple AI agents, tools, and dependent tasks. It helps you compose collaborative AI workflows where one task can consume the output of previous tasks.

## What this project does

SquadAI provides a simple abstraction layer around:

- **Agents**: role-based workers with backstory/prompts (`squadAI/createAgent.py`)
- **Tools**: Python functions wrapped as callable tools for agents (`squadAI/tools.py`)
- **Tasks**: structured units of work assigned to agents (`squadAI/task.py`)
- **Orchestration**: sequential task execution with dependency context passing (`squadAI/squadAgent.py`)
- **ReAct-style execution loop**: tool-calling interaction with an LLM (`squadAI/reactAgent.py`)

## Repository structure

```text
squadAI/
  chat.py         # chat history helper
  createAgent.py  # Agent model and execution entrypoint
  llm.py          # client factory helper
  reactAgent.py   # ReAct loop and tool-call parsing
  squadAgent.py   # multi-task orchestrator
  task.py         # Task model
  tools.py        # Tool class and decorator wrapper
  utils.py        # function signature utilities
squadRun.py       # runnable examples
requirements.txt  # Python dependencies
```

## Core features

1. **Agent creation** with reusable backstories/prompts
2. **Tool integration** from normal Python functions via `@tool_wrapper`
3. **Task automation** with parameterized task templates (e.g. `{a}`, `{b}`)
4. **Task dependency chaining** where downstream tasks consume upstream output
5. **Multi-agent orchestration** through `SquadAgents.run()`

## Installation

### Prerequisites

- Python 3.10+ recommended
- Access to an LLM provider compatible with the configured client

### Setup

```bash
git clone https://github.com/skhanal123/squadAI.git
cd squadAI
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment configuration

The runtime loads environment variables using `python-dotenv`.

Create a `.env` file in the project root:

```env
DEEPSEEK_API_KEY=your_api_key_here
LLM_MODEL=deepseek-chat
```

Notes:

- `ReactAgent` obtains its LLM client via `create_client()` in `squadAI/llm.py` (DeepSeek when `LLM_MODEL` starts with `deepseek`, otherwise Groq).
- `LLM_MODEL` is read at runtime when invoking chat completions and when resolving the default client.

## Quick start

You can run the included examples:

```bash
python squadRun.py
```

`squadRun.py` demonstrates:

- Wrapping Python functions (`add_two_numbers`, `multiply_two_numbers`) as tools
- Creating specialized agents with those tools
- Defining dependent tasks where task 2 consumes task 1 output
- Running a squad with `SquadAgents(...).run(a=2, b=3, c=4)`

## Minimal usage example

```python
from squadAI.tools import tool_wrapper
from squadAI.task import Task
from squadAI.createAgent import Agent
from squadAI.squadAgent import SquadAgents

@tool_wrapper
def add_two_numbers(first_number: float, second_number: float):
    """Add two numbers."""
    return first_number + second_number

math_agent = Agent(
    backstory="You are an expert in math.",
    tools=[add_two_numbers],
)

task = Task(
    task_description="Please add two numbers {a} and {b}",
    agent=math_agent,
)

squad = SquadAgents(agents=[math_agent], tasks=[task])
result = squad.run(a=2, b=3)
print(result)
```

## How orchestration works

1. `SquadAgents.run()` iterates through tasks in order.
2. If a task has dependencies, dependent task outputs are concatenated into context.
3. The assigned `Agent.run()` formats the task description with runtime kwargs.
4. `ReactAgent.invoke()` executes an LLM loop:
   - receives model output
   - parses `<tool_call>...</tool_call>` when needed
   - executes the mapped Python tool
   - feeds tool output back as `<observation>`
   - returns `<response>...</response>` content when available

## Current limitations and notes

- Task execution order is list-based and sequential.
- Dependency context is currently concatenated as plain text.
- Error handling for malformed tool-call responses can be expanded.
- The codebase is intentionally small and aimed at experimentation and learning.

## Development

To iterate locally:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python squadRun.py
```

If you add new tools, ensure function annotations and docstrings are present so signature extraction remains useful.