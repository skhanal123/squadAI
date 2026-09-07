"""Temporal activities for squad task execution."""

from __future__ import annotations

from types import SimpleNamespace

from temporalio import activity

from squadAI.createAgent import Agent
from squadAI.llm import create_provider
from squadAI.providers.base import LLMResponse
from squadAI.providers.mock import MockProvider
from squadAI.temporal.models import TaskActivityInput, TaskActivityOutput
from squadAI.temporal.registry import get_tools


def _build_agent(agent_spec) -> Agent:
    if agent_spec.provider == "mock":
        responses = []
        if agent_spec.mock_response is not None:
            responses = [LLMResponse(content=agent_spec.mock_response)]
        provider = MockProvider(responses)
    else:
        provider = create_provider()

    return Agent(
        backstory=agent_spec.backstory,
        tools=get_tools(agent_spec.tool_names),
        max_iterations=agent_spec.max_iterations,
        provider=provider,
    )


@activity.defn(name="execute_squad_task")
async def execute_squad_task(input: TaskActivityInput) -> TaskActivityOutput:
    """Run one squad task inside a Temporal activity."""
    agent = _build_agent(input.agent)
    task = SimpleNamespace(
        task_description=input.description,
        task_output=input.task_output,
        output_schema=None,
    )
    if input.output_json_schema is not None:
        task.get_output_json_schema = lambda: input.output_json_schema
    else:
        task.get_output_json_schema = lambda: None
    run_result = agent.run(task, context=input.context or None, **input.run_kwargs)
    usage = run_result.task_usage
    return TaskActivityOutput(
        task_id=input.task_id,
        description=input.description,
        output=run_result.output,
        model=usage.model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )
