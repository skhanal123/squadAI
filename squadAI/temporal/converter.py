"""Convert in-memory squads to Temporal workflow payloads."""

from __future__ import annotations

from squadAI.providers.mock import MockProvider
from squadAI.squadAgent import SquadAgents
from squadAI.temporal.models import AgentSpec, SquadWorkflowInput, TaskSpec
from squadAI.temporal.schedule import execution_levels
from squadAI.usage import resolve_model_name


def _base_agent_spec(agent) -> AgentSpec:
    provider_mode = "env"
    if agent.provider is not None and isinstance(agent.provider, MockProvider):
        provider_mode = "mock"

    return AgentSpec(
        backstory=agent.backstory or "",
        tool_names=[tool.function_name for tool in agent.tools],
        max_iterations=agent.max_iterations,
        provider=provider_mode,
        model=resolve_model_name(agent.provider),
        mock_response=None,
    )


def squad_to_workflow_input(squad: SquadAgents, **kwargs) -> SquadWorkflowInput:
    """Serialize a :class:`SquadAgents` instance for Temporal execution."""
    task_by_id = {str(task.id): task for task in squad.tasks}
    specs = [
        TaskSpec(
            task_id=str(task.id),
            description=task.task_description,
            dependency_ids=[str(dep.id) for dep in task.dependency],
            task_output=task.task_output,
            agent=_base_agent_spec(task.agent),
        )
        for task in squad.tasks
    ]

    mock_queues: dict[int, list] = {}
    for level in execution_levels(specs):
        for spec in level:
            task = task_by_id[spec.task_id]
            if not isinstance(task.agent.provider, MockProvider):
                continue

            agent_key = id(task.agent)
            if agent_key not in mock_queues:
                mock_queues[agent_key] = list(task.agent.provider.responses)
            queue = mock_queues[agent_key]
            if queue:
                spec.agent.mock_response = queue.pop(0).content

    return SquadWorkflowInput(tasks=specs, run_kwargs=dict(kwargs))
