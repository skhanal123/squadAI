"""Temporal client helpers for squad execution."""

from __future__ import annotations

from uuid import UUID, uuid4

from squadAI.squadAgent import SquadResult, TaskResult
from squadAI.temporal.models import SquadWorkflowInput, SquadWorkflowResult, TaskActivityOutput
from squadAI.temporal.workflow import SquadWorkflow
from squadAI.usage import TaskUsage, merge_usage_by_model, sum_task_usages

DEFAULT_TASK_QUEUE = "squadai"
EXECUTE_SQUAD_TASK = "execute_squad_task"


def workflow_result_to_squad_result(result: SquadWorkflowResult | dict) -> SquadResult:
    """Convert a Temporal workflow result into a :class:`SquadResult`."""
    if isinstance(result, dict):
        task_results = []
        for task_result in result.get("task_results", []):
            if isinstance(task_result, dict):
                task_results.append(TaskActivityOutput(**task_result))
            else:
                task_results.append(task_result)
        result = SquadWorkflowResult(
            final=result.get("final"),
            task_results=task_results,
        )
    task_results = [
        TaskResult(
            task_id=UUID(task_result.task_id),
            description=task_result.description,
            output=task_result.output,
            usage=TaskUsage(
                model=task_result.model,
                input_tokens=task_result.input_tokens,
                output_tokens=task_result.output_tokens,
            ),
        )
        for task_result in result.task_results
    ]
    return SquadResult(
        final=result.final,
        task_results=task_results,
        usage=sum_task_usages(task_results),
        usage_by_model=merge_usage_by_model(task_results),
    )


async def execute_squad_workflow(
    client,
    workflow_input: SquadWorkflowInput,
    *,
    task_queue: str = DEFAULT_TASK_QUEUE,
    workflow_id: str | None = None,
) -> SquadResult:
    """Start ``SquadWorkflow`` and wait for the structured squad result."""
    result = await client.execute_workflow(
        SquadWorkflow.run,
        workflow_input,
        id=workflow_id or f"squad-{uuid4()}",
        task_queue=task_queue,
    )
    return workflow_result_to_squad_result(result)
