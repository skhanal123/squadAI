"""Temporal workflow for durable squad orchestration."""

from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from squadAI.temporal.models import (
        SquadWorkflowInput,
        SquadWorkflowResult,
        TaskActivityInput,
        TaskActivityOutput,
    )
    from squadAI.temporal.schedule import (
        build_task_context_from_specs,
        execution_levels,
    )

EXECUTE_SQUAD_TASK = "execute_squad_task"


def _as_task_output(raw) -> TaskActivityOutput:
    if isinstance(raw, TaskActivityOutput):
        return raw
    return TaskActivityOutput(
        task_id=raw["task_id"],
        description=raw["description"],
        output=raw["output"],
        model=raw.get("model", ""),
        input_tokens=int(raw.get("input_tokens", 0)),
        output_tokens=int(raw.get("output_tokens", 0)),
        trace=raw.get("trace"),
    )


@workflow.defn(name="SquadWorkflow")
class SquadWorkflow:
    """Durable squad orchestrator with parallel execution per DAG level."""

    @workflow.run
    async def run(self, input: SquadWorkflowInput) -> SquadWorkflowResult:
        if not input.tasks:
            return SquadWorkflowResult(final=None, task_results=[])

        task_by_id = {task.task_id: task for task in input.tasks}
        task_index = {task.task_id: index for index, task in enumerate(input.tasks)}
        context_lookup: dict[str, str] = {}
        task_results: list[TaskActivityOutput] = []

        levels = execution_levels(input.tasks)

        for level_index, level in enumerate(levels):
            activity_inputs = []
            for task in level:
                context = build_task_context_from_specs(
                    task.dependency_ids,
                    task_by_id,
                    context_lookup,
                )
                activity_inputs.append(
                    TaskActivityInput(
                        task_id=task.task_id,
                        description=task.description,
                        task_output=task.task_output,
                        output_json_schema=task.output_json_schema,
                        run_kwargs=input.run_kwargs,
                        context=context,
                        agent=task.agent,
                    )
                )

            handles = [
                workflow.execute_activity(
                    EXECUTE_SQUAD_TASK,
                    activity_input,
                    start_to_close_timeout=timedelta(minutes=10),
                    retry_policy=RetryPolicy(maximum_attempts=3),
                )
                for activity_input in activity_inputs
            ]
            level_outputs = await asyncio.gather(*handles)
            ran_in_parallel = len(activity_inputs) > 1

            for raw_result in level_outputs:
                result = _as_task_output(raw_result)
                if result.trace is not None:
                    result.trace["dag_level"] = level_index
                    result.trace["parallel"] = ran_in_parallel
                context_lookup[result.task_id] = result.output
                task_results.append(result)

        final_output: str | None = None
        if levels:
            last_task = max(
                levels[-1],
                key=lambda task: task_index[task.task_id],
            )
            final_output = context_lookup.get(last_task.task_id)

        return SquadWorkflowResult(final=final_output, task_results=task_results)
