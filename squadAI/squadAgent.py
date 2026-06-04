from uuid import UUID
import asyncio

from pydantic import BaseModel, Field, InstanceOf, UUID4, model_validator

from squadAI.createAgent import Agent
from squadAI.task import Task
from squadAI.validation import TaskValidationError, call_task_validator


def build_task_context(
    dependencies: list[Task],
    context_lookup: dict[UUID, str],
) -> str:
    """Build labeled context blocks from upstream task outputs."""
    blocks = []
    for index, dependency in enumerate(dependencies, start=1):
        blocks.append(
            f'<upstream_task index="{index}">\n'
            f"<description>{dependency.task_description}</description>\n"
            f"<output>{context_lookup[dependency.id]}</output>\n"
            f"</upstream_task>"
        )
    return "\n\n".join(blocks)


def append_validation_feedback(context: str | None, feedback: str | None) -> str | None:
    """Append validation feedback to an existing task context block."""
    if not feedback:
        return context or None

    feedback_block = (
        f"<validation_feedback>\n{feedback}\n</validation_feedback>"
    )
    if context:
        return f"{context}\n\n{feedback_block}"
    return feedback_block


class TaskResult(BaseModel):
    """Output of a single task within a squad run."""

    task_id: UUID4
    description: str
    output: str


class SquadResult(BaseModel):
    """Structured result from ``SquadAgents.run()`` or ``SquadAgents.run_temporal()``."""

    final: str | None = None
    task_results: list[TaskResult] = Field(default_factory=list)

    @property
    def outputs(self) -> dict[UUID, str]:
        """Map each task id to its output string."""
        return {result.task_id: result.output for result in self.task_results}

    def get(self, task: Task) -> str | None:
        """Return the output for a specific task, or ``None`` if not found."""
        for result in self.task_results:
            if result.task_id == task.id:
                return result.output
        return None


class SquadAgents(BaseModel):
    """
    This is the base class to create the squad of agents to perform the list of tasks.

    Attributes:
    -----------
    agents: list of instances of agents to perform various task
    tasks: list of tasks to be completed. ``run()`` and ``run_async()`` execute
        tasks by DAG level; independent tasks in the same level run in parallel.
        Use ``run_temporal()`` for durable orchestration via Temporal.
        Optional per-task validators support bounded self-retries and
        upstream validation gates.

    Methods:
    validate_task_dependencies: validates the dependency graph at construction
    run: sync wrapper around :meth:`run_async`
    run_async: async DAG orchestration with parallel independent tasks
    run_temporal: executes tasks via a Temporal workflow (supports parallel branches)
    """

    agents: list[InstanceOf[Agent]] = []
    tasks: list[InstanceOf[Task]] = []

    @model_validator(mode="after")
    def validate_task_dependencies(self):
        """
        Validates the task dependency graph when a SquadAgents instance is created.

        Ensures every dependency is included in the squad tasks list and that the
        graph has no cycles. Task list order is not required for :meth:`run` or
        :meth:`run_async`; execution resolves order from the DAG.

        Raises:
        -------
        ValueError: if a dependency is missing or the graph contains a cycle
        """
        if not self.tasks:
            return self

        task_ids = {task.id for task in self.tasks}

        for index, task in enumerate(self.tasks):
            for dependency in task.dependency:
                if dependency.id not in task_ids:
                    raise ValueError(
                        f"Task at index {index} depends on a task that is not "
                        f"included in the squad tasks list: {dependency.task_description!r}"
                    )

        from squadAI.temporal.schedule import validate_task_specs

        validate_task_specs(self.to_workflow_input().tasks)
        return self

    def _validated_targets(self) -> dict[UUID, Task]:
        validated_targets = {
            task.validates.id: task
            for task in self.tasks
            if task.validates is not None
        }
        if len(validated_targets) != sum(1 for task in self.tasks if task.validates):
            raise ValueError("Each upstream task may have only one validation gate")
        return validated_targets

    async def _invoke_agent(
        self,
        task: Task,
        context_lookup: dict[UUID, str],
        *,
        validation_feedback: str | None = None,
        **kwargs,
    ) -> str:
        if task.dependency:
            task_context = build_task_context(task.dependency, context_lookup)
        else:
            task_context = None

        task_context = append_validation_feedback(task_context, validation_feedback)
        return await task.agent.run_async(task, context=task_context, **kwargs)

    async def _run_self_validated(
        self,
        task: Task,
        context_lookup: dict[UUID, str],
        *,
        initial_feedback: str | None = None,
        **kwargs,
    ) -> str:
        """Stage 1: validate this task's output and retry with feedback."""
        feedback = initial_feedback
        last_output = ""

        for attempt in range(task.max_retries + 1):
            last_output = await self._invoke_agent(
                task,
                context_lookup,
                validation_feedback=feedback,
                **kwargs,
            )
            result = call_task_validator(task, output=last_output, **kwargs)
            if result.approved:
                return last_output

            feedback = result.feedback or "Validation failed."
            if attempt >= task.max_retries:
                raise TaskValidationError(task, last_output, feedback)

        return last_output

    async def _run_validation_gate(
        self,
        gate_task: Task,
        context_lookup: dict[UUID, str],
        **kwargs,
    ) -> tuple[str, str]:
        """Stage 2: run an upstream task and gate task until validation passes."""
        target = gate_task.validates
        target_feedback: str | None = None
        last_target_output = ""
        last_gate_output = ""

        for attempt in range(gate_task.max_retries + 1):
            last_target_output = await self._execute_task(
                target,
                context_lookup,
                validation_feedback=target_feedback,
                **kwargs,
            )
            context_lookup[target.id] = last_target_output

            last_gate_output = await self._invoke_agent(gate_task, context_lookup, **kwargs)
            result = call_task_validator(
                gate_task,
                output=last_gate_output,
                upstream_output=last_target_output,
                **kwargs,
            )
            if result.approved:
                return last_gate_output, last_target_output

            target_feedback = result.feedback or "Validation failed."
            if attempt >= gate_task.max_retries:
                raise TaskValidationError(gate_task, last_gate_output, target_feedback)

        return last_gate_output, last_target_output

    async def _execute_task(
        self,
        task: Task,
        context_lookup: dict[UUID, str],
        *,
        validation_feedback: str | None = None,
        **kwargs,
    ) -> str:
        if task.validates is not None:
            raise ValueError(
                f"Task {task.task_description!r} is a validation gate and must be "
                f"executed via run_async(), not as a standalone task"
            )

        if task.validator is None:
            return await self._invoke_agent(
                task,
                context_lookup,
                validation_feedback=validation_feedback,
                **kwargs,
            )

        return await self._run_self_validated(
            task,
            context_lookup,
            initial_feedback=validation_feedback,
            **kwargs,
        )

    async def run_async(self, **kwargs) -> SquadResult:
        """Run the squad asynchronously by DAG level with parallel branches."""
        if not self.tasks:
            return SquadResult()

        from squadAI.temporal.schedule import execution_levels

        validated_targets = self._validated_targets()
        workflow_input = self.to_workflow_input(**kwargs)
        levels = execution_levels(workflow_input.tasks)
        task_by_id = {str(task.id): task for task in self.tasks}
        task_index = {str(task.id): index for index, task in enumerate(self.tasks)}

        context_lookup: dict[UUID, str] = {}
        task_results: list[TaskResult] = []

        for level in levels:
            gate_tasks: list[Task] = []
            parallel_tasks: list[Task] = []

            for spec in level:
                task = task_by_id[spec.task_id]
                if task.id in validated_targets:
                    continue
                if task.validates is not None:
                    gate_tasks.append(task)
                else:
                    parallel_tasks.append(task)

            for gate_task in gate_tasks:
                gate_output, target_output = await self._run_validation_gate(
                    gate_task,
                    context_lookup,
                    **kwargs,
                )
                target = gate_task.validates
                context_lookup[target.id] = target_output
                context_lookup[gate_task.id] = gate_output
                task_results.append(
                    TaskResult(
                        task_id=target.id,
                        description=target.task_description,
                        output=target_output,
                    )
                )
                task_results.append(
                    TaskResult(
                        task_id=gate_task.id,
                        description=gate_task.task_description,
                        output=gate_output,
                    )
                )

            if parallel_tasks:
                outputs = await asyncio.gather(
                    *[
                        self._execute_task(task, context_lookup, **kwargs)
                        for task in parallel_tasks
                    ]
                )
                for task, output in zip(parallel_tasks, outputs):
                    context_lookup[task.id] = output
                    task_results.append(
                        TaskResult(
                            task_id=task.id,
                            description=task.task_description,
                            output=output,
                        )
                    )

        final_output: str | None = None
        if levels:
            last_spec = max(
                levels[-1],
                key=lambda spec: task_index[spec.task_id],
            )
            final_output = context_lookup.get(task_by_id[last_spec.task_id].id)

        return SquadResult(final=final_output, task_results=task_results)

    def run(self, **kwargs) -> SquadResult:
        """Run the squad synchronously (wrapper around :meth:`run_async`)."""
        return asyncio.run(self.run_async(**kwargs))

    def to_workflow_input(self, **kwargs):
        """Build a serializable Temporal workflow input from this squad."""
        from squadAI.temporal.converter import squad_to_workflow_input

        return squad_to_workflow_input(self, **kwargs)

    async def run_temporal(
        self,
        client,
        *,
        task_queue: str = "squadai",
        workflow_id: str | None = None,
        **kwargs,
    ) -> SquadResult:
        """Execute the squad as a durable Temporal workflow."""
        from squadAI.temporal.client import execute_squad_workflow

        return await execute_squad_workflow(
            client,
            self.to_workflow_input(**kwargs),
            task_queue=task_queue,
            workflow_id=workflow_id,
        )
