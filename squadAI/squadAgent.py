from uuid import UUID
import asyncio
from dataclasses import replace
from pathlib import Path

from pydantic import BaseModel, Field, InstanceOf, UUID4, model_validator

from squadAI.config import get_settings
from squadAI.createAgent import Agent
from squadAI.task import Task
from squadAI.usage import (
    AgentRunResult,
    TaskUsage,
    TokenUsage,
    format_usage_display,
    merge_usage_by_model,
    sum_task_usages,
)
from squadAI.output_schema import format_context_block, normalize_task_output, TaskOutputParseError
from squadAI.trace import (
    AgentRunTrace,
    GateRoundRecord,
    TaskAttemptRecord,
    TaskExecutionTrace,
    ValidationRecord,
    build_squad_trace_payload,
    write_trace_file,
)
from squadAI.validation import TaskValidationError, call_task_validator


def build_task_context(
    dependencies: list[Task],
    context_lookup: dict[UUID, str],
) -> str:
    """Build labeled context blocks from upstream task outputs."""
    blocks = []
    for index, dependency in enumerate(dependencies, start=1):
        blocks.append(
            format_context_block(
                index=index,
                description=dependency.task_description,
                output=context_lookup[dependency.id],
            )
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


def _attempt_from_run_result(
    run_result: AgentRunResult,
    attempt: int,
    *,
    validation: ValidationRecord | None = None,
) -> TaskAttemptRecord:
    return TaskAttemptRecord(
        attempt=attempt,
        agent=run_result.trace or AgentRunTrace(),
        validation=validation,
        input_tokens=run_result.usage.input_tokens,
        output_tokens=run_result.usage.output_tokens,
    )


def _single_run_trace(run_result: AgentRunResult) -> TaskExecutionTrace:
    return TaskExecutionTrace(
        status="success",
        attempts=[_attempt_from_run_result(run_result, attempt=0)],
    )


def _pop_run_options(kwargs: dict) -> tuple[bool, Path | None, dict]:
    include_usage = kwargs.pop("include_usage_in_result", None)
    if include_usage is None:
        include_usage = get_settings().include_usage_in_result
    trace_output_dir = kwargs.pop("trace_output_dir", None)
    trace_dir = Path(trace_output_dir) if trace_output_dir is not None else None
    return bool(include_usage), trace_dir, kwargs


def _maybe_write_trace(result: "SquadResult", trace_output_dir: Path | None) -> None:
    if trace_output_dir is not None:
        result.write_trace(trace_output_dir)


def _build_squad_result(
    task_results: list["TaskResult"],
    final_output: str | None,
    *,
    include_usage_in_result: bool,
) -> "SquadResult":
    usage = sum_task_usages(task_results)
    usage_by_model = merge_usage_by_model(task_results)
    usage_display = (
        format_usage_display(task_results) if include_usage_in_result else None
    )
    return SquadResult(
        final=final_output,
        task_results=task_results,
        usage=usage,
        usage_by_model=usage_by_model,
        usage_display=usage_display,
    )


class TaskResult(BaseModel):
    """Output of a single task within a squad run."""

    task_id: UUID4
    description: str
    output: str
    usage: TaskUsage
    trace: TaskExecutionTrace = Field(default_factory=TaskExecutionTrace)


class SquadResult(BaseModel):
    """Structured result from ``SquadAgents.run()`` or ``SquadAgents.run_temporal()``."""

    final: str | None = None
    task_results: list[TaskResult] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    usage_by_model: dict[str, TokenUsage] = Field(default_factory=dict)
    usage_display: str | None = None

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

    def get_usage(self, task: Task) -> TaskUsage | None:
        """Return token usage for a specific task, or ``None`` if not found."""
        for result in self.task_results:
            if result.task_id == task.id:
                return result.usage
        return None

    def get_trace(self, task: Task) -> TaskExecutionTrace | None:
        """Return the execution trace for a specific task, or ``None`` if not found."""
        for result in self.task_results:
            if result.task_id == task.id:
                return result.trace
        return None

    def trace_payload(self) -> dict:
        """Return a JSON-serializable execution trace for all tasks."""
        return build_squad_trace_payload(self.task_results)

    def write_trace(
        self,
        path: Path | str,
        *,
        filename: str | None = None,
    ) -> Path:
        """Persist the squad execution trace under ``path``.

        When ``path`` is a directory, writes ``execution_trace.json`` inside it.
        When ``path`` ends with ``.json``, writes to that file directly.
        """
        from squadAI.trace import DEFAULT_TRACE_FILENAME

        return write_trace_file(
            self.trace_payload(),
            path,
            filename=filename or DEFAULT_TRACE_FILENAME,
        )


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
        Pass ``trace_output_dir`` to persist ``execution_trace.json`` after a run.

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
    ) -> AgentRunResult:
        if task.dependency:
            task_context = build_task_context(task.dependency, context_lookup)
        else:
            task_context = None

        task_context = append_validation_feedback(task_context, validation_feedback)
        run_result = await task.agent.run_async(task, context=task_context, **kwargs)
        if task.output_schema is not None:
            try:
                normalized = normalize_task_output(
                    run_result.output,
                    model=task.output_schema,
                )
            except TaskOutputParseError as exc:
                raise TaskValidationError(
                    task,
                    run_result.output,
                    str(exc),
                ) from exc
            run_result = replace(run_result, output=normalized)
        return run_result

    async def _run_self_validated(
        self,
        task: Task,
        context_lookup: dict[UUID, str],
        *,
        initial_feedback: str | None = None,
        **kwargs,
    ) -> tuple[str, TaskUsage, TaskExecutionTrace]:
        """Stage 1: validate this task's output and retry with feedback."""
        feedback = initial_feedback
        last_output = ""
        total_usage: TaskUsage | None = None
        trace = TaskExecutionTrace()
        attempt_records: list[TaskAttemptRecord] = []

        for attempt in range(task.max_retries + 1):
            run_result = await self._invoke_agent(
                task,
                context_lookup,
                validation_feedback=feedback,
                **kwargs,
            )
            if total_usage is None:
                total_usage = run_result.task_usage
            else:
                total_usage = total_usage + run_result.task_usage
            last_output = run_result.output

            validation = call_task_validator(task, output=last_output)
            validation_record = ValidationRecord(
                approved=validation.approved,
                feedback=validation.feedback,
            )
            attempt_records.append(
                _attempt_from_run_result(
                    run_result,
                    attempt,
                    validation=validation_record,
                )
            )
            if validation.approved:
                trace.status = "success"
                trace.attempts = attempt_records
                return last_output, total_usage, trace

            feedback = validation.feedback or "Validation failed."
            if attempt >= task.max_retries:
                trace.status = "validation_failed"
                trace.attempts = attempt_records
                raise TaskValidationError(task, last_output, feedback)

        trace.status = "success"
        trace.attempts = attempt_records
        return last_output, total_usage or TaskUsage.from_tokens(
            run_result.model, TokenUsage()
        ), trace

    async def _run_validation_gate(
        self,
        gate_task: Task,
        context_lookup: dict[UUID, str],
        **kwargs,
    ) -> tuple[str, str, TaskUsage, TaskUsage, TaskExecutionTrace, TaskExecutionTrace]:
        """Stage 2: run an upstream task and gate task until validation passes."""
        target = gate_task.validates
        agent_gate = gate_task.agent is not None
        target_feedback: str | None = None
        last_target_output = ""
        last_gate_output = ""
        target_usage: TaskUsage | None = None
        gate_usage: TaskUsage | None = None
        target_trace = TaskExecutionTrace(validated_by_gate=True)
        gate_trace = TaskExecutionTrace(
            is_validation_gate=True,
            validates_task_id=str(target.id),
        )
        gate_rounds: list[GateRoundRecord] = []
        empty_gate_usage = TaskUsage.from_tokens("", TokenUsage())
        gate_result: AgentRunResult | None = None

        for attempt in range(gate_task.max_retries + 1):
            last_target_output, attempt_target_usage, upstream_trace = (
                await self._execute_task(
                    target,
                    context_lookup,
                    validation_feedback=target_feedback,
                    **kwargs,
                )
            )
            if target_usage is None:
                target_usage = attempt_target_usage
            else:
                target_usage = target_usage + attempt_target_usage
            context_lookup[target.id] = last_target_output

            if agent_gate:
                gate_result = await self._invoke_agent(
                    gate_task, context_lookup, **kwargs
                )
                if gate_usage is None:
                    gate_usage = gate_result.task_usage
                else:
                    gate_usage = gate_usage + gate_result.task_usage
                last_gate_output = gate_result.output
            else:
                last_gate_output = last_target_output

            validation = call_task_validator(
                gate_task,
                output=last_gate_output if agent_gate else "",
                upstream_output=last_target_output,
            )
            validation_record = ValidationRecord(
                approved=validation.approved,
                feedback=validation.feedback,
            )
            gate_rounds.append(
                GateRoundRecord(
                    round=attempt,
                    upstream_attempts=list(upstream_trace.attempts),
                    gate_agent=(
                        (gate_result.trace or AgentRunTrace()) if agent_gate else None
                    ),
                    gate_skipped=not agent_gate,
                    validation=validation_record,
                )
            )
            if validation.approved:
                target_trace.status = "success"
                target_trace.attempts = list(upstream_trace.attempts)
                gate_trace.status = "success"
                gate_trace.gate_rounds = gate_rounds
                gate_trace.attempts = (
                    [_attempt_from_run_result(gate_result, attempt=0)]
                    if agent_gate
                    else []
                )
                return (
                    last_gate_output,
                    last_target_output,
                    gate_usage or empty_gate_usage,
                    target_usage,
                    gate_trace,
                    target_trace,
                )

            target_feedback = validation.feedback or "Validation failed."
            if attempt >= gate_task.max_retries:
                target_trace.status = "validation_failed"
                target_trace.attempts = list(upstream_trace.attempts)
                gate_trace.status = "validation_failed"
                gate_trace.gate_rounds = gate_rounds
                failed_output = (
                    last_target_output if not agent_gate else last_gate_output
                )
                raise TaskValidationError(gate_task, failed_output, target_feedback)

        target_trace.attempts = list(upstream_trace.attempts)
        gate_trace.gate_rounds = gate_rounds
        return (
            last_gate_output,
            last_target_output,
            gate_usage or empty_gate_usage,
            target_usage or attempt_target_usage,
            gate_trace,
            target_trace,
        )

    async def _execute_task(
        self,
        task: Task,
        context_lookup: dict[UUID, str],
        *,
        validation_feedback: str | None = None,
        **kwargs,
    ) -> tuple[str, TaskUsage, TaskExecutionTrace]:
        if task.validates is not None:
            raise ValueError(
                f"Task {task.task_description!r} is a validation gate and must be "
                f"executed via run_async(), not as a standalone task"
            )

        if task.validator is None:
            run_result = await self._invoke_agent(
                task,
                context_lookup,
                validation_feedback=validation_feedback,
                **kwargs,
            )
            return run_result.output, run_result.task_usage, _single_run_trace(run_result)

        output, usage, trace = await self._run_self_validated(
            task,
            context_lookup,
            initial_feedback=validation_feedback,
            **kwargs,
        )
        return output, usage, trace

    async def run_async(self, **kwargs) -> SquadResult:
        """Run the squad asynchronously by DAG level with parallel branches."""
        include_usage_in_result, trace_output_dir, run_kwargs = _pop_run_options(
            dict(kwargs)
        )

        if not self.tasks:
            return SquadResult()

        from squadAI.temporal.schedule import execution_levels

        validated_targets = self._validated_targets()
        workflow_input = self.to_workflow_input(**run_kwargs)
        levels = execution_levels(workflow_input.tasks)
        task_by_id = {str(task.id): task for task in self.tasks}
        task_index = {str(task.id): index for index, task in enumerate(self.tasks)}

        context_lookup: dict[UUID, str] = {}
        task_results: list[TaskResult] = []

        for level_index, level in enumerate(levels):
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
                (
                    gate_output,
                    target_output,
                    gate_usage,
                    target_usage,
                    gate_trace,
                    target_trace,
                ) = await self._run_validation_gate(
                    gate_task,
                    context_lookup,
                    **run_kwargs,
                )
                target = gate_task.validates
                gate_trace.dag_level = level_index
                target_trace.dag_level = level_index
                context_lookup[target.id] = target_output
                context_lookup[gate_task.id] = gate_output
                task_results.append(
                    TaskResult(
                        task_id=target.id,
                        description=target.task_description,
                        output=target_output,
                        usage=target_usage,
                        trace=target_trace,
                    )
                )
                task_results.append(
                    TaskResult(
                        task_id=gate_task.id,
                        description=gate_task.task_description,
                        output=gate_output,
                        usage=gate_usage,
                        trace=gate_trace,
                    )
                )

            if parallel_tasks:
                results = await asyncio.gather(
                    *[
                        self._execute_task(task, context_lookup, **run_kwargs)
                        for task in parallel_tasks
                    ]
                )
                ran_in_parallel = len(parallel_tasks) > 1
                for task, (output, usage, trace) in zip(parallel_tasks, results):
                    trace.dag_level = level_index
                    trace.parallel = ran_in_parallel
                    context_lookup[task.id] = output
                    task_results.append(
                        TaskResult(
                            task_id=task.id,
                            description=task.task_description,
                            output=output,
                            usage=usage,
                            trace=trace,
                        )
                    )

        final_output: str | None = None
        if levels:
            last_spec = max(
                levels[-1],
                key=lambda spec: task_index[spec.task_id],
            )
            final_output = context_lookup.get(task_by_id[last_spec.task_id].id)

        result = _build_squad_result(
            task_results,
            final_output,
            include_usage_in_result=include_usage_in_result,
        )
        _maybe_write_trace(result, trace_output_dir)
        return result

    def run(self, **kwargs) -> SquadResult:
        """Run the squad synchronously (wrapper around :meth:`run_async`)."""
        return asyncio.run(self.run_async(**kwargs))

    def to_workflow_input(self, **kwargs):
        """Build a serializable Temporal workflow input from this squad."""
        from squadAI.temporal.converter import squad_to_workflow_input

        return squad_to_workflow_input(self, **kwargs)

    def _has_validation_gates(self) -> bool:
        return any(task.validates is not None for task in self.tasks)

    async def run_temporal(
        self,
        client,
        *,
        task_queue: str = "squadai",
        workflow_id: str | None = None,
        **kwargs,
    ) -> SquadResult:
        """Execute the squad as a durable Temporal workflow."""
        if self._has_validation_gates():
            raise ValueError(
                "Validation gates require run() or run_async(); "
                "run_temporal() does not support them yet."
            )
        from squadAI.temporal.client import execute_squad_workflow

        include_usage_in_result, trace_output_dir, run_kwargs = _pop_run_options(
            dict(kwargs)
        )
        result = await execute_squad_workflow(
            client,
            self.to_workflow_input(**run_kwargs),
            task_queue=task_queue,
            workflow_id=workflow_id,
        )
        if include_usage_in_result and result.usage_display is None:
            result = result.model_copy(
                update={"usage_display": format_usage_display(result.task_results)}
            )
        _maybe_write_trace(result, trace_output_dir)
        return result
