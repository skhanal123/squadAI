from uuid import UUID

from pydantic import BaseModel, Field, InstanceOf, UUID4, model_validator

from squadAI.createAgent import Agent
from squadAI.task import Task


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
    tasks: list of tasks to be completed. ``run()`` executes in list order;
        use ``run_temporal()`` for durable DAG orchestration via Temporal.

    Methods:
    validate_task_dependencies: validates the dependency graph at construction
    run: executes tasks sequentially in-process and returns a :class:`SquadResult`
    run_temporal: executes tasks via a Temporal workflow (supports parallel branches)
    """

    agents: list[InstanceOf[Agent]] = []
    tasks: list[InstanceOf[Task]] = []

    @model_validator(mode="after")
    def validate_task_dependencies(self):
        """
        Validates the task dependency graph when a SquadAgents instance is created.

        Ensures every dependency is included in the squad tasks list and that the
        graph has no cycles. List order is only required for :meth:`run`; Temporal
        execution resolves order from the DAG.

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

    def _validate_list_order(self) -> None:
        """Ensure dependencies appear before dependents for in-process execution."""
        task_index = {task.id: index for index, task in enumerate(self.tasks)}

        for index, task in enumerate(self.tasks):
            for dependency in task.dependency:
                dep_index = task_index[dependency.id]
                if dep_index >= index:
                    raise ValueError(
                        f"Task at index {index} ({task.task_description!r}) depends on "
                        f"task at index {dep_index} ({dependency.task_description!r}); "
                        f"dependencies must appear earlier in the tasks list for run()"
                    )

    def run(self, **kwargs) -> SquadResult:
        """Run the squad in-process, sequentially in task list order."""
        if not self.tasks:
            return SquadResult()

        self._validate_list_order()

        context_lookup: dict[UUID, str] = {}
        task_results: list[TaskResult] = []
        task_output: str | None = None

        for task in self.tasks:
            if task.dependency:
                task_context = build_task_context(task.dependency, context_lookup)
                task_output = task.agent.run(task, context=task_context, **kwargs)
            else:
                task_output = task.agent.run(task, **kwargs)

            context_lookup[task.id] = task_output
            task_results.append(
                TaskResult(
                    task_id=task.id,
                    description=task.task_description,
                    output=task_output,
                )
            )

        return SquadResult(final=task_output, task_results=task_results)

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
