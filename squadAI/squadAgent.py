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
    """Structured result from ``SquadAgents.run()``."""

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
    tasks: list of tasks to be completed. Tasks will be executed in the same order provided in the list

    Methods:
    validate_task_dependency_order: validates that task dependencies are present and ordered correctly
    run: executes tasks sequentially and returns a :class:`SquadResult`
    """

    agents: list[InstanceOf[Agent]] = []
    tasks: list[InstanceOf[Task]] = []

    @model_validator(mode="after")
    def validate_task_dependency_order(self):
        """
        Validates task dependencies when a SquadAgents instance is created.

        Ensures every dependency is included in the squad tasks list and appears
        earlier in the list than the task that depends on it.

        Returns:
        --------
        self: the validated SquadAgents instance

        Raises:
        -------
        ValueError: if a dependency is missing from the tasks list or appears at
            the same index or after its dependent task
        """
        if not self.tasks:
            return self

        task_index = {task.id: index for index, task in enumerate(self.tasks)}

        for index, task in enumerate(self.tasks):
            for dependency in task.dependency:
                dep_index = task_index.get(dependency.id)
                if dep_index is None:
                    raise ValueError(
                        f"Task at index {index} depends on a task that is not "
                        f"included in the squad tasks list: {dependency.task_description!r}"
                    )
                if dep_index >= index:
                    raise ValueError(
                        f"Task at index {index} ({task.task_description!r}) depends on "
                        f"task at index {dep_index} ({dependency.task_description!r}); "
                        f"dependencies must appear earlier in the tasks list"
                    )

        return self

    def run(self, **kwargs) -> SquadResult:
        if not self.tasks:
            return SquadResult()

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

