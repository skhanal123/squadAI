from pydantic import BaseModel, InstanceOf, model_validator
from squadAI.createAgent import Agent
from squadAI.task import Task


class SquadAgents(BaseModel):
    """
    This is the base class to create the squad of agents to perform the list of tasks.

    Attributes:
    -----------
    agents: list of instances of agents to perform various task
    tasks: list of tasks to be completed. Tasks will be executed in the same order provided in the list
    """

    agents: list[InstanceOf[Agent]] = []
    tasks: list[InstanceOf[Task]] = []

    @model_validator(mode="after")
    def validate_task_dependency_order(self):
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

    def run(self, **kwargs):
        if not self.tasks:
            return None

        context_lookup = {}
        task_output = None

        for task in self.tasks:
            if task.dependency:
                task_context = " ".join(
                    context_lookup[i.id] for i in task.dependency
                )
                task_output = task.agent.run(task, context=task_context, **kwargs)
            else:
                task_output = task.agent.run(task, **kwargs)

            context_lookup[task.id] = task_output

        return task_output
