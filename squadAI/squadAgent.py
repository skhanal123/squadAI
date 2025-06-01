from pydantic import BaseModel, InstanceOf
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

    def run(self, **kwargs):
        context_lookup = {}

        if self.tasks:
            for task in self.tasks:
                if task.dependency:
                    task_context = " ".join(
                        context_lookup[i.id] for i in task.dependency
                    )
                    task_output = task.agent.run(task, context=task_context, **kwargs)

                    context_lookup[task.id] = task_output
                else:
                    task_output = task.agent.run(task, **kwargs)
                    context_lookup[task.id] = task_output

        return task_output
