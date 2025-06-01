from pydantic import BaseModel, Field, UUID4, InstanceOf
from typing import Optional, Any
import uuid
from squadAI.createAgent import Agent


class Task(BaseModel):
    """
    This is the base class to define task for an agent.

    Attributes:
    -----------
    id: unique id provided to the task when instantiated
    task_description (str): description of the task
    agent: agent to which the task is assigned
    dependency: list of other dependent tasks, if any, for this task to get executed
    task_output: format of the output when task is executed. This is an optional field

    """

    id: UUID4 = Field(
        default_factory=uuid.uuid4, description="Provides the unique id for task"
    )
    task_description: str
    agent: InstanceOf[Agent]
    dependency: list[Any] = Field(
        default=[], description="List of dependency tasks for this task to get complete"
    )
    task_output: Optional[str] = None
