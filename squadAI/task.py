from pydantic import BaseModel, Field, UUID4, InstanceOf, model_validator
from typing import Optional, Any
import uuid
from squadAI.createAgent import Agent
from squadAI.validation import TaskValidator, ValidationResult


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
    validator: optional callable that approves or rejects task output
    max_retries: number of re-attempts after a failed validation (0 = one attempt only)
    validates: optional upstream task to re-run when this task's validator rejects it

    Validation modes (optional):
    - Stage 1 (self): set ``validator`` only. The task re-runs with feedback until
      approved or ``max_retries`` is exhausted.
    - Stage 2 (upstream gate): set ``validator`` and ``validates`` to an upstream
      task in ``dependency``. The upstream task re-runs until this task's validator
      approves the pair of outputs.

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
    validator: TaskValidator | None = Field(
        default=None,
        description="Optional validator callable; omit for tasks without validation",
    )
    max_retries: int = Field(
        default=0,
        ge=0,
        description="Validation re-attempts after the first failure",
    )
    validates: Any | None = Field(
        default=None,
        description="Upstream task to retry when this validation gate rejects output",
    )

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_validator_config(self):
        if self.validates is not None and self.validator is None:
            raise ValueError(
                f"Task {self.task_description!r} sets validates=... but has no validator"
            )
        if self.validates is not None and self.validates not in self.dependency:
            raise ValueError(
                f"Task {self.task_description!r} must list validates target "
                f"in dependency"
            )
        return self


Task.model_rebuild()
