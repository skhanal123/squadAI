from pydantic import BaseModel, Field, UUID4, InstanceOf, model_validator
from typing import Optional, Any
import uuid
from squadAI.createAgent import Agent
from squadAI.output_schema import model_to_json_schema
from squadAI.validation import TaskValidator, ValidationResult


class Task(BaseModel):
    """
    This is the base class to define task for an agent.

    Attributes:
    -----------
    id: unique id provided to the task when instantiated
    task_description (str): description of the task
    agent: agent to which the task is assigned (optional for validation gates)
    dependency: list of other dependent tasks, if any, for this task to get executed
    task_output: format of the output when task is executed. This is an optional field
    output_schema: optional Pydantic model defining structured task output
    validator: optional callable that approves or rejects task output
    max_retries: number of re-attempts after a failed validation (0 = one attempt only)
    validates: optional upstream task to re-run when this task's validator rejects it

    Validation modes (optional):
    - Stage 1 (self): set ``validator`` only. The task re-runs with feedback until
      approved or ``max_retries`` is exhausted.
    - Stage 2 (upstream gate): set ``validator`` and ``validates`` to an upstream
      task in ``dependency``. The upstream task re-runs until this task's validator
      approves. Omit ``agent`` for a programmatic gate (validator only); include
      ``agent`` when the gate also runs a critic agent before validation.

    Validator signatures (``squad.run()`` template kwargs are not passed through):
    - Self: ``validator(output: str) -> ValidationResult | bool``
    - Gate: ``validator(gate_output: str, *, upstream_output: str) -> ValidationResult | bool``
      (``gate_output`` is empty for programmatic gates)

    """

    id: UUID4 = Field(
        default_factory=uuid.uuid4, description="Provides the unique id for task"
    )
    task_description: str
    agent: InstanceOf[Agent] | None = None
    dependency: list[Any] = Field(
        default=[], description="List of dependency tasks for this task to get complete"
    )
    task_output: Optional[str] = None
    output_schema: Any | None = Field(
        default=None,
        description="Optional Pydantic BaseModel subclass for structured task output",
    )
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
        if self.output_schema is not None:
            if not isinstance(self.output_schema, type) or not issubclass(
                self.output_schema, BaseModel
            ):
                raise ValueError(
                    f"Task {self.task_description!r} output_schema must be a "
                    f"Pydantic BaseModel subclass"
                )
        if self.validates is not None and self.validator is None:
            raise ValueError(
                f"Task {self.task_description!r} sets validates=... but has no validator"
            )
        if self.validates is not None and self.validates not in self.dependency:
            raise ValueError(
                f"Task {self.task_description!r} must list validates target "
                f"in dependency"
            )
        if self.validates is None and self.agent is None:
            raise ValueError(
                f"Task {self.task_description!r} requires an agent"
            )
        return self

    def get_output_json_schema(self) -> dict | None:
        """Return a JSON Schema dict for :attr:`output_schema`, if configured."""
        if self.output_schema is None:
            return None
        return model_to_json_schema(self.output_schema)


Task.model_rebuild()
