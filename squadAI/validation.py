"""Validation helpers for bounded task retries in core squad execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from squadAI.task import Task

TaskValidator = Callable[..., "ValidationResult | bool"]


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating a task output."""

    approved: bool
    feedback: str = ""


class TaskValidationError(Exception):
    """Raised when validation retries are exhausted."""

    def __init__(
        self,
        task: Task,
        output: str,
        feedback: str,
    ) -> None:
        self.task = task
        self.output = output
        self.feedback = feedback
        super().__init__(
            f"Task {task.task_description!r} failed validation after "
            f"{task.max_retries + 1} attempt(s): {feedback}"
        )


def normalize_validation(result: ValidationResult | bool) -> ValidationResult:
    """Convert a validator return value into a :class:`ValidationResult`."""
    if isinstance(result, ValidationResult):
        return result
    if isinstance(result, bool):
        return ValidationResult(
            approved=result,
            feedback="" if result else "Validation failed.",
        )
    raise TypeError(
        "Validator must return ValidationResult or bool, "
        f"got {type(result).__name__}"
    )


def call_task_validator(
    task: Task,
    *,
    output: str,
    upstream_output: str | None = None,
) -> ValidationResult:
    """Invoke a task validator for self-validation or upstream validation.

    Validators receive only task output strings — not ``squad.run()`` template
    parameters (those are used for ``task_description.format()`` only).
    """
    if task.validator is None:
        raise ValueError(f"Task {task.task_description!r} has no validator configured")

    if upstream_output is not None:
        raw = task.validator(output, upstream_output=upstream_output)
    else:
        raw = task.validator(output)

    return normalize_validation(raw)
