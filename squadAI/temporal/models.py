"""Serializable models for Temporal squad workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentSpec:
    """Serializable agent configuration for a Temporal activity."""

    backstory: str
    tool_names: list[str] = field(default_factory=list)
    max_iterations: int = 4
    provider: str = "env"
    model: str = ""
    mock_response: str | None = None


@dataclass
class TaskSpec:
    """Serializable task definition passed to the Temporal workflow."""

    task_id: str
    description: str
    dependency_ids: list[str] = field(default_factory=list)
    task_output: str | None = None
    output_json_schema: dict | None = None
    agent: AgentSpec = field(default_factory=lambda: AgentSpec(backstory=""))


@dataclass
class SquadWorkflowInput:
    """Input payload for :class:`SquadWorkflow`."""

    tasks: list[TaskSpec]
    run_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskActivityInput:
    """Input payload for the ``execute_squad_task`` activity."""

    task_id: str
    description: str
    task_output: str | None
    output_json_schema: dict | None
    run_kwargs: dict[str, Any]
    context: str
    agent: AgentSpec


@dataclass
class TaskActivityOutput:
    """Output from a single squad task activity."""

    task_id: str
    description: str
    output: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class SquadWorkflowResult:
    """Workflow result returned to callers."""

    final: str | None
    task_results: list[TaskActivityOutput]
