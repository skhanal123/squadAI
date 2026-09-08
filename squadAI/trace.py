"""Structured execution traces for squad task runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

TRACE_RESULT_MAX_LEN = 2000
DEFAULT_TRACE_FILENAME = "execution_trace.json"


def truncate_trace_text(value: str, *, max_len: int = TRACE_RESULT_MAX_LEN) -> str:
    """Truncate long tool results for trace storage."""
    if len(value) <= max_len:
        return value
    omitted = len(value) - max_len
    return f"{value[:max_len]}... [{omitted} chars truncated]"


def is_tool_error_result(result: str) -> bool:
    """Return True when a tool result string indicates a tool failure."""
    return result.startswith("Error executing tool ") or result.startswith(
        "Unknown tool "
    )


class ToolCallRecord(BaseModel):
    """One tool invocation within a ReAct step."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    error: bool = False


class ReActStepRecord(BaseModel):
    """One LLM turn in the ReAct loop."""

    iteration: int
    assistant_content: str | None = None
    finish_reason: str | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)


class ValidationRecord(BaseModel):
    """Outcome of a validator check."""

    approved: bool
    feedback: str = ""


class AgentRunTrace(BaseModel):
    """Trace of a single agent ReAct invocation."""

    react_steps: list[ReActStepRecord] = Field(default_factory=list)
    iterations: int = 0
    status: Literal["success", "react_exhausted"] = "success"


class TaskAttemptRecord(BaseModel):
    """One agent run, optionally followed by validation."""

    attempt: int
    agent: AgentRunTrace = Field(default_factory=AgentRunTrace)
    validation: ValidationRecord | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class GateRoundRecord(BaseModel):
    """One upstream+gate cycle in a validation gate loop."""

    round: int
    upstream_attempts: list[TaskAttemptRecord] = Field(default_factory=list)
    gate_agent: AgentRunTrace | None = None
    gate_skipped: bool = False
    validation: ValidationRecord = Field(
        default_factory=lambda: ValidationRecord(approved=False)
    )


class TaskExecutionTrace(BaseModel):
    """Full execution trace for one squad task."""

    status: Literal["success", "validation_failed"] = "success"
    dag_level: int | None = None
    parallel: bool = False
    is_validation_gate: bool = False
    validates_task_id: str | None = None
    validated_by_gate: bool = False
    attempts: list[TaskAttemptRecord] = Field(default_factory=list)
    gate_rounds: list[GateRoundRecord] = Field(default_factory=list)


def build_squad_trace_payload(task_results: list[Any]) -> dict[str, Any]:
    """Build a JSON-serializable execution trace for all squad tasks."""
    return {
        "tasks": [
            {
                "task_id": str(task_result.task_id),
                "description": task_result.description,
                "trace": task_result.trace.model_dump(),
            }
            for task_result in task_results
        ],
    }


def write_trace_file(
    payload: dict[str, Any],
    path: Path | str,
    *,
    filename: str = DEFAULT_TRACE_FILENAME,
) -> Path:
    """Write an execution trace payload to ``path`` (directory or ``.json`` file)."""
    destination = Path(path)
    if destination.suffix.lower() == ".json":
        trace_path = destination
        trace_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        destination.mkdir(parents=True, exist_ok=True)
        trace_path = destination / filename

    trace_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return trace_path
