"""Structured execution traces for squad task runs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

TRACE_RESULT_MAX_LEN = 2000


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
    gate_agent: AgentRunTrace = Field(default_factory=AgentRunTrace)
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
