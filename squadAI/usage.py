"""Token usage and billing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from squadAI.trace import AgentRunTrace


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True)
class TaskUsage:
    """Billing unit for one squad task (model + token counts)."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @classmethod
    def from_tokens(cls, model: str, usage: TokenUsage) -> TaskUsage:
        return cls(
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )

    def __add__(self, other: TaskUsage) -> TaskUsage:
        if self.model != other.model:
            raise ValueError(
                f"Cannot merge TaskUsage for different models: "
                f"{self.model!r} vs {other.model!r}"
            )
        return TaskUsage(
            model=self.model,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass
class AgentRunResult:
    output: str
    usage: TokenUsage
    model: str
    trace: "AgentRunTrace | None" = None

    def __str__(self) -> str:
        return self.output

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.output == other
        return NotImplemented

    @property
    def task_usage(self) -> TaskUsage:
        return TaskUsage.from_tokens(self.model, self.usage)


def resolve_model_name(provider: Any | None) -> str:
    """Resolve the LLM model name from a provider or environment settings."""
    if provider is not None:
        model = getattr(provider, "model", None)
        if model:
            return str(model)

    from squadAI.config import get_llm_settings

    return get_llm_settings().model


def usage_from_openai_response(raw: Any) -> TokenUsage:
    usage = getattr(raw, "usage", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
    )


def usage_from_anthropic_response(raw: Any) -> TokenUsage:
    usage = getattr(raw, "usage", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
    )


def merge_usage_by_model(task_results: list) -> dict[str, TokenUsage]:
    """Aggregate token usage keyed by model name."""
    by_model: dict[str, TokenUsage] = {}
    for result in task_results:
        usage = getattr(result, "usage", None)
        if usage is None:
            continue
        model = usage.model if hasattr(usage, "model") else None
        if not model:
            continue
        tokens = TokenUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
        by_model[model] = by_model.get(model, TokenUsage()) + tokens
    return by_model


def sum_task_usages(task_results: list) -> TokenUsage:
    total = TokenUsage()
    for result in task_results:
        usage = getattr(result, "usage", None)
        if usage is None:
            continue
        total += TokenUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
    return total


def format_usage_display(task_results: list) -> str:
    """Build a human-readable usage summary for customer-facing display."""
    lines = ["Token usage:"]
    squad_total = sum_task_usages(task_results)
    lines.append(
        f"  Total: {squad_total.total_tokens:,} "
        f"({squad_total.input_tokens:,} input, {squad_total.output_tokens:,} output)"
    )
    lines.append("  Per task:")
    for result in task_results:
        usage = result.usage
        desc = result.description[:50]
        if len(result.description) > 50:
            desc += "..."
        lines.append(
            f"    - {desc} ({usage.model}): {usage.total_tokens:,} tokens "
            f"({usage.input_tokens:,} in, {usage.output_tokens:,} out)"
        )
    by_model = merge_usage_by_model(task_results)
    if by_model:
        model_parts = [
            f"{model} {tokens.total_tokens:,}"
            for model, tokens in sorted(by_model.items())
        ]
        lines.append(f"  By model: {', '.join(model_parts)}")
    return "\n".join(lines)
