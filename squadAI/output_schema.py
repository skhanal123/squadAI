"""Structured task output schemas and normalization helpers."""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_EXTRAS_KEYS = 3
MAX_EXTRA_KEY_LENGTH = 40
MAX_EXTRA_VALUE_LENGTH = 300
_EXTRA_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,39}$")

ModelT = TypeVar("ModelT", bound=BaseModel)


class TaskOutputBase(BaseModel):
    """Base for task output schemas with bounded optional extensions."""

    model_config = ConfigDict(extra="forbid")

    extras: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional extra key/value pairs when the core schema is insufficient. "
            f"At most {MAX_EXTRAS_KEYS} entries; values are short strings only."
        ),
    )

    @field_validator("extras")
    @classmethod
    def validate_extras(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > MAX_EXTRAS_KEYS:
            raise ValueError(
                f"extras may contain at most {MAX_EXTRAS_KEYS} keys, got {len(value)}"
            )

        for key, item in value.items():
            if len(key) > MAX_EXTRA_KEY_LENGTH:
                raise ValueError(
                    f"extras key {key!r} exceeds max length {MAX_EXTRA_KEY_LENGTH}"
                )
            if not _EXTRA_KEY_PATTERN.match(key):
                raise ValueError(
                    f"extras key {key!r} must match {_EXTRA_KEY_PATTERN.pattern}"
                )
            if len(item) > MAX_EXTRA_VALUE_LENGTH:
                raise ValueError(
                    f"extras[{key!r}] exceeds max length {MAX_EXTRA_VALUE_LENGTH}"
                )
        return value


class TaskOutputParseError(ValueError):
    """Raised when task output cannot be parsed or validated against a schema."""


def is_json_task_output(text: str) -> bool:
    """Return True if ``text`` looks like a JSON object payload."""
    stripped = text.strip()
    return stripped.startswith("{") and stripped.endswith("}")


def model_to_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a JSON Schema dict for ``model``."""
    return model.model_json_schema()


def build_openai_response_format(model: type[BaseModel]) -> dict[str, Any]:
    """Build an OpenAI ``response_format`` payload for structured JSON output."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "strict": False,
            "schema": model_to_json_schema(model),
        },
    }


def build_openai_response_format_from_schema(json_schema: dict[str, Any]) -> dict[str, Any]:
    """Build OpenAI ``response_format`` from a pre-serialized JSON Schema dict."""
    title = json_schema.get("title", "TaskOutput")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": str(title),
            "strict": False,
            "schema": json_schema,
        },
    }


def provider_supports_structured_output(provider: Any) -> bool:
    """Return True when the provider accepts ``response_format`` on completions."""
    from squadAI.providers.mock import MockProvider
    from squadAI.providers.openai_compatible import OpenAIChatProvider

    return isinstance(provider, (OpenAIChatProvider, MockProvider))


def normalize_task_output(
    raw: str,
    *,
    model: type[ModelT] | None = None,
    json_schema: dict[str, Any] | None = None,
) -> str:
    """Parse and validate task output, returning canonical JSON text."""
    if model is None and json_schema is None:
        return raw

    stripped = raw.strip()
    if not stripped:
        raise TaskOutputParseError("Task output is empty.")

    try:
        if model is not None:
            parsed = model.model_validate_json(stripped)
            return parsed.model_dump_json()
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise TaskOutputParseError(f"Task output is not valid JSON: {exc}") from exc
    except Exception as exc:
        raise TaskOutputParseError(f"Task output failed schema validation: {exc}") from exc

    if not isinstance(payload, dict):
        raise TaskOutputParseError("Task output JSON must be an object.")

    if json_schema is not None:
        _validate_required_fields(payload, json_schema)

    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def try_normalize_task_output(
    raw: str,
    *,
    model: type[ModelT] | None = None,
    json_schema: dict[str, Any] | None = None,
) -> str | None:
    """Return normalized JSON text, or ``None`` if parsing/validation fails."""
    try:
        return normalize_task_output(raw, model=model, json_schema=json_schema)
    except TaskOutputParseError:
        return None


def format_context_output(raw: str) -> str:
    """Format a task output string for inclusion in upstream context blocks."""
    if is_json_task_output(raw):
        try:
            return json.dumps(json.loads(raw), indent=2)
        except json.JSONDecodeError:
            pass
    return raw


def format_context_block(
    *,
    index: int,
    description: str,
    output: str,
) -> str:
    """Build one labeled upstream context block."""
    formatted_output = format_context_output(output)
    if is_json_task_output(output):
        return (
            f'<upstream_task index="{index}" format="application/json">\n'
            f"<description>{description}</description>\n"
            f"<output>\n{formatted_output}\n</output>\n"
            f"</upstream_task>"
        )
    return (
        f'<upstream_task index="{index}">\n'
        f"<description>{description}</description>\n"
        f"<output>{formatted_output}</output>\n"
        f"</upstream_task>"
    )


def _validate_required_fields(payload: dict[str, Any], json_schema: dict[str, Any]) -> None:
    required = json_schema.get("required", [])
    missing = [field for field in required if field not in payload]
    if missing:
        raise TaskOutputParseError(
            f"Task output missing required fields: {', '.join(missing)}"
        )

    properties = json_schema.get("properties", {})
    extras_schema = properties.get("extras", {})
    extras = payload.get("extras")
    if isinstance(extras, dict):
        max_props = extras_schema.get("maxProperties")
        if max_props is not None and len(extras) > max_props:
            raise TaskOutputParseError(
                f"extras may contain at most {max_props} keys, got {len(extras)}"
            )
