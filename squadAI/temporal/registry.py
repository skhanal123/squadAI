"""Tool registry used by the Temporal worker."""

from __future__ import annotations

from squadAI.tools import Tool

TOOL_REGISTRY: dict[str, Tool] = {}


def register_tool(tool: Tool) -> Tool:
    """Register a tool for use inside Temporal activities."""
    TOOL_REGISTRY[tool.function_name] = tool
    return tool


def register_tools(*tools: Tool) -> None:
    """Register multiple tools for the worker."""
    for tool in tools:
        register_tool(tool)


def get_tools(names: list[str]) -> list[Tool]:
    """Resolve registered tools by function name."""
    missing = [name for name in names if name not in TOOL_REGISTRY]
    if missing:
        raise ValueError(
            "Tools not registered in Temporal worker: "
            + ", ".join(sorted(missing))
        )
    return [TOOL_REGISTRY[name] for name in names]
