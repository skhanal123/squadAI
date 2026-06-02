"""SquadAI — lightweight multi-agent orchestration."""

__all__ = [
    "Agent",
    "LLMSettings",
    "SquadAgents",
    "Task",
    "Tool",
    "create_provider",
    "tool_wrapper",
]


def __getattr__(name: str):
    if name == "Agent":
        from squadAI.createAgent import Agent

        return Agent
    if name == "LLMSettings":
        from squadAI.llm import LLMSettings

        return LLMSettings
    if name == "create_provider":
        from squadAI.llm import create_provider

        return create_provider
    if name == "SquadAgents":
        from squadAI.squadAgent import SquadAgents

        return SquadAgents
    if name == "Task":
        from squadAI.task import Task

        return Task
    if name == "Tool":
        from squadAI.tools import Tool

        return Tool
    if name == "tool_wrapper":
        from squadAI.tools import tool_wrapper

        return tool_wrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
