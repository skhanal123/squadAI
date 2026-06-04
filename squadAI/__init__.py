"""SquadAI — lightweight multi-agent orchestration."""

__all__ = [
    "Agent",
    "AgentRunResult",
    "LLMSettings",
    "SquadAISettings",
    "SquadAgents",
    "SquadResult",
    "Task",
    "TaskResult",
    "TaskUsage",
    "TaskValidationError",
    "TokenUsage",
    "Tool",
    "ValidationResult",
    "clear_settings_cache",
    "create_provider",
    "get_llm_settings",
    "get_settings",
    "tool_wrapper",
]


def __getattr__(name: str):
    if name == "Agent":
        from squadAI.createAgent import Agent

        return Agent
    if name == "AgentRunResult":
        from squadAI.usage import AgentRunResult

        return AgentRunResult
    if name == "TokenUsage":
        from squadAI.usage import TokenUsage

        return TokenUsage
    if name == "TaskUsage":
        from squadAI.usage import TaskUsage

        return TaskUsage
    if name == "LLMSettings":
        from squadAI.config import LLMSettings

        return LLMSettings
    if name == "SquadAISettings":
        from squadAI.config import SquadAISettings

        return SquadAISettings
    if name == "clear_settings_cache":
        from squadAI.config import clear_settings_cache

        return clear_settings_cache
    if name == "get_llm_settings":
        from squadAI.config import get_llm_settings

        return get_llm_settings
    if name == "get_settings":
        from squadAI.config import get_settings

        return get_settings
    if name == "create_provider":
        from squadAI.llm import create_provider

        return create_provider
    if name == "SquadAgents":
        from squadAI.squadAgent import SquadAgents

        return SquadAgents
    if name == "SquadResult":
        from squadAI.squadAgent import SquadResult

        return SquadResult
    if name == "TaskResult":
        from squadAI.squadAgent import TaskResult

        return TaskResult
    if name == "Task":
        from squadAI.task import Task

        return Task
    if name == "Tool":
        from squadAI.tools import Tool

        return Tool
    if name == "tool_wrapper":
        from squadAI.tools import tool_wrapper

        return tool_wrapper
    if name == "ValidationResult":
        from squadAI.validation import ValidationResult

        return ValidationResult
    if name == "TaskValidationError":
        from squadAI.validation import TaskValidationError

        return TaskValidationError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
