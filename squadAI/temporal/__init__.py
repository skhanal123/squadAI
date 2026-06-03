"""Temporal integration for SquadAI.

Import submodules directly to avoid loading heavy dependencies into the
Temporal workflow sandbox, e.g. ``from squadAI.temporal.client import ...``.
"""

__all__ = [
    "DEFAULT_TASK_QUEUE",
    "EXECUTE_SQUAD_TASK",
    "SquadWorkflow",
    "SquadWorkflowInput",
    "create_temporal_client",
    "create_worker",
    "execute_squad_workflow",
    "register_tool",
    "register_tools",
    "workflow_result_to_squad_result",
]


def __getattr__(name: str):
    if name in ("DEFAULT_TASK_QUEUE", "EXECUTE_SQUAD_TASK", "execute_squad_workflow", "workflow_result_to_squad_result"):
        from squadAI.temporal.client import (
            DEFAULT_TASK_QUEUE,
            EXECUTE_SQUAD_TASK,
            execute_squad_workflow,
            workflow_result_to_squad_result,
        )

        exports = {
            "DEFAULT_TASK_QUEUE": DEFAULT_TASK_QUEUE,
            "EXECUTE_SQUAD_TASK": EXECUTE_SQUAD_TASK,
            "execute_squad_workflow": execute_squad_workflow,
            "workflow_result_to_squad_result": workflow_result_to_squad_result,
        }
        return exports[name]
    if name == "SquadWorkflowInput":
        from squadAI.temporal.models import SquadWorkflowInput

        return SquadWorkflowInput
    if name == "SquadWorkflow":
        from squadAI.temporal.workflow import SquadWorkflow

        return SquadWorkflow
    if name in ("create_temporal_client", "create_worker"):
        from squadAI.temporal.worker import create_temporal_client, create_worker

        return create_temporal_client if name == "create_temporal_client" else create_worker
    if name in ("register_tool", "register_tools"):
        from squadAI.temporal.registry import register_tool, register_tools

        return register_tool if name == "register_tool" else register_tools
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
