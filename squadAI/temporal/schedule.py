"""Dependency graph helpers for Temporal squad workflows."""

from __future__ import annotations

from squadAI.output_schema import format_context_block
from squadAI.temporal.models import TaskSpec


def validate_task_specs(tasks: list[TaskSpec]) -> None:
    """Validate dependency membership, duplicates, and cycles."""
    if not tasks:
        return

    task_by_id: dict[str, TaskSpec] = {}
    task_index: dict[str, int] = {}

    for index, task in enumerate(tasks):
        if task.task_id in task_by_id:
            raise ValueError(
                f"Duplicate task id in squad workflow: {task.description!r}"
            )
        task_by_id[task.task_id] = task
        task_index[task.task_id] = index

    task_ids = set(task_by_id)

    in_degree = {task.task_id: 0 for task in tasks}
    dependents: dict[str, list[str]] = {task.task_id: [] for task in tasks}

    for task in tasks:
        for dependency_id in task.dependency_ids:
            if dependency_id not in task_ids:
                raise ValueError(
                    f"Task {task.description!r} depends on unknown task id "
                    f"{dependency_id!r}"
                )
            in_degree[task.task_id] += 1
            dependents[dependency_id].append(task.task_id)

    ready = sorted(
        (task_id for task_id, degree in in_degree.items() if degree == 0),
        key=lambda task_id: task_index[task_id],
    )
    visited = 0

    while ready:
        current_id = ready.pop(0)
        visited += 1

        for dependent_id in sorted(
            dependents[current_id], key=lambda task_id: task_index[task_id]
        ):
            in_degree[dependent_id] -= 1
            if in_degree[dependent_id] == 0:
                ready.append(dependent_id)
                ready.sort(key=lambda task_id: task_index[task_id])

    if visited != len(tasks):
        remaining = [
            task.description for task in tasks if in_degree[task.task_id] > 0
        ]
        raise ValueError(
            "Circular task dependency detected among: "
            f"{', '.join(repr(name) for name in remaining) or 'unknown tasks'}"
        )


def execution_levels(tasks: list[TaskSpec]) -> list[list[TaskSpec]]:
    """Return tasks grouped by DAG level for parallel Temporal execution."""
    if not tasks:
        return []

    validate_task_specs(tasks)

    task_by_id = {task.task_id: task for task in tasks}
    task_index = {task.task_id: index for index, task in enumerate(tasks)}

    in_degree = {task.task_id: len(task.dependency_ids) for task in tasks}
    dependents: dict[str, list[str]] = {task.task_id: [] for task in tasks}

    for task in tasks:
        for dependency_id in task.dependency_ids:
            dependents[dependency_id].append(task.task_id)

    levels: list[list[TaskSpec]] = []
    ready = sorted(
        (task_id for task_id, degree in in_degree.items() if degree == 0),
        key=lambda task_id: task_index[task_id],
    )

    while ready:
        current_level = [task_by_id[task_id] for task_id in ready]
        levels.append(current_level)
        next_ready: list[str] = []

        for task_id in ready:
            for dependent_id in sorted(
                dependents[task_id], key=lambda value: task_index[value]
            ):
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    next_ready.append(dependent_id)

        ready = sorted(next_ready, key=lambda task_id: task_index[task_id])

    return levels


def build_task_context_from_specs(
    dependency_ids: list[str],
    task_by_id: dict[str, TaskSpec],
    context_lookup: dict[str, str],
) -> str:
    """Build labeled upstream context blocks for Temporal activities."""
    blocks = []
    for index, dependency_id in enumerate(dependency_ids, start=1):
        dependency = task_by_id[dependency_id]
        blocks.append(
            format_context_block(
                index=index,
                description=dependency.description,
                output=context_lookup[dependency_id],
            )
        )
    return "\n\n".join(blocks)
