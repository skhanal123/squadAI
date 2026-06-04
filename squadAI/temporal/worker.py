"""Temporal worker factory for SquadAI."""

from __future__ import annotations

from temporalio.client import Client
from temporalio.worker import Worker

from squadAI.config import get_settings
from squadAI.temporal.activities import execute_squad_task
from squadAI.temporal.client import DEFAULT_TASK_QUEUE
from squadAI.temporal.workflow import SquadWorkflow

DEFAULT_TEMPORAL_ADDRESS = "localhost:7233"


async def create_temporal_client(
    target_host: str | None = None,
) -> Client:
    """Connect to a Temporal server."""
    return await Client.connect(
        target_host or get_settings().temporal_address
    )


def create_worker(
    client: Client,
    *,
    task_queue: str = DEFAULT_TASK_QUEUE,
) -> Worker:
    """Create a worker that executes squad workflows and activities."""
    return Worker(
        client,
        task_queue=task_queue,
        workflows=[SquadWorkflow],
        activities=[execute_squad_task],
    )
