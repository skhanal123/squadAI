"""
Run the SquadAI Temporal worker.

Start Temporal locally first, for example with the Temporal CLI:
    temporal server start-dev

Then register your tools and start the worker:
    python temporal_worker.py
"""

import asyncio
import os

from dotenv import load_dotenv

from squadAI.temporal.registry import register_tools
from squadAI.temporal.worker import create_temporal_client, create_worker

load_dotenv()


def register_default_tools() -> None:
    """Register tools used by the example squads."""
    try:
        from example_run import (
            add_two_numbers,
            lookup_product_price,
            lookup_tax_rate,
            multiply_two_numbers,
        )
    except ImportError:
        return

    register_tools(
        add_two_numbers,
        multiply_two_numbers,
        lookup_product_price,
        lookup_tax_rate,
    )


async def main() -> None:
    register_default_tools()
    client = await create_temporal_client()
    task_queue = os.getenv("TEMPORAL_TASK_QUEUE", "squadai")
    worker = create_worker(client, task_queue=task_queue)
    print(f"SquadAI Temporal worker listening on queue '{task_queue}'")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
