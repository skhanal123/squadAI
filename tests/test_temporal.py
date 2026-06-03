import unittest
import uuid

from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from squadAI.createAgent import Agent
from squadAI.providers.base import LLMResponse
from squadAI.providers.mock import MockProvider
from squadAI.squadAgent import SquadAgents
from squadAI.task import Task
from squadAI.temporal.activities import execute_squad_task
from squadAI.temporal.models import SquadWorkflowInput, TaskSpec
from squadAI.temporal.schedule import execution_levels, validate_task_specs
from squadAI.temporal.workflow import SquadWorkflow


class TestTemporalSchedule(unittest.TestCase):
    def test_execution_levels_for_branching_dag(self):
        tasks = [
            TaskSpec(task_id="total", description="Total", dependency_ids=["price", "tax"]),
            TaskSpec(task_id="price", description="Price"),
            TaskSpec(task_id="tax", description="Tax"),
        ]

        levels = execution_levels(tasks)

        self.assertEqual([[task.description for task in level] for level in levels], [
            ["Price", "Tax"],
            ["Total"],
        ])

    def test_rejects_cycle(self):
        tasks = [
            TaskSpec(task_id="a", description="A", dependency_ids=["c"]),
            TaskSpec(task_id="b", description="B", dependency_ids=["a"]),
            TaskSpec(task_id="c", description="C", dependency_ids=["b"]),
        ]

        with self.assertRaises(ValueError) as ctx:
            validate_task_specs(tasks)

        self.assertIn("Circular task dependency", str(ctx.exception))


class TestTemporalWorkflow(unittest.IsolatedAsyncioTestCase):
    async def test_chained_tasks_via_temporal(self):
        provider = MockProvider(
            [
                LLMResponse(content="Sum is 5."),
                LLMResponse(content="Product is 20."),
            ]
        )
        agent = Agent(backstory="Math helper.", provider=provider)

        task1 = Task(task_description="Add {a} and {b}", agent=agent)
        task2 = Task(
            task_description="Multiply previous result by {c}",
            dependency=[task1],
            agent=agent,
        )
        squad = SquadAgents(tasks=[task2, task1])

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="test-squad",
                workflows=[SquadWorkflow],
                activities=[execute_squad_task],
            ):
                result = await squad.run_temporal(
                    env.client,
                    task_queue="test-squad",
                    a=2,
                    b=3,
                    c=4,
                )

        self.assertEqual(result.final, "Product is 20.")
        self.assertEqual(result.get(task1), "Sum is 5.")
        self.assertEqual(result.get(task2), "Product is 20.")

    async def test_parallel_branching_via_temporal(self):
        provider = MockProvider(
            [
                LLMResponse(content="Price is $999."),
                LLMResponse(content="Tax rate is 8.75%."),
                LLMResponse(content="Total is $3,259.24."),
            ]
        )
        agent = Agent(backstory="Billing helper.", provider=provider)

        task_price = Task(task_description="Look up price", agent=agent)
        task_tax = Task(task_description="Look up tax rate", agent=agent)
        task_total = Task(
            task_description="Calculate checkout total",
            dependency=[task_price, task_tax],
            agent=agent,
        )
        squad = SquadAgents(tasks=[task_total, task_price, task_tax])

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="test-squad",
                workflows=[SquadWorkflow],
                activities=[execute_squad_task],
            ):
                result = await squad.run_temporal(
                    env.client,
                    task_queue="test-squad",
                )

        self.assertEqual(result.final, "Total is $3,259.24.")
        self.assertEqual(result.get(task_price), "Price is $999.")
        self.assertEqual(result.get(task_tax), "Tax rate is 8.75%.")

    async def test_workflow_input_directly(self):
        task_id = str(uuid.uuid4())
        workflow_input = SquadWorkflowInput(
            tasks=[
                TaskSpec(task_id=task_id, description="Say hello"),
            ],
            run_kwargs={},
        )
        workflow_input.tasks[0].agent.backstory = "You are concise."
        workflow_input.tasks[0].agent.provider = "mock"
        workflow_input.tasks[0].agent.mock_response = "Hello."

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="test-squad",
                workflows=[SquadWorkflow],
                activities=[execute_squad_task],
            ):
                from squadAI.temporal.client import execute_squad_workflow

                result = await execute_squad_workflow(
                    env.client,
                    workflow_input,
                    task_queue="test-squad",
                )

        self.assertEqual(result.final, "Hello.")


if __name__ == "__main__":
    unittest.main()
