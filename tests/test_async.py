import asyncio
import unittest

from squadAI.createAgent import Agent
from squadAI.providers.base import LLMResponse
from squadAI.providers.mock import MockProvider
from squadAI.reactAgent import ReactAgent
from squadAI.squadAgent import SquadAgents
from squadAI.task import Task


class TestReactAgentAsync(unittest.IsolatedAsyncioTestCase):
    async def test_invoke_async(self):
        provider = MockProvider([LLMResponse(content="Async answer.")])
        agent = ReactAgent(
            prompt="You are helpful.",
            provider=provider,
        )

        result = await agent.invoke_async("Hello")

        self.assertEqual(result, "Async answer.")
        self.assertEqual(len(provider.calls), 1)


class TestSquadRunAsync(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_independent_tasks(self):
        price_provider = MockProvider([LLMResponse(content="Price is $999.")])
        tax_provider = MockProvider([LLMResponse(content="Tax rate is 8.75%.")])
        total_provider = MockProvider([LLMResponse(content="Total is $3,259.24.")])

        task_price = Task(
            task_description="Look up price",
            agent=Agent(backstory="Billing.", provider=price_provider),
        )
        task_tax = Task(
            task_description="Look up tax rate",
            agent=Agent(backstory="Billing.", provider=tax_provider),
        )
        task_total = Task(
            task_description="Calculate checkout total",
            dependency=[task_price, task_tax],
            agent=Agent(backstory="Billing.", provider=total_provider),
        )
        squad = SquadAgents(tasks=[task_price, task_tax, task_total])

        result = await squad.run_async()

        self.assertEqual(result.get(task_price), "Price is $999.")
        self.assertEqual(result.get(task_tax), "Tax rate is 8.75%.")
        self.assertEqual(result.final, "Total is $3,259.24.")
        self.assertEqual(len(price_provider.calls), 1)
        self.assertEqual(len(tax_provider.calls), 1)
        self.assertEqual(len(total_provider.calls), 1)

    async def test_run_wrapper_uses_async_engine(self):
        provider = MockProvider([LLMResponse(content="Done.")])
        agent = Agent(backstory="Helper.", provider=provider)
        task = Task(task_description="Say hello.", agent=agent)
        squad = SquadAgents(tasks=[task])

        result = await asyncio.to_thread(squad.run)

        self.assertEqual(result.final, "Done.")


if __name__ == "__main__":
    unittest.main()
