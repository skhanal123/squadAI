import unittest

from squadAI.createAgent import Agent
from squadAI.providers.base import LLMResponse
from squadAI.providers.mock import MockProvider
from squadAI.squadAgent import SquadAgents, SquadResult
from squadAI.task import Task


class TestSquadResult(unittest.TestCase):
    def test_empty_squad_returns_empty_result(self):
        squad = SquadAgents(tasks=[])
        result = squad.run()

        self.assertIsInstance(result, SquadResult)
        self.assertIsNone(result.final)
        self.assertEqual(result.task_results, [])
        self.assertEqual(result.outputs, {})

    def test_single_task_result(self):
        provider = MockProvider([LLMResponse(content="Physics answer.")])
        agent = Agent(backstory="You teach physics.", provider=provider)
        task = Task(task_description="Explain inertia.", agent=agent)
        squad = SquadAgents(tasks=[task])

        result = squad.run()

        self.assertEqual(result.final, "Physics answer.")
        self.assertEqual(len(result.task_results), 1)
        self.assertEqual(result.task_results[0].description, "Explain inertia.")
        self.assertEqual(result.get(task), "Physics answer.")
        self.assertEqual(result.outputs[task.id], "Physics answer.")

    def test_chained_tasks_capture_all_outputs(self):
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
        squad = SquadAgents(tasks=[task1, task2])

        result = squad.run(a=2, b=3, c=4)

        self.assertEqual(result.final, "Product is 20.")
        self.assertEqual(len(result.task_results), 2)
        self.assertEqual(result.get(task1), "Sum is 5.")
        self.assertEqual(result.get(task2), "Product is 20.")


if __name__ == "__main__":
    unittest.main()
