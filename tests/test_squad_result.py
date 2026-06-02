import unittest

from squadAI.createAgent import Agent
from squadAI.providers.base import LLMResponse
from squadAI.providers.mock import MockProvider
from squadAI.squadAgent import SquadAgents, SquadResult, build_task_context
from squadAI.task import Task


class TestBuildTaskContext(unittest.TestCase):
    def test_single_dependency(self):
        agent = Agent(backstory="Helper.")
        task1 = Task(task_description="Look up price", agent=agent)
        task2 = Task(task_description="Calculate total", agent=agent)
        context_lookup = {task1.id: "Unit price is $999."}

        context = build_task_context([task1], context_lookup)

        self.assertIn('<upstream_task index="1">', context)
        self.assertIn("<description>Look up price</description>", context)
        self.assertIn("<output>Unit price is $999.</output>", context)
        self.assertIn("</upstream_task>", context)

    def test_multiple_dependencies_preserve_order(self):
        agent = Agent(backstory="Helper.")
        task_price = Task(task_description="Look up price", agent=agent)
        task_tax = Task(task_description="Look up tax rate", agent=agent)
        context_lookup = {
            task_price.id: "Price is $999.",
            task_tax.id: "Tax rate is 8.75%.",
        }

        context = build_task_context([task_price, task_tax], context_lookup)

        price_index = context.index('<upstream_task index="1">')
        tax_index = context.index('<upstream_task index="2">')
        self.assertLess(price_index, tax_index)
        self.assertIn("<output>Price is $999.</output>", context)
        self.assertIn("<output>Tax rate is 8.75%.", context)

    def test_empty_dependencies_returns_empty_string(self):
        context = build_task_context([], {})
        self.assertEqual(context, "")


class TestSquadValidation(unittest.TestCase):
    def test_rejects_missing_dependency_in_task_list(self):
        agent = Agent(backstory="Helper.")
        dependency = Task(task_description="Upstream task", agent=agent)
        dependent = Task(
            task_description="Downstream task",
            dependency=[dependency],
            agent=agent,
        )

        with self.assertRaises(ValueError) as ctx:
            SquadAgents(tasks=[dependent])

        self.assertIn("not included in the squad tasks list", str(ctx.exception))

    def test_rejects_dependency_after_dependent(self):
        agent = Agent(backstory="Helper.")
        task1 = Task(task_description="First task", agent=agent)
        task2 = Task(
            task_description="Second task",
            dependency=[task1],
            agent=agent,
        )

        with self.assertRaises(ValueError) as ctx:
            SquadAgents(tasks=[task2, task1])

        self.assertIn("must appear earlier in the tasks list", str(ctx.exception))


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

    def test_dependent_task_receives_labeled_context(self):
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
        squad.run(a=2, b=3, c=4)

        self.assertEqual(len(provider.calls), 2)
        second_user_message = provider.calls[1]["messages"][-1]["content"]
        self.assertIn("<context>", second_user_message)
        self.assertIn('<upstream_task index="1">', second_user_message)
        self.assertIn("<description>Add {a} and {b}</description>", second_user_message)
        self.assertIn("<output>Sum is 5.</output>", second_user_message)

    def test_branching_dependencies_include_all_labeled_outputs(self):
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
        squad = SquadAgents(tasks=[task_price, task_tax, task_total])
        squad.run()

        third_user_message = provider.calls[2]["messages"][-1]["content"]
        self.assertIn('<upstream_task index="1">', third_user_message)
        self.assertIn("<output>Price is $999.</output>", third_user_message)
        self.assertIn('<upstream_task index="2">', third_user_message)
        self.assertIn("<output>Tax rate is 8.75%.</output>", third_user_message)


if __name__ == "__main__":
    unittest.main()
