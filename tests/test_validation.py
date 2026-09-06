import unittest

from squadAI.createAgent import Agent
from squadAI.providers.base import LLMResponse
from squadAI.providers.mock import MockProvider
from squadAI.squadAgent import SquadAgents
from squadAI.task import Task
from squadAI.validation import TaskValidationError, ValidationResult


class TestTaskValidationConfig(unittest.TestCase):
    def test_validates_requires_validator(self):
        agent = Agent(backstory="Helper.")
        writer = Task(task_description="Write draft", agent=agent)
        with self.assertRaises(ValueError) as ctx:
            Task(
                task_description="Review draft",
                agent=agent,
                dependency=[writer],
                validates=writer,
            )

        self.assertIn("has no validator", str(ctx.exception))

    def test_validates_target_must_be_dependency(self):
        agent = Agent(backstory="Helper.")
        writer = Task(task_description="Write draft", agent=agent)
        with self.assertRaises(ValueError) as ctx:
            Task(
                task_description="Review draft",
                agent=agent,
                validator=lambda output: True,
                validates=writer,
            )

        self.assertIn("must list validates target", str(ctx.exception))


class TestStageOneSelfValidation(unittest.TestCase):
    def test_retries_task_until_validator_passes(self):
        attempts = {"count": 0}

        def min_length_validator(output: str) -> ValidationResult:
            attempts["count"] += 1
            if len(output) >= 10:
                return ValidationResult(approved=True)
            return ValidationResult(
                approved=False,
                feedback="Response is too short.",
            )

        provider = MockProvider(
            [
                LLMResponse(content="short"),
                LLMResponse(content="long enough"),
            ]
        )
        agent = Agent(backstory="Writer.", provider=provider)
        task = Task(
            task_description="Write a sentence.",
            agent=agent,
            validator=min_length_validator,
            max_retries=2,
        )
        squad = SquadAgents(tasks=[task])

        result = squad.run()

        self.assertEqual(result.final, "long enough")
        self.assertEqual(attempts["count"], 2)
        self.assertEqual(len(provider.calls), 2)
        second_prompt = provider.calls[1]["messages"][-1]["content"]
        self.assertIn("<validation_feedback>", second_prompt)
        self.assertIn("Response is too short.", second_prompt)

    def test_task_without_validator_runs_once(self):
        provider = MockProvider([LLMResponse(content="Done.")])
        agent = Agent(backstory="Writer.", provider=provider)
        task = Task(task_description="Write.", agent=agent)
        squad = SquadAgents(tasks=[task])

        result = squad.run()

        self.assertEqual(result.final, "Done.")
        self.assertEqual(len(provider.calls), 1)

    def test_raises_when_validation_retries_exhausted(self):
        provider = MockProvider(
            [
                LLMResponse(content="bad"),
                LLMResponse(content="still bad"),
            ]
        )
        agent = Agent(backstory="Writer.", provider=provider)
        task = Task(
            task_description="Write JSON.",
            agent=agent,
            validator=lambda output: False,
            max_retries=1,
        )
        squad = SquadAgents(tasks=[task])

        with self.assertRaises(TaskValidationError):
            squad.run()


class TestStageTwoValidationGate(unittest.TestCase):
    def test_retries_upstream_task_when_gate_rejects(self):
        gate_attempts = {"count": 0}

        def critic_validator(
            critic_output: str,
            *,
            upstream_output: str,
        ) -> ValidationResult:
            gate_attempts["count"] += 1
            if upstream_output.startswith("APPROVED:"):
                return ValidationResult(approved=True)
            return ValidationResult(
                approved=False,
                feedback="Add more concrete detail.",
            )

        provider = MockProvider(
            [
                LLMResponse(content="Draft v1"),
                LLMResponse(content="REJECTED: too vague"),
                LLMResponse(content="APPROVED: Draft v2"),
                LLMResponse(content="Looks good"),
            ]
        )
        writer_agent = Agent(backstory="Writer.", provider=provider)
        critic_agent = Agent(backstory="Critic.", provider=provider)

        writer = Task(task_description="Write draft", agent=writer_agent)
        critic = Task(
            task_description="Review draft",
            agent=critic_agent,
            dependency=[writer],
            validates=writer,
            validator=critic_validator,
            max_retries=2,
        )
        squad = SquadAgents(tasks=[writer, critic])

        result = squad.run()

        self.assertEqual(result.get(writer), "APPROVED: Draft v2")
        self.assertEqual(result.get(critic), "Looks good")
        self.assertEqual(gate_attempts["count"], 2)
        self.assertEqual(len(provider.calls), 4)

        writer_retry_prompt = provider.calls[2]["messages"][-1]["content"]
        self.assertIn("<validation_feedback>", writer_retry_prompt)
        self.assertIn("Add more concrete detail.", writer_retry_prompt)

    def test_skips_standalone_run_of_validated_upstream_task(self):
        provider = MockProvider(
            [
                LLMResponse(content="APPROVED: final draft"),
                LLMResponse(content="Approved"),
            ]
        )
        agent = Agent(backstory="Helper.", provider=provider)
        writer = Task(task_description="Write draft", agent=agent)
        critic = Task(
            task_description="Review draft",
            agent=agent,
            dependency=[writer],
            validates=writer,
            validator=lambda critic_output, upstream_output: ValidationResult(
                approved=upstream_output.startswith("APPROVED:")
            ),
        )
        squad = SquadAgents(tasks=[writer, critic])

        result = squad.run()

        self.assertEqual(len(result.task_results), 2)
        self.assertEqual(len(provider.calls), 2)


class TestValidationWithRunKwargs(unittest.TestCase):
    """Regression: squad.run(template=...) must not leak kwargs into validators."""

    def test_self_validator_ignores_run_template_kwargs(self):
        provider = MockProvider([LLMResponse(content="long enough")])
        agent = Agent(backstory="Writer.", provider=provider)
        task = Task(
            task_description="Write about {topic}.",
            agent=agent,
            validator=lambda output: ValidationResult(approved=len(output) >= 5),
            max_retries=1,
        )
        squad = SquadAgents(tasks=[task])

        result = squad.run(topic="latency")

        self.assertEqual(result.final, "long enough")

    def test_gate_validator_ignores_run_template_kwargs(self):
        provider = MockProvider(
            [
                LLMResponse(content="APPROVED: draft"),
                LLMResponse(content="LGTM"),
            ]
        )
        agent = Agent(backstory="Helper.", provider=provider)
        writer = Task(task_description="Draft for {service}", agent=agent)
        critic = Task(
            task_description="Review draft",
            agent=agent,
            dependency=[writer],
            validates=writer,
            validator=lambda critic_output, *, upstream_output: ValidationResult(
                approved=upstream_output.startswith("APPROVED:")
            ),
        )
        squad = SquadAgents(tasks=[writer, critic])

        result = squad.run(service="checkout-service", region="us-east-1")

        self.assertEqual(result.get(writer), "APPROVED: draft")
        self.assertEqual(result.get(critic), "LGTM")


if __name__ == "__main__":
    unittest.main()
