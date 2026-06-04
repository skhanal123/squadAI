import os
import unittest
from unittest import mock

from pydantic import ValidationError

from squadAI.config import (
    LLMSettings,
    SquadAISettings,
    clear_settings_cache,
    get_llm_settings,
    get_settings,
)
from squadAI.llm import create_provider
from squadAI.providers.mock import MockProvider


class TestLLMSettings(unittest.TestCase):
    def setUp(self):
        clear_settings_cache()

    def tearDown(self):
        clear_settings_cache()

    def test_mock_provider_needs_no_api_key(self):
        settings = LLMSettings(provider="mock", model="mock-model")

        self.assertEqual(settings.provider, "mock")
        self.assertIsNone(settings.api_key)

    def test_infers_openai_compatible_for_deepseek_model(self):
        settings = LLMSettings(
            _env_file=None,
            model="deepseek-chat",
            deepseek_api_key="test-key",
        )

        self.assertEqual(settings.provider, "openai_compatible")
        self.assertEqual(settings.api_key, "test-key")
        self.assertEqual(settings.base_url, "https://api.deepseek.com")

    def test_infers_groq_for_non_deepseek_model(self):
        settings = LLMSettings(
            _env_file=None,
            model="llama-3.3-70b-versatile",
            groq_api_key="groq-key",
        )

        self.assertEqual(settings.provider, "groq")
        self.assertEqual(settings.api_key, "groq-key")

    def test_llm_api_key_takes_precedence(self):
        settings = LLMSettings(
            provider="groq",
            model="llama-3.3-70b-versatile",
            api_key="primary-key",
            groq_api_key="fallback-key",
        )

        self.assertEqual(settings.api_key, "primary-key")

    def test_requires_api_key_for_live_providers(self):
        with self.assertRaises(ValidationError):
            LLMSettings(
                _env_file=None,
                provider="groq",
                model="llama-3.3-70b-versatile",
                groq_api_key=None,
                api_key=None,
            )

    def test_from_env_alias(self):
        with mock.patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "mock",
                "LLM_MODEL": "mock-model",
            },
            clear=False,
        ):
            clear_settings_cache()
            settings = LLMSettings.from_env()

        self.assertEqual(settings.provider, "mock")

    def test_create_provider_uses_mock_from_settings(self):
        settings = LLMSettings(provider="mock")
        provider = create_provider(settings)

        self.assertIsInstance(provider, MockProvider)


class TestSquadAISettings(unittest.TestCase):
    def setUp(self):
        clear_settings_cache()

    def tearDown(self):
        clear_settings_cache()

    def test_defaults(self):
        settings = SquadAISettings()

        self.assertEqual(settings.temporal_address, "localhost:7233")
        self.assertEqual(settings.temporal_task_queue, "squadai")
        self.assertEqual(settings.react_max_iterations, 4)

    def test_reads_temporal_env_vars(self):
        with mock.patch.dict(
            os.environ,
            {
                "TEMPORAL_ADDRESS": "temporal.example.com:7233",
                "TEMPORAL_TASK_QUEUE": "custom-queue",
                "REACT_MAX_ITERATIONS": "8",
            },
            clear=False,
        ):
            clear_settings_cache()
            settings = get_settings()

        self.assertEqual(settings.temporal_address, "temporal.example.com:7233")
        self.assertEqual(settings.temporal_task_queue, "custom-queue")
        self.assertEqual(settings.react_max_iterations, 8)

    def test_get_llm_settings_is_cached(self):
        with mock.patch.dict(
            os.environ,
            {"LLM_PROVIDER": "mock"},
            clear=False,
        ):
            clear_settings_cache()
            first = get_llm_settings()
            second = get_llm_settings()

        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
