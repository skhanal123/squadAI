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
from squadAI.config import infer_provider_from_model
from squadAI.llm import create_provider
from squadAI.providers.anthropic import AnthropicProvider
from squadAI.providers.deepseek import DeepSeekProvider
from squadAI.providers.gemini import GeminiProvider
from squadAI.providers.mock import MockProvider
from squadAI.providers.openai import OpenAIProvider


class TestLLMSettings(unittest.TestCase):
    def setUp(self):
        clear_settings_cache()

    def tearDown(self):
        clear_settings_cache()

    def test_mock_provider_needs_no_api_key(self):
        settings = LLMSettings(provider="mock", model="mock-model")

        self.assertEqual(settings.provider, "mock")
        self.assertIsNone(settings.api_key)

    def test_infers_deepseek_for_deepseek_model(self):
        settings = LLMSettings(
            _env_file=None,
            model="deepseek-chat",
            deepseek_api_key="test-key",
        )

        self.assertEqual(settings.provider, "deepseek")
        self.assertEqual(settings.api_key, "test-key")
        self.assertEqual(settings.base_url, "https://api.deepseek.com")

    def test_infers_openai_for_gpt_model(self):
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("LLM_API_KEY", "OPENAI_API_KEY")
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = LLMSettings(
                _env_file=None,
                model="gpt-4o",
                openai_api_key="openai-key",
            )

        self.assertEqual(settings.provider, "openai")
        self.assertEqual(settings.api_key, "openai-key")
        self.assertEqual(settings.base_url, "https://api.openai.com/v1")

    def test_infers_gemini_for_gemini_model(self):
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("LLM_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = LLMSettings(
                _env_file=None,
                model="gemini-2.0-flash",
                gemini_api_key="gemini-key",
            )

        self.assertEqual(settings.provider, "gemini")
        self.assertEqual(settings.api_key, "gemini-key")
        self.assertEqual(
            settings.base_url,
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    def test_infers_anthropic_for_claude_model(self):
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("LLM_API_KEY", "ANTHROPIC_API_KEY")
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = LLMSettings(
                _env_file=None,
                model="claude-sonnet-4-20250514",
                anthropic_api_key="anthropic-key",
            )

        self.assertEqual(settings.provider, "anthropic")
        self.assertEqual(settings.api_key, "anthropic-key")

    def test_gemini_accepts_google_api_key(self):
        settings = LLMSettings(
            _env_file=None,
            provider="gemini",
            model="gemini-2.0-flash",
            google_api_key="google-key",
        )

        self.assertEqual(settings.api_key, "google-key")

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

    def test_create_provider_openai(self):
        settings = LLMSettings(
            provider="openai",
            model="gpt-4o",
            api_key="key",
        )
        provider = create_provider(settings)
        self.assertIsInstance(provider, OpenAIProvider)

    def test_create_provider_gemini(self):
        settings = LLMSettings(
            provider="gemini",
            model="gemini-2.0-flash",
            api_key="key",
        )
        provider = create_provider(settings)
        self.assertIsInstance(provider, GeminiProvider)

    def test_create_provider_deepseek(self):
        settings = LLMSettings(
            provider="deepseek",
            model="deepseek-chat",
            api_key="key",
        )
        provider = create_provider(settings)
        self.assertIsInstance(provider, DeepSeekProvider)

    def test_create_provider_anthropic(self):
        settings = LLMSettings(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="key",
        )
        provider = create_provider(settings)
        self.assertIsInstance(provider, AnthropicProvider)

    def test_infer_provider_from_model(self):
        self.assertEqual(infer_provider_from_model("deepseek-chat"), "deepseek")
        self.assertEqual(infer_provider_from_model("gemini-2.0-flash"), "gemini")
        self.assertEqual(
            infer_provider_from_model("claude-sonnet-4-20250514"), "anthropic"
        )
        self.assertEqual(infer_provider_from_model("gpt-4o"), "openai")
        self.assertEqual(infer_provider_from_model("llama-3.3-70b"), "groq")


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
