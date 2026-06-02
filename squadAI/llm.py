import os
from typing import Literal

from pydantic import BaseModel
from dotenv import load_dotenv

from squadAI.providers.base import LLMProvider
from squadAI.providers.groq import GroqProvider
from squadAI.providers.mock import MockProvider
from squadAI.providers.openai_compatible import OpenAICompatibleProvider

load_dotenv()

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

ProviderName = Literal["openai_compatible", "groq", "mock"]


class LLMSettings(BaseModel):
    provider: ProviderName = "openai_compatible"
    model: str = "deepseek-chat"
    api_key: str | None = None
    base_url: str | None = None

    @classmethod
    def from_env(cls) -> "LLMSettings":
        model = os.getenv("LLM_MODEL", "deepseek-chat")

        provider_env = os.getenv("LLM_PROVIDER")
        if provider_env in ("openai_compatible", "groq", "mock"):
            provider = provider_env
        elif model.startswith("deepseek"):
            provider = "openai_compatible"
        else:
            provider = "groq"

        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")

        if provider == "openai_compatible":
            api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            base_url = base_url or (
                DEEPSEEK_BASE_URL if model.startswith("deepseek") else None
            )
        elif provider == "groq":
            api_key = api_key or os.getenv("GROQ_API_KEY")
            base_url = base_url or GROQ_BASE_URL

        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )


def create_provider(settings: LLMSettings | None = None) -> LLMProvider:
    """Create an LLM provider from settings or environment variables."""
    settings = settings or LLMSettings.from_env()

    if settings.provider == "mock":
        return MockProvider()

    if settings.provider == "groq":
        return GroqProvider(model=settings.model, api_key=settings.api_key)

    return OpenAICompatibleProvider(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )


def create_client(llm: str | None = None):
    """
    Legacy client factory. Prefer ``create_provider()`` for new code.

    Returns a Groq or OpenAI SDK client based on the model name.
    """
    settings = LLMSettings.from_env()
    if llm:
        settings = settings.model_copy(update={"model": llm})
        if llm.startswith("deepseek"):
            settings = settings.model_copy(
                update={
                    "provider": "openai_compatible",
                    "base_url": DEEPSEEK_BASE_URL,
                    "api_key": settings.api_key or os.getenv("DEEPSEEK_API_KEY"),
                }
            )
        else:
            settings = settings.model_copy(
                update={
                    "provider": "groq",
                    "api_key": settings.api_key or os.getenv("GROQ_API_KEY"),
                }
            )

    provider = create_provider(settings)
    return provider.client
