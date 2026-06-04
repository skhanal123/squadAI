from squadAI.config import (
    DEEPSEEK_BASE_URL,
    GROQ_BASE_URL,
    LLMSettings,
    ProviderName,
    get_llm_settings,
)
from squadAI.providers.base import LLMProvider
from squadAI.providers.groq import GroqProvider
from squadAI.providers.mock import MockProvider
from squadAI.providers.openai_compatible import OpenAICompatibleProvider

__all__ = [
    "DEEPSEEK_BASE_URL",
    "GROQ_BASE_URL",
    "LLMSettings",
    "ProviderName",
    "create_client",
    "create_provider",
]


def create_provider(settings: LLMSettings | None = None) -> LLMProvider:
    """Create an LLM provider from settings or environment variables."""
    settings = settings or get_llm_settings()

    if settings.provider == "mock":
        return MockProvider()

    if settings.provider == "groq":
        return GroqProvider(model=settings.model, api_key=settings.api_key)

    return OpenAICompatibleProvider(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )


def _llm_settings_for_model(model: str) -> LLMSettings:
    """Build validated LLM settings for a specific model name."""
    data = get_llm_settings().model_dump()
    data["model"] = model
    if model.startswith("deepseek"):
        data["provider"] = "openai_compatible"
        data["base_url"] = data.get("base_url") or DEEPSEEK_BASE_URL
    else:
        data["provider"] = "groq"
        data["base_url"] = data.get("base_url") or GROQ_BASE_URL
    return LLMSettings(**data)


def create_client(llm: str | None = None):
    """
    Legacy client factory. Prefer ``create_provider()`` for new code.

    Returns a Groq or OpenAI SDK client based on the model name.
    """
    settings = _llm_settings_for_model(llm) if llm else get_llm_settings()
    provider = create_provider(settings)
    return provider.client
