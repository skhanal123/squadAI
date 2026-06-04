from squadAI.config import (
    DEEPSEEK_BASE_URL,
    GEMINI_BASE_URL,
    GROQ_BASE_URL,
    LLMSettings,
    OPENAI_BASE_URL,
    ProviderName,
    get_llm_settings,
    infer_provider_from_model,
)
from squadAI.providers.anthropic import AnthropicProvider
from squadAI.providers.base import LLMProvider
from squadAI.providers.deepseek import DeepSeekProvider
from squadAI.providers.gemini import GeminiProvider
from squadAI.providers.groq import GroqProvider
from squadAI.providers.mock import MockProvider
from squadAI.providers.openai import OpenAIProvider
from squadAI.providers.openai_compatible import OpenAICompatibleProvider

__all__ = [
    "DEEPSEEK_BASE_URL",
    "GEMINI_BASE_URL",
    "GROQ_BASE_URL",
    "LLMSettings",
    "OPENAI_BASE_URL",
    "ProviderName",
    "create_client",
    "create_provider",
    "infer_provider_from_model",
]


def create_provider(settings: LLMSettings | None = None) -> LLMProvider:
    """Create an LLM provider from settings or environment variables."""
    settings = settings or get_llm_settings()

    if settings.provider == "mock":
        return MockProvider()

    if settings.provider == "groq":
        return GroqProvider(model=settings.model, api_key=settings.api_key)

    if settings.provider == "openai":
        return OpenAIProvider(
            model=settings.model,
            api_key=settings.api_key,
            base_url=settings.base_url,
        )

    if settings.provider == "gemini":
        return GeminiProvider(
            model=settings.model,
            api_key=settings.api_key,
            base_url=settings.base_url,
        )

    if settings.provider == "deepseek":
        return DeepSeekProvider(
            model=settings.model,
            api_key=settings.api_key,
            base_url=settings.base_url,
        )

    if settings.provider == "anthropic":
        return AnthropicProvider(model=settings.model, api_key=settings.api_key)

    return OpenAICompatibleProvider(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )


def _llm_settings_for_model(model: str) -> LLMSettings:
    """Build validated LLM settings for a specific model name."""
    data = get_llm_settings().model_dump()
    data["model"] = model
    provider = infer_provider_from_model(model)
    data["provider"] = provider

    if provider == "deepseek":
        data["base_url"] = data.get("base_url") or DEEPSEEK_BASE_URL
    elif provider == "openai":
        data["base_url"] = data.get("base_url") or OPENAI_BASE_URL
    elif provider == "gemini":
        data["base_url"] = data.get("base_url") or GEMINI_BASE_URL
    elif provider == "groq":
        data["base_url"] = data.get("base_url") or GROQ_BASE_URL

    return LLMSettings(**data)


def create_client(llm: str | None = None):
    """
    Legacy client factory. Prefer ``create_provider()`` for new code.

    Returns the underlying SDK client for the configured provider.
    """
    settings = _llm_settings_for_model(llm) if llm else get_llm_settings()
    provider = create_provider(settings)
    return provider.client
