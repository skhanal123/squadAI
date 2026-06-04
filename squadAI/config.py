"""Centralized configuration loaded via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal[
    "openai_compatible",
    "groq",
    "mock",
    "openai",
    "gemini",
    "anthropic",
    "deepseek",
]

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

_ENV_CONFIG = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
)


def infer_provider_from_model(model: str) -> ProviderName:
    """Infer provider from model name when ``LLM_PROVIDER`` is unset."""
    lower = model.lower()
    if lower.startswith("deepseek"):
        return "deepseek"
    if lower.startswith("gemini"):
        return "gemini"
    if lower.startswith("claude"):
        return "anthropic"
    if lower.startswith(("gpt-", "o1-", "o3-", "o4-", "chatgpt-")):
        return "openai"
    return "groq"


class LLMSettings(BaseSettings):
    """LLM provider configuration.

    Environment variables (also read from ``.env``):

    - ``LLM_PROVIDER`` — provider id (see ``ProviderName``)
    - ``LLM_MODEL`` — model name (default: ``deepseek-chat``)
    - ``LLM_API_KEY`` — generic API key override
    - ``LLM_BASE_URL`` — optional base URL override
    - ``OPENAI_API_KEY``, ``GEMINI_API_KEY``, ``ANTHROPIC_API_KEY``
    - ``DEEPSEEK_API_KEY``, ``GROQ_API_KEY`` — legacy per-provider keys
    """

    model_config = SettingsConfigDict(
        **_ENV_CONFIG,
        env_prefix="LLM_",
        populate_by_name=True,
    )

    provider: ProviderName | None = None
    model: str = "deepseek-chat"
    api_key: str | None = None
    base_url: str | None = None
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    gemini_api_key: str | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    google_api_key: str | None = Field(default=None, validation_alias="GOOGLE_API_KEY")
    anthropic_api_key: str | None = Field(
        default=None, validation_alias="ANTHROPIC_API_KEY"
    )
    deepseek_api_key: str | None = Field(
        default=None, validation_alias="DEEPSEEK_API_KEY"
    )
    groq_api_key: str | None = Field(default=None, validation_alias="GROQ_API_KEY")

    @classmethod
    def from_env(cls) -> LLMSettings:
        """Load settings from environment variables and ``.env``."""
        return cls()

    @model_validator(mode="after")
    def resolve_provider_and_credentials(self) -> Self:
        provider = self.provider
        if provider is None:
            provider = infer_provider_from_model(self.model)
            object.__setattr__(self, "provider", provider)

        api_key = self.api_key
        base_url = self.base_url

        if provider == "openai":
            api_key = api_key or self.openai_api_key
            base_url = base_url or OPENAI_BASE_URL
        elif provider == "gemini":
            api_key = api_key or self.gemini_api_key or self.google_api_key
            base_url = base_url or GEMINI_BASE_URL
        elif provider == "anthropic":
            api_key = api_key or self.anthropic_api_key
        elif provider == "deepseek":
            api_key = api_key or self.deepseek_api_key
            base_url = base_url or DEEPSEEK_BASE_URL
        elif provider == "openai_compatible":
            api_key = api_key or self.deepseek_api_key or self.openai_api_key
            if base_url is None and self.model.lower().startswith("deepseek"):
                base_url = DEEPSEEK_BASE_URL
        elif provider == "groq":
            api_key = api_key or self.groq_api_key
            base_url = base_url or GROQ_BASE_URL

        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "base_url", base_url)

        if provider != "mock" and not api_key:
            raise ValueError(
                f"An API key is required for provider {provider!r}. "
                "Set LLM_API_KEY or the provider-specific key "
                "(OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, "
                "DEEPSEEK_API_KEY, GROQ_API_KEY)."
            )

        return self


class SquadAISettings(BaseSettings):
    """Top-level SquadAI runtime configuration."""

    model_config = _ENV_CONFIG

    temporal_address: str = "localhost:7233"
    temporal_task_queue: str = "squadai"
    react_max_iterations: int = Field(default=4, ge=1)


@lru_cache
def get_llm_settings() -> LLMSettings:
    """Return cached LLM settings loaded from the environment."""
    return LLMSettings()


@lru_cache
def get_settings() -> SquadAISettings:
    """Return cached SquadAI settings loaded from the environment."""
    return SquadAISettings()


def clear_settings_cache() -> None:
    """Clear cached settings (useful in tests)."""
    get_llm_settings.cache_clear()
    get_settings.cache_clear()
