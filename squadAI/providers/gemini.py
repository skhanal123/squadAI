from typing import Any

from squadAI.config import GEMINI_BASE_URL
from squadAI.providers.openai_compatible import OpenAIChatProvider


class GeminiProvider(OpenAIChatProvider):
    """Provider for Gemini via Google's OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        async_client: Any | None = None,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or GEMINI_BASE_URL,
            client=client,
            async_client=async_client,
        )
