from typing import Any

from squadAI.config import DEEPSEEK_BASE_URL
from squadAI.providers.openai_compatible import OpenAIChatProvider


class DeepSeekProvider(OpenAIChatProvider):
    """Provider for the DeepSeek API."""

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
            base_url=base_url or DEEPSEEK_BASE_URL,
            client=client,
            async_client=async_client,
        )
