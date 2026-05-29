import os
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

load_dotenv()

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def _is_deepseek_model(model: str | None) -> bool:
    return bool(model and model.startswith("deepseek"))


def create_client(llm: str | None = None):
    """
    Create an LLM client for the given model.

    If `llm` is omitted, uses the ``LLM_MODEL`` environment variable.
    DeepSeek models use an OpenAI-compatible client; all others default to Groq.

    Parameters
    ----------
    llm:
        Model name (e.g. ``deepseek-chat``). When omitted, reads ``LLM_MODEL`` from the environment.

    Returns
    -------
    Groq or OpenAI client with a ``chat.completions`` API.
    """
    model = llm or os.getenv("LLM_MODEL")
    if _is_deepseek_model(model):
        return OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=DEEPSEEK_BASE_URL,
        )
    return Groq()
