import os
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

load_dotenv()


def create_client(llm=None):
    """
    This method creates the client of an llm model. The default client would be an llm model from Groq.

    Arguments:
    ----------
    llm: name of the llm model

    Returns:
    --------
    client of an llm model

    Arg
    """
    if not llm:
        return Groq()
    elif llm == "deepseek-chat":
        return OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
        )
