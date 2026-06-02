from pydantic import BaseModel
from typing import Callable
import json
from squadAI.utils import get_fn_signature


class Tool(BaseModel):
    """
    Base class to create a tool which are assigned to agent.

    Attributes:
    -----------
    function_name: name of the function which is used to create the tool
    fn: function itself
    fn_signature: dictionary about the details(name, arguments, etc.) of the funtions

    Methods:
    run: this method accepts the key, value as kwargs and runs the function
    to_openai_schema: returns OpenAI-compatible tool schema for native tool calling
    """

    function_name: str
    fn: Callable
    fn_signature: dict

    def run(self, **kwargs):
        return self.fn(**kwargs)

    def to_openai_schema(self) -> dict:
        """Return an OpenAI-compatible tool definition for native tool calling."""
        parameters = self.fn_signature.get("parameters", {})
        if "type" not in parameters:
            parameters = {
                "type": "object",
                "properties": parameters.get("properties", {}),
                "required": parameters.get("required", []),
            }

        return {
            "type": "function",
            "function": {
                "name": self.function_name,
                "description": self.fn_signature.get("description") or "",
                "parameters": parameters,
            },
        }


def tool_wrapper(fn: Callable):
    """
    This function is a wrapper to convert the function into tool.

    Parameters
    ----------
    fn(callable): Function to be converted into Tool

    Returns
    -------
    Instance of Tool
    """

    def wrapper():
        fn_signature = json.loads(get_fn_signature(fn))
        fn_name = fn_signature["name"]
        return Tool(function_name=fn_name, fn=fn, fn_signature=fn_signature)

    return wrapper()
