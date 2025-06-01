from typing import Callable
import json


def get_fn_signature(fn: Callable):
    """
    This function provides the details about the input function as its signature

    Parameters:
    -----------
        fn (Callable): input function

    Returns:
    --------
        dict: The signature of the input function in the dictionary format
    """

    fn_signature = {
        "name": fn.__name__,
        "description": fn.__doc__,
        "parameters": {"properties": {}},
    }

    input_schema = {k: {"type": v.__name__} for k, v in fn.__annotations__.items()}

    fn_signature["parameters"]["properties"] = input_schema

    json_fn_signature = json.dumps(fn_signature)

    return json_fn_signature
