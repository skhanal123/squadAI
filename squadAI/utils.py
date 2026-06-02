import inspect
import json
from typing import Callable, get_args, get_origin


JSON_TYPE_MAP = {
    "int": "integer",
    "float": "number",
    "str": "string",
    "bool": "boolean",
}


def _annotation_to_json_schema(annotation) -> dict:
    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)
        if origin is list:
            item_schema = _annotation_to_json_schema(args[0]) if args else {"type": "string"}
            return {"type": "array", "items": item_schema}
        if str(origin) in ("typing.Union", "types.UnionType"):
            non_none = [arg for arg in args if arg is not type(None)]
            if len(non_none) == 1:
                return _annotation_to_json_schema(non_none[0])

    if hasattr(annotation, "__name__"):
        json_type = JSON_TYPE_MAP.get(annotation.__name__)
        if json_type:
            return {"type": json_type}

    return {"type": "string"}


def get_fn_signature(fn: Callable) -> str:
    """
    Build an OpenAI-compatible function signature for a Python callable.

    Parameters
    ----------
    fn:
        Input function with type annotations and a docstring.

    Returns
    -------
    JSON string describing name, description, and parameters schema.
    """
    sig = inspect.signature(fn)
    properties: dict = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if param.annotation is not inspect.Parameter.empty:
            properties[name] = _annotation_to_json_schema(param.annotation)
        else:
            properties[name] = {"type": "string"}

        if param.default is inspect.Parameter.empty:
            required.append(name)

    fn_signature = {
        "name": fn.__name__,
        "description": (fn.__doc__ or "").strip(),
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }

    return json.dumps(fn_signature)
