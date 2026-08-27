import importlib
from typing import Union


def dynamic_import(module_name: str, attribute_name: str, surrogate: Union[str, None]):
    """
    Dynamically import the components from surrogate module if not available (e.g., `Literal` is only available in
    typing from python3.8 but typing_extension provides the same functionality for python <=3.7.
    Args:
        module_name:
        attribute_name:
        surrogate:

    Returns:

    """
    module = importlib.import_module(module_name)
    attribute = getattr(module, attribute_name, None)
    if attribute is not None:
        return attribute
    if surrogate is not None:
        return dynamic_import(surrogate, attribute_name, None)
    raise ImportError(f"Cannot Import {attribute_name} from either {module_name} or {surrogate}")
