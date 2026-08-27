from .helper import dynamic_import
from typing import Type, Callable, Tuple, Any, TypeVar

__TYPE_GET_ARGS = Callable[[Type, ], Tuple[Any, ...]]
Literal: TypeVar = dynamic_import("typing", "Literal", "typing_extensions")
get_args: __TYPE_GET_ARGS = dynamic_import("typing", "get_args", "typing_extensions")
