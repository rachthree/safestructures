"""Import utilities."""
import importlib
from types import ModuleType
from typing import Type

from safestructures.constants import DATACLASS_NAME
from safestructures.utils.dataclass import Dataclass


def load_module(module_name: str) -> ModuleType:
    """Load a specified Python module.

    Args:
        module_name (str): The module name.

    Returns:
        ModuleType: The module.
    """
    return importlib.import_module(module_name, package=None)


def is_available(module_name: str) -> bool:
    """Check if a Python module is available.

    Args:
        module_name (str): The module name.

    Returns:
        bool: True if available, False if not.
    """
    return importlib.util.find_spec(module_name)


def get_import_path(data_type: Type):
    """Provide the full Python import path for a data type (class)."""
    if data_type is Dataclass:
        return DATACLASS_NAME

    module = data_type.__module__
    name = data_type.__qualname__
    if module == "builtins":
        return name  # No need to import built-in types
    return f"{module}.{name}"
