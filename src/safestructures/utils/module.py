"""Import utilities."""
import importlib
from types import ModuleType


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
