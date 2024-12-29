"""Wrapper functions for end-users."""

from pathlib import PosixPath
from typing import Any, Optional, Union

from safestructures.processors.base import DataProcessor, TensorProcessor
from safestructures.serializer import Serializer

BASE_PROCESSORS = (TensorProcessor, DataProcessor)
PROCESSOR_TYPES = Union[TensorProcessor, DataProcessor]


def save_file(
    data: Any,
    save_path: Union[str, PosixPath],
    metadata: Optional[dict[str, str]] = None,
    plugins: Optional[Union[PROCESSOR_TYPES, list[PROCESSOR_TYPES]]] = None,
):
    """Save data using Safestructures.

    Args:
        data (Any): Data to save.
            Data must use data types that are serializable
            by Safetensors/Safestructures.
        save_path (Union[str, PosixPath]): Path to save the data.
            Directory must already exist.
        metadata (Optional[dict[str, str]], optional): Additional metadata to save.
            Defaults to None for no metadata.
        plugins (Optional[Union[PROCESSOR_TYPES, list[PROCESSOR_TYPES]]], optional):
            Additional plugins to serialize data for data types not covered
            by safestructures. Defaults to None for no plugins.

    Returns:
        None.
    """
    if plugins and not isinstance(plugins, list):
        plugins = [plugins]

    return Serializer(plugins=plugins).save(data, save_path, metadata=metadata)


def load_file(
    load_path: Union[str, PosixPath],
    framework: str = "np",
    device: str = "cpu",
    plugins: Optional[Union[PROCESSOR_TYPES, list[PROCESSOR_TYPES]]] = None,
):
    """Load data using Safestructures.

    The file must be a valid Safestructures file,
    i.e Safetensors file with additional Safestructures metadata.

    Args:
        load_path (Union[str, PosixPath]): Path to a valid Safestructures file.
        framework (str, optional): The framework to load tensors into.
            Defaults to "np" for Numpy.
        device (str, optional): Device to allocate tensors to.
            Defaults to "cpu" to allocate on CPU.
        plugins (Optional[Union[PROCESSOR_TYPES, list[PROCESSOR_TYPES]]], optional):
            Additional plugins to serialize data for data types not covered
            by safestructures. Defaults to None for no plugins.

    Returns:
        The loaded data.
    """
    if plugins and not isinstance(plugins, list):
        plugins = [plugins]

    return Serializer(plugins=plugins).load(
        load_path, framework=framework, device=device
    )
