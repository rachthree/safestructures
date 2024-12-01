"""Main serialization utility."""
import builtins
import importlib
import json
import types
from copy import deepcopy
from dataclasses import is_dataclass
from pathlib import Path, PosixPath
from typing import Any, Optional, Type, Union

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from safestructures.constants import (
    DATACLASS_NAME,
    Mode,
    SCHEMA_FIELD,
    SCHEMA_VERSION,
    TYPE_FIELD,
    VERSION_FIELD,
)
from safestructures.defaults import DEFAULT_PROCESS_MAP
from safestructures.processors.base import DataProcessor
from safestructures.processors.iterable import Dataclass


class Serializer:
    """Serializer for general data structures, using safetensors."""

    def __init__(self, plugins: Optional[list[DataProcessor]] = None):
        """Initialize the serializer."""
        self.tensors = {}

        self.process_map: dict[type, DataProcessor] = deepcopy(DEFAULT_PROCESS_MAP)

        if plugins:
            for p in plugins:
                self._check_plugin(p)
                self.process_map[str(p.data_type)] = p

        self.mode: Optional[Mode] = None

    @staticmethod
    def _get_data_type(type_str: str) -> Type:
        # Handle "None" or "NoneType"
        if type_str in {"None", "NoneType"}:
            return types.NoneType

        if type_str == DATACLASS_NAME:
            return Dataclass

        # Check if the type is a built-in (e.g., "int", "str", "list")
        if hasattr(builtins, type_str):
            return getattr(builtins, type_str)

        # Otherwise, assume it is a fully qualified name (e.g., "a.b.c.some_type")
        try:
            module_name, class_name = type_str.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ValueError, ImportError, AttributeError) as e:
            raise ImportError(f"Cannot import type: {type_str}") from e

    def _check_plugin(self, plugin: DataProcessor):
        """Verify the external plugin.

        Args:
            plugin (DataProcessor): The plugin processor.
        """
        error_msg = (
            f"{plugin} must be a safestructures.DataProcessor"
            " or safestructures.TensorProcessor"
        )
        assert issubclass(plugin, DataProcessor), error_msg

        if plugin.data_type in self.process_map:
            error_msg = (
                f"Processor for {plugin.data_type} already exists."
                "Processors must be unique."
            )
            raise ValueError(error_msg)

    def serialize(self, data: Any) -> dict:
        """Serialize the data so that it can be stored in a safetensors file.

        Args:
            data (Any): The data.

        Returns:
            dict: The schema showing the datatypes and serialized values.
        """
        data_type = type(data)
        if data_type not in self.process_map and is_dataclass(data):
            data_type = Dataclass

        try:
            return self.process_map[data_type](self)(data)
        except KeyError:
            raise TypeError(
                f"Processor for type {data_type} not found."
                " Please create a plugin and use the plugins kwarg for the Serializer."
            )

    def deserialize(self, schema: dict) -> Any:
        """Deserialize the schema, providing the reconstructed data.

        Args:
            schema (dict): The schema showing the datatypes and serialized values.

        Returns:
            Any: The reconstructed data.
        """
        data_type_str = schema[TYPE_FIELD]

        data_type = self._get_data_type(data_type_str)

        try:
            return self.process_map[data_type](self)(schema)
        except KeyError:
            raise TypeError(
                f"Processor for type {data_type} not found."
                " Please create a plugin and use the plugins kwarg for the Serializer."
            )

    def save(
        self,
        data: Any,
        save_path: Union[str, PosixPath],
        metadata: Optional[dict[str, str]] = None,
    ) -> None:
        """Save the data to a safetensors file.

        Args:
            data (Any): The data.
            save_path (Union[str, PosixPath]): Path to save the safetensors file.
            metadata (Optional[dict[str, str]]): Additional metadata to save.
                Keys and values must be strings.
        """
        save_path = Path(save_path).expanduser().resolve()
        self.tensors.clear()
        self.mode = Mode.SAVE
        schema = self.serialize(data)
        if not metadata:
            metadata = {}
        metadata.update(
            {SCHEMA_FIELD: json.dumps(schema), VERSION_FIELD: SCHEMA_VERSION}
        )

        if not self.tensors:
            # In case there are no tensors, provide a dummy tensor.
            # Safetensors requires at least 1 tensor to save.
            self.tensors["null"] = np.array([0])

        save_file(self.tensors, save_path, metadata=metadata)
        return

    def load(
        self,
        load_path: Union[str, PosixPath],
        framework: str = "np",
        device: str = "cpu",
    ) -> Any:
        """Load the safetensors file, reconstructing the data.

        Args:
            load_path (Union[str, PosixPath]): The safetensors file path to load.
            framework (str, optional): The framework to provide for loaded tensors.
                Defaults to "np" to provide `numpy.ndarrays`.
            device (str, optional): The device to host tensors.
                Defaults to "cpu" for CPU.

        Returns:
            Any: The reconstructed data.
        """
        load_path = Path(load_path).expanduser().resolve()
        self.tensors.clear()
        self.mode = Mode.LOAD
        with safe_open(load_path, framework=framework, device=device) as f:
            for k in f.keys():
                self.tensors[k] = f.get_tensor(k)
            metadata = f.metadata()

        try:
            schema = json.loads(metadata[SCHEMA_FIELD])
        except KeyError:
            raise ValueError(
                f"File {load_path} is not a valid safetensors file"
                " to use with safestructures."
            )

        results = self.deserialize(schema)
        return results
