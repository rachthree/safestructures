"""Main serialization utility."""
import json
from copy import deepcopy
from pathlib import Path, PosixPath
from pydoc import locate
from typing import Any, Optional, Union

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from safestructures.constants import (
    DEFAULT_PROCESS_MAP,
    SCHEMA_FIELD,
    SCHEMA_VERSION,
    TYPE_FIELD,
    VERSION_FIELD,
)
from safestructures.processors.base import DataProcessor


class Serializer:
    """Serializer for general data structures, using safetensors."""

    def __init__(self, plugins: Optional[list[DataProcessor]] = None):
        """Initialize the serializer."""
        self.tensors = {}

        self.process_map: dict[type, DataProcessor] = deepcopy(DEFAULT_PROCESS_MAP)
        if plugins:
            for p in plugins:
                self._check_plugin(p)
                self.process_map[p.data_type] = p

    def _check_plugin(self, plugin: DataProcessor):
        """Verify the external plugin.

        Args:
            plugin (DataProcessor): The plugin processor.
        """
        error_msg = (
            f"{plugin} must be a safestructures.DataProcessor"
            " or safestructures.TensorProcessor"
        )
        assert isinstance(plugin, DataProcessor), error_msg

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
        try:
            return self.process_map[data_type].serialize(data)
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
        data_type = locate(schema[TYPE_FIELD])
        try:
            return self.process_map[data_type].deserialize(schema)
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
        self.tensors.clear()
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
        with safe_open(load_path, framework=framework, device=device) as f:
            tensors = {}
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
            metadata = f.metadata()

        try:
            schema = json.loads(metadata[SCHEMA_FIELD])
        except KeyError:
            raise ValueError(
                f"File {load_path} is not a valid safetensors file"
                " to use with safestructures."
            )

        results = self.deserialize(schema)
        self.tensors.clear()
        return results
