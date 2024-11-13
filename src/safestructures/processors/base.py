"""Base classes for plugins."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pydoc import locate
from typing import Any, Type, TYPE_CHECKING, Union

import numpy as np

from safestructures.constants import TYPE_FIELD, VALUE_FIELD

if TYPE_CHECKING:
    from safestructures import Serializer


class DataProcessor(ABC):
    """Base class for data processors other than tensors."""

    data_type: Type[Any]

    def __init__(self, serializer: Serializer):
        """Initialize the DataProcessor.

        Args:
            serializer (Serializer): The serializer.
                This may be used to help recursively process.
        """
        self.serializer = serializer

    @abstractmethod
    def serialize(self, data: Any) -> dict:
        """Serialize the data.

        Args:
            data (Any): The data to serialize.

        Returns:
            dict: The schema. This consists of keys:
                * `TYPE_FIELD` ("type"): The data type.
                * `VALUE_FIELD` ("value"): the serialized value.
        """
        pass

    @abstractmethod
    def deserialize(self, schema: dict) -> Any:
        """Deserialize the schema into data.

        Args:
            schema (dict): The schema.

        Returns:
            Any: The loaded value.
        """
        pass


class ListBaseProcessor(DataProcessor):
    """Base class to process list-like data."""

    def serialize(self, data: Union[list, set, tuple]) -> dict:
        """Overload `DataProcessor.serialize`."""
        data_list = []
        for d in data:
            data_list.append(self.serializer.serialize(d))

        return {TYPE_FIELD: self.data_type.__name__, VALUE_FIELD: data_list}

    def deserialize(self, schema: dict) -> list:
        """Overload `DataProcessor.deserialize`."""
        results = []
        t = locate(schema[TYPE_FIELD])
        for v in schema[VALUE_FIELD]:
            results.append(self.serializer.deserialize(v))

        return t(results)


class TensorProcessor(DataProcessor, ABC):
    """Base class to process tensors."""

    @abstractmethod
    def to_cpu(self, tensor: Any) -> Any:
        """Move tensor to CPU."""
        pass

    @abstractmethod
    def to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert tensor to Numpy array."""
        pass

    def process_tensor(self, tensor: Any) -> dict:
        """Process a tensor for serialization.

        Args:
            tensor (Any): The tensor to serialize.

        Returns:
            dict: The schema to describe the tensor, namely the type
                and its ID which will be used as a key in safetensors.
        """
        _id = str(len(self.serializer.tensors))
        self.serializer.tensors[_id] = self.serialize(tensor)
        return {TYPE_FIELD: self.data_type.__name__, VALUE_FIELD: _id}

    def serialize(self, tensor: Any) -> dict:
        """Overload `DataProcessor.serialize`."""
        tensor = self.to_cpu(tensor)
        tensor = self.to_numpy(tensor)
        return self.process_tensor(tensor)

    def deserialize(self, schema: dict) -> Any:
        """Overload `DataProcessor.deserialize`."""
        tensor_id = schema[VALUE_FIELD]
        return self.serializer.tensors[tensor_id]
