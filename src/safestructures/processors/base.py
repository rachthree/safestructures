"""Base classes for plugins."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Type, TYPE_CHECKING, Union

import numpy as np

from safestructures.constants import Mode, TYPE_FIELD, VALUE_FIELD
from safestructures.utils.module import get_import_path

if TYPE_CHECKING:
    from safestructures import Serializer


class DataProcessor(ABC):
    """Base class for data processors other than tensors."""

    data_type: Type[Any] = Any

    def __init__(self, serializer: Serializer):
        """Initialize the DataProcessor.

        Args:
            serializer (Serializer): The serializer.
                This may be used to help recursively process.
        """
        self.serializer = serializer

    def get_schema_type(self, *, data: Any = None) -> str:
        """Provide the data type in a compatible format for the schema."""
        return get_import_path(self.data_type)

    @abstractmethod
    def serialize(self, data: Any) -> Union[str, None, bool, list, dict]:
        """Serialize the data.

        Args:
            data (Any): The data to serialize.

        Returns:
            Union[str, None, bool, list, dict]: The serialized value.
                Safetensors only accepts strings in the metadata.
                json.dumps is used, so None and booleans are handled.
                Lists and dictionaries of accepted data types as indicated
                here are acceptable, including nested types.
        """
        pass

    def serialize_extra(self, data: Any) -> dict:
        """Provide extra serialization details to the schema.

        Args:
            data (Any): The data to generate additional schema.

        Returns:
            dict: The additional schema to add onto the data's schema.
                The keys must not conflict with TYPE_FIELD and VALUE_FIELD,
                and must be strings.
                The values must also be of acceptable type to Safetensors metadata.
                See `DataProcessor.serialize`
        """
        return {}

    @abstractmethod
    def deserialize(
        self, serialized: Union[str, None, bool, list, dict], **kwargs
    ) -> Any:
        """Deserialize the schema into data.

        Args:
            schema (Any): The serialized value.

        Any additional schema other than VALUE_FIELD will be
        passed as keyword arguments.

        Returns:
            Any: The loaded value.
        """
        pass

    def __call__(self, data_or_schema: Any) -> Any:
        """Process the data or schema.

        Args:
            data_or_schema (Any): The data (Any) or schema (dict).

        Returns:
            The schema if the serializer is in save mode,
            The loaded data if the serializer is in load mode.
        """
        mode = self.serializer.mode
        if mode == Mode.SAVE:
            schema = {TYPE_FIELD: self.get_schema_type(data=data_or_schema)}

            schema[VALUE_FIELD] = self.serialize(data_or_schema)
            extra = self.serialize_extra(data_or_schema)

            if not isinstance(extra, dict):
                raise TypeError(
                    f"{type(self)}.serialize_extra must return a dictionary."
                )
            if TYPE_FIELD in extra:
                raise KeyError(
                    f"{type(self)}.serialize_extra must not have a {TYPE_FIELD} key."
                )
            if VALUE_FIELD in extra:
                raise KeyError(
                    f"{type(self)}.serialize_extra must not have a {VALUE_FIELD} key."
                )

            for k in extra.keys():
                if not isinstance(k, str):
                    raise TypeError(
                        (
                            f"Dictionary returned by {type(self)}.serialize_extra"
                            " must have string keys only."
                        )
                    )
            schema.update(extra)

            return schema

        elif mode == Mode.LOAD:
            kwargs = {}
            for k in data_or_schema:
                if k not in [VALUE_FIELD]:
                    kwargs[k] = data_or_schema[k]
            return self.deserialize(data_or_schema[VALUE_FIELD], **kwargs)

        else:
            raise ValueError(f"Mode {mode} not recognized.")


class ListBaseProcessor(DataProcessor):
    """Base class to process list-like data."""

    def serialize(self, data: Union[list, set, tuple]) -> dict:
        """Overload `DataProcessor.serialize`."""
        data_list = []
        for d in data:
            data_list.append(self.serializer.serialize(d))

        return data_list

    def deserialize(self, serialized: list, **kwargs) -> list:
        """Overload `DataProcessor.deserialize`."""
        results = []
        for v in serialized:
            results.append(self.serializer.deserialize(v))

        return self.data_type(results)


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

    def process_tensor(self, tensor: Any) -> str:
        """Process a tensor for serialization.

        Args:
            tensor (Any): The tensor to serialize.

        Returns:
            dict: The schema to describe the tensor, namely the type
                and its ID which will be used as a key in safetensors.
        """
        _id = str(len(self.serializer.tensors))
        self.serializer.tensors[_id] = self.serialize(tensor)
        return _id

    def serialize(self, tensor: Any) -> dict:
        """Overload `DataProcessor.serialize`."""
        tensor = self.to_cpu(tensor)
        tensor = self.to_numpy(tensor)
        return self.process_tensor(tensor)

    def deserialize(self, tensor_id: str, **kwargs) -> Any:
        """Overload `DataProcessor.deserialize`."""
        return self.serializer.tensors[tensor_id]
