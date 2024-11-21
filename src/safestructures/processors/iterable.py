"""Plugins to process iterables, including dataclasses."""

import dataclasses
from typing import Protocol, runtime_checkable

from safestructures.constants import KEYS_FIELD
from safestructures.processors.base import DataProcessor, ListBaseProcessor


class ListProcessor(ListBaseProcessor):
    """Processor for list data."""

    data_type = list


class SetProcessor(ListBaseProcessor):
    """Processor for set data."""

    data_type = set


class TupleProcessor(ListBaseProcessor):
    """Processor for tuple data."""

    data_type = tuple


class DictProcessor(DataProcessor):
    """Processor for dictionary data."""

    data_type = dict

    def serialize(self, data: dict) -> dict:
        """Overload `DataProcessor.serialize`."""
        results = {}
        for k, v in data.items():
            key = str(k)
            results[key] = self.serializer.serialize(v)

        return results

    def serialize_extra(self, data: dict) -> dict:
        """Overload `DataProcessor.serialize_extra`.

        The additional schema to provide helps serialize
        the dictionary keys themselves.
        """
        # Keys can be numerical or tuple, not just strings.
        schema = {KEYS_FIELD: {}}
        for k in data:
            key = str(k)
            schema[KEYS_FIELD][key] = self.serializer.serialize(k)
        return schema

    def deserialize(self, serialized: dict, **kwargs) -> dict:
        """Overload `DataProcessor.deserialize`."""
        results = {}
        key_schemas = kwargs[KEYS_FIELD]
        for k, v in serialized.items():
            key_schema = key_schemas[k]
            key = self.serializer.deserialize(key_schema)
            results[key] = self.serializer.deserialize(v)

        return results


@runtime_checkable
@dataclasses.dataclass
class Dataclass(Protocol):
    """Protocol to help provide a 'dataclass' type."""

    pass


class DataclassProcessor(DataProcessor):
    """Processor for dataclass data."""

    data_type = Dataclass

    def get_schema_type(self, *, data: Dataclass = None) -> str:
        """Overload `DataProcessor.get_schema_type`.

        Provide a consistent schema type for all dataclasses.
        """
        return self.data_type.__name__

    def serialize(self, data: Dataclass) -> dict:
        """Overload `DataProcessor.serialize`."""
        fields = dataclasses.fields(data)
        results = {}
        for f in fields:
            name = f.name
            results[name] = self.serializer.serialize(getattr(data, name))

        return results

    def deserialize(self, serialized: dict, **kwargs) -> Dataclass:
        """Overload `DataProcessor.deserialize`."""
        fields = list(serialized.keys())
        cls = dataclasses.make_dataclass("Dataclass", fields)
        dc_kwargs = {}
        for k, v in serialized.items():
            dc_kwargs[k] = self.serializer.deserialize(v)

        return cls(**dc_kwargs)
