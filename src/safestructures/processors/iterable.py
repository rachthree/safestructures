"""Plugins to process iterables, including dataclasses."""

import dataclasses
from typing import Any, Protocol, runtime_checkable

from safestructures.constants import TYPE_FIELD, VALUE_FIELD
from safestructures.processors.base import DataProcessor, ListBaseProcessor


class ListProcessor(ListBaseProcessor):
    """Processor for list data."""

    data_type: list


class SetProcessor(ListBaseProcessor):
    """Processor for set data."""

    data_type: set


class TupleProcessor(ListBaseProcessor):
    """Processor for tuple data."""

    data_type: tuple


class DictProcessor(DataProcessor):
    """Processor for dictionary data."""

    data_type = dict

    def serialize(self, data: Any):
        """Overload `DataProcessor.serialize`."""
        schema = {TYPE_FIELD: self.data_type.__name__, VALUE_FIELD: {}}
        for k, v in data.items():
            schema[VALUE_FIELD][k] = self.serializer.serialize(v)

        return schema

    def deserialize(self, schema: dict) -> dict:
        """Overload `DataProcessor.deserialize`."""
        results = {}
        for k, v in schema[VALUE_FIELD].items():
            results[k] = self.serializer.deserialize(v)

        return results


@runtime_checkable
@dataclasses.dataclass
class Dataclass(Protocol):
    """Protocol to help provide a "dataclass" type."""

    pass


class DataclassProcessor(DataProcessor):
    """Processor for dataclass data."""

    def serialize(self, data: Any):
        """Overload `DataProcessor.serialize`."""
        fields = dataclasses.fields(data)
        schema = {TYPE_FIELD: self.data_type.__name__, VALUE_FIELD: {}}
        for f in fields:
            name = f.name
            schema[VALUE_FIELD][name] = self.serializer.serialize(getattr(data, name))

        return schema

    def deserialize(self, schema: dict):
        """Overload `DataProcessor.deserialize`."""
        fields = list(schema[VALUE_FIELD].keys())
        cls = dataclasses.make_dataclass("Dataclass", fields)
        kwargs = {}
        for k, v in schema[VALUE_FIELD].items():
            kwargs[k] = self.serializer.deserialize(v)

        return cls(**kwargs)
