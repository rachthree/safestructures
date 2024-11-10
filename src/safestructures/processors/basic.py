"""Plugins to process basic data types."""
import numbers
from pydoc import locate
from typing import Union

from safestructures.constants import TYPE_FIELD, VALUE_FIELD
from safestructures.processors.base import DataProcessor

BASIC_TYPES = (str, numbers.Number, type(None), bool)


class BasicProcessor(DataProcessor):
    """Processor for basic datatypes."""

    def serialize(self, value: Union[str, numbers.Number, None, bool]):
        """Overload `DataProcessor.serialize`.

        For simple data types other than None and boolean, this
        casts to a string to be stored in the metadata, as is
        required by safetensors.
        """
        value_type = type(value).__name__
        if value is not None and not isinstance(value_type, bool):
            value = str(value)

        return {TYPE_FIELD: value_type, VALUE_FIELD: value}

    def deserialize(self, schema: dict):
        """Overload `DataProcessor.deserialize`."""
        if schema[VALUE_FIELD] is None:
            return None

        t = locate(schema[TYPE_FIELD])
        return t(schema[VALUE_FIELD])
