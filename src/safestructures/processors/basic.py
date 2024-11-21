"""Plugins to process basic data types."""
import numbers
from pydoc import locate
from typing import Union

from safestructures.constants import TYPE_FIELD
from safestructures.processors.base import DataProcessor

BASIC_TYPES = (str, numbers.Number, type(None), bool)


class BasicProcessor(DataProcessor):
    """Processor for basic datatypes."""

    def get_schema_type(self, *, data: Union[str, numbers.Number, None, bool]) -> str:
        """Overload `DataProcessor.get_schema_type`.

        Provide the actual type of the basic type rather than just `Any`.
        """
        return type(data).__name__

    def serialize(
        self, value: Union[str, numbers.Number, None, bool]
    ) -> Union[str, None, bool]:
        """Overload `DataProcessor.serialize`.

        For simple data types other than None and boolean, this
        casts to a string to be stored in the metadata, as is
        required by safetensors.
        """
        if value is not None and not isinstance(value, bool):
            value = str(value)

        return value

    def deserialize(
        self, serialized: Union[str, None, bool], **kwargs
    ) -> Union[str, numbers.Number, None, bool]:
        """Overload `DataProcessor.deserialize`."""
        if serialized is None:
            return None

        t = locate(kwargs[TYPE_FIELD])
        return t(serialized)
