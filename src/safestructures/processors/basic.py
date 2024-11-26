"""Plugins to process basic data types."""
import numbers

from safestructures.processors.base import DataProcessor


class NumberProcessor(DataProcessor):
    """Processor for numeric datatypes."""

    def serialize(self, value: numbers.Number) -> str:
        """Overload `DataProcessor.serialize`.

        For numeric datatypes, this casts to a string to be stored
        in the metadata, as is required by safetensors.
        """
        return str(value)

    def deserialize(self, serialized: str) -> numbers.Number:
        """Overload `DataProcessor.deserialize`."""
        return self.data_type(serialized)


class IntProcessor(NumberProcessor):
    """Int processor."""

    data_type = int


class FloatProcessor(NumberProcessor):
    """Float processor."""

    data_type = float


class ComplexProcessor(NumberProcessor):
    """Complex number processor."""

    data_type = complex


class PassthroughProcessor(DataProcessor):
    """Processor that passes through compatible values.

    Values are already compatible with JSON + Safetensors.
    """

    def serialize(self, value: str | bool | None) -> str | bool | None:
        """Overload `DataProcessor.serialize`."""
        # value is already compatible, just pass through
        return value

    def deserialize(self, serialized: str | bool | None) -> str | bool | None:
        """Overload `DataProcessor.deserialize`."""
        # value is already deserialized, just pass through
        return serialized


class StringProcessor(PassthroughProcessor):
    """String processor."""

    data_type = str


class BoolProcessor(PassthroughProcessor):
    """Boolean processor."""

    data_type = bool


class NoneProcessor(PassthroughProcessor):
    """None / null processor."""

    data_type = type(None)
