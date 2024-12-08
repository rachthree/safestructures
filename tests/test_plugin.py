"""Test plugin support."""
from dataclasses import make_dataclass

from transformers.modeling_outputs import ModelOutput

from safestructures.processors.iterable import DataclassProcessor
from safestructures.utils.dataclass import Dataclass


class ModelOutputProcessor(DataclassProcessor):
    """Processor for transformer's ModelOutput class."""

    data_type = ModelOutput

    def deserialize(self, serialized: dict, **kwargs) -> Dataclass:
        """Overload `DataProcessor.deserialize`."""
        fields = list(serialized.keys())
        cls = make_dataclass(self.data_type.__name__, fields)
        dc_kwargs = {}
        for k, v in serialized.items():
            dc_kwargs[k] = self.serializer.deserialize(v)

        return cls(**dc_kwargs)
