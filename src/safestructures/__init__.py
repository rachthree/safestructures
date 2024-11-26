from importlib import metadata

from safestructures.processors.base import DataProcessor, TensorProcessor
from safestructures.serializer import Serializer

__version__ = metadata.version("safestructures")

__all__ = ["Serializer", "DataProcessor", "TensorProcessor"]
