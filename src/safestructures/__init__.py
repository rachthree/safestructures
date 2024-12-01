from importlib import metadata

from safestructures.processors.base import DataProcessor, TensorProcessor
from safestructures.wrapper import load_file, save_file

__version__ = metadata.version("safestructures")

__all__ = ["save_file", "load_file", "DataProcessor", "TensorProcessor"]
