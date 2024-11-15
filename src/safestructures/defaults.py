"""Defaults."""

from safestructures.processors.basic import BASIC_TYPES, BasicProcessor
from safestructures.processors.iterable import (
    DataclassProcessor,
    DictProcessor,
    ListProcessor,
    SetProcessor,
    TupleProcessor,
)
from safestructures.processors.tensor import NumpyProcessor
from safestructures.utils.module import is_available

BASIC_PROCESS_MAP = {str(t): BasicProcessor for t in BASIC_TYPES}
iterable_cls_list = [
    ListProcessor,
    SetProcessor,
    TupleProcessor,
    DictProcessor,
    DataclassProcessor,
]
ITERABLE_PROCESS_MAP = {
    str(processor.data_type): processor for processor in iterable_cls_list
}
DEFAULT_PROCESS_MAP = {**BASIC_PROCESS_MAP, **ITERABLE_PROCESS_MAP}
DEFAULT_PROCESS_MAP[str(NumpyProcessor.data_type)] = NumpyProcessor

if is_available("torch"):
    from safestructures.processors.tensor import TorchProcessor

    DEFAULT_PROCESS_MAP[str(TorchProcessor.data_type)] = TorchProcessor
