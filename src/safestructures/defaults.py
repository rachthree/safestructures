"""Defaults."""

from safestructures.processors.basic import (
    BoolProcessor,
    ComplexProcessor,
    FloatProcessor,
    IntProcessor,
    NoneProcessor,
    StringProcessor,
)
from safestructures.processors.iterable import (
    DataclassProcessor,
    DictProcessor,
    ListProcessor,
    SetProcessor,
    TupleProcessor,
)
from safestructures.processors.tensor import NumpyProcessor
from safestructures.utils.module import is_available

basic_cls_list = [
    IntProcessor,
    FloatProcessor,
    ComplexProcessor,
    StringProcessor,
    BoolProcessor,
    NoneProcessor,
]

BASIC_PROCESS_MAP = {processor.data_type: processor for processor in basic_cls_list}
iterable_cls_list = [
    ListProcessor,
    SetProcessor,
    TupleProcessor,
    DictProcessor,
    DataclassProcessor,
]
ITERABLE_PROCESS_MAP = {
    processor.data_type: processor for processor in iterable_cls_list
}
DEFAULT_PROCESS_MAP = {**BASIC_PROCESS_MAP, **ITERABLE_PROCESS_MAP}
DEFAULT_PROCESS_MAP[NumpyProcessor.data_type] = NumpyProcessor

if is_available("torch"):
    from safestructures.processors.tensor import TorchProcessor

    DEFAULT_PROCESS_MAP[TorchProcessor.data_type] = TorchProcessor
