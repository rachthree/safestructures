"""Defaults."""

from safestructures.processors.base import DataProcessor
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


def register_processor(mapping: dict, processor: DataProcessor) -> dict:
    """Register a processor for process mapping."""
    mapping[processor.data_type] = processor
    return mapping


BASIC_PROCESS_MAP = {}
for processor in basic_cls_list:
    register_processor(BASIC_PROCESS_MAP, processor)

iterable_cls_list = [
    ListProcessor,
    SetProcessor,
    TupleProcessor,
    DictProcessor,
    DataclassProcessor,
]

ITERABLE_PROCESS_MAP = {}
for processor in iterable_cls_list:
    register_processor(ITERABLE_PROCESS_MAP, processor)

DEFAULT_PROCESS_MAP = {**BASIC_PROCESS_MAP, **ITERABLE_PROCESS_MAP}

register_processor(DEFAULT_PROCESS_MAP, NumpyProcessor)


if is_available("torch"):
    from safestructures.processors.tensor import TorchProcessor

    register_processor(DEFAULT_PROCESS_MAP, TorchProcessor)

if is_available("tensorflow"):
    from safestructures.processors.tensor import TFProcessor

    register_processor(DEFAULT_PROCESS_MAP, TFProcessor)

if is_available("jax"):
    from safestructures.processors.tensor import JaxProcessor

    register_processor(DEFAULT_PROCESS_MAP, JaxProcessor)
