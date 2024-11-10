"""Constants and defaults."""
from safestructures.processors.basic import BASIC_TYPES, BasicProcessor
from safestructures.processors.iterable import (
    DataclassProcessor,
    DictProcessor,
    ListProcessor,
    SetProcessor,
    TupleProcessor,
)

TYPE_FIELD = "type"
VALUE_FIELD = "value"
SCHEMA_FIELD = "_safestructures_schema_"
VERSION_FIELD = "_safestructures_schema_version_"
SCHEMA_VERSION = "1.0.0"

BASIC_PROCESS_MAP = {t: BasicProcessor for t in BASIC_TYPES}
ITERABLE_PROCESS_MAP = {
    ListProcessor.data_type: ListProcessor,
    SetProcessor.data_type: SetProcessor,
    TupleProcessor.data_type: TupleProcessor,
    DictProcessor.data_type: DictProcessor,
    DataclassProcessor.data_type: DataclassProcessor,
}
DEFAULT_PROCESS_MAP = {**BASIC_PROCESS_MAP, **ITERABLE_PROCESS_MAP}
