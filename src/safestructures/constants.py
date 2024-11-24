"""Constants."""
from enum import auto, Enum

TYPE_FIELD = "data_type"
VALUE_FIELD = "value"
KEYS_FIELD = "keys"
SCHEMA_FIELD = "_safestructures_schema_"
VERSION_FIELD = "_safestructures_schema_version_"
SCHEMA_VERSION = "1.0.0"
DATACLASS_NAME = "dataclass"


class Mode(Enum):
    """Serializer modes."""

    SAVE = auto()
    LOAD = auto()
