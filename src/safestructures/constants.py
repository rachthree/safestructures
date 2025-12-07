"""Constants."""
from enum import auto, Enum

ID_FIELD = "id"
TYPE_FIELD = "data_type"
VALUE_FIELD = "value"
KEYS_FIELD = "keys"
CHILDREN_FIELD = "_safestructures_children_ids"
PARENTS_FIELD = "_safestructures_parent_ids"
NODES_FIELD = "_safestructures_nodes_"
SCHEMA_FIELD = "_safestructures_schema_"
VERSION_FIELD = "_safestructures_schema_version_"
SCHEMA_VERSION = "1.0.0"


class Mode(Enum):
    """Serializer modes."""

    SAVE = auto()
    LOAD = auto()
