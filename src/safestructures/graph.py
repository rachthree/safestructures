"""Graph-based serialization utilities."""

import json
from collections import deque
from enum import Enum
from typing import Any

from safestructures.constants import (
    CHILDREN_FIELD,
    ID_FIELD,
    PARENTS_FIELD,
    TYPE_FIELD,
    VALUE_FIELD,
)

SCHEMA_FIELDS = [ID_FIELD, TYPE_FIELD, VALUE_FIELD, CHILDREN_FIELD, PARENTS_FIELD]
SCHEMA_FIELDS_SET = set(SCHEMA_FIELDS)

KWARGS_MAP = {
    ID_FIELD: "node_id",
    TYPE_FIELD: "schema_type",
    VALUE_FIELD: "serialized_value",
    CHILDREN_FIELD: "children_ids",
    PARENTS_FIELD: "parents_ids",
}


class DataNode:
    """Node containing serialized data."""

    def __init__(
        self,
        *,
        node_id: int,
        schema_type: str,
        serialized_value: Any,
        schema_ext: dict[str, str] = None,
        children_ids: list[int] = None,
        parents_ids: list[int] = None,
    ):
        """Initialize a DataNode."""
        self.id = node_id
        self.schema_type = schema_type
        self.serialized_value = serialized_value
        self.schema_ext = schema_ext
        self.children = []
        self.parents = []
        self.children_ids = children_ids if children_ids is not None else set()
        self.parents_ids = parents_ids if parents_ids is not None else set()

    @classmethod
    def from_schema(cls, schema: dict):
        """Initialize a DataNode from schema."""
        kwargs = {}
        schema_ext = {}
        for k, v in schema.items():
            if k in SCHEMA_FIELDS_SET:
                kwargs[KWARGS_MAP[k]] = v
            else:
                schema_ext[k] = v
        kwargs["schema_ext"] = schema_ext
        return DataNode(**kwargs)

    def add_child(self, child: "DataNode"):
        """Add a child DataNode to current node."""
        self.children.append(child)
        child.parents.append(self)

    @property
    def is_parent(self) -> bool:
        """Provide whether or not DataNode is a parent."""
        return len(self.children_ids) > 0

    @property
    def is_child(self) -> bool:
        """Provide whether or not DataNode is a child."""
        return len(self.parents_ids) > 0

    @property
    def schema(self):
        """Provide the schema."""
        schema = {
            ID_FIELD: self.id,
            TYPE_FIELD: self.schema_type,
            VALUE_FIELD: self.serialized_value,
            CHILDREN_FIELD: [child.id for child in self.children],
            PARENTS_FIELD: [parent.id for parent in self.parents],
        }
        for k, v in self.schema_ext:
            if k in schema:
                raise KeyError("Key {k} is not allowed.")

            if not isinstance(k, str):
                raise TypeError("Keys of schema extensions must be strings")

            if not isinstance(v, str) and v is not None:
                raise TypeError("Values of schema extensions must be strings or None")
            schema[k] = v

        return schema

    def to_schema_json(self):
        """Convert schema to JSON."""
        schema = self.schema
        try:
            json.dumps(schema)
        except Exception as e:
            raise TypeError("Schema is not JSON serializable") from e

        return schema


class TraversalMode(Enum):
    """Options for traversal modes."""

    BFS = "bfs"
    DFS = "dfs"


class GraphExecutor:
    """Base class for graph-based execution."""

    def __init__(self, *, mode: TraversalMode):
        """Initialize the GraphExecutor."""
        self._node_map = {}
        self._queue = deque()
        self._traverse_fn = self._bfs if mode == TraversalMode.BFS else self._dfs

    def process(self, node: DataNode, **kwargs):
        """Process a node."""
        pass

    def _bfs(self, start_nodes: list[DataNode]):
        """Execute BFS traversal."""
        self._queue.extend(start_nodes)
        visited = set()

        while self._queue:
            node = self._queue.popleft()
            if node.id in visited:
                continue

            self.process(node)
            visited.add(node.id)
            self._node_map[node.id] = node

            for child in node.children:
                if child.id not in visited:
                    self._queue.append(child)

    def _dfs(self, start_nodes: list[DataNode]):
        """Execute DFS traversal."""
        self._queue.extend(start_nodes)
        visited = set()

        while self._queue:
            node = self._queue.popleft()
            if node.id in visited:
                continue

            self.process(node)
            visited.add(node.id)
            self._node_map[node.id] = node

            for child in node.children:
                if child.id not in visited:
                    self._queue.appendleft(child)

    def __call__(self, start_nodes: list[DataNode]):
        """Execute the graph."""
        self._traverse_fn(start_nodes)


class GraphSerializer(GraphExecutor):
    """Graph-based serializer."""

    def __init__(self, *, serializer, **kwargs):
        """Initialize a GraphSerializer."""
        self.serializer = serializer
        super().__init__(**kwargs)

    def process(self, node: DataNode):
        """Override GraphExecutor.process."""
        pass
