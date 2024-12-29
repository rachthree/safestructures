"""Dataclass utilities."""
import dataclasses
from typing import Protocol, runtime_checkable


@runtime_checkable
@dataclasses.dataclass
class SafeStructuresDataclass(Protocol):
    """Protocol to help provide a 'dataclass' type."""

    pass
