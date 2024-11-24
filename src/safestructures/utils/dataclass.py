"""Dataclass utilities."""
import dataclasses
from typing import Protocol, runtime_checkable


@runtime_checkable
@dataclasses.dataclass
class Dataclass(Protocol):
    """Protocol to help provide a 'dataclass' type."""

    pass
