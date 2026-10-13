"""Sink / Store protocols and registry."""
from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import ClassVar, Iterator, Protocol, runtime_checkable

import numpy as np

from sonar_probe.types import EmbeddingRecord, SegmentID

_SINKS: dict[str, type] = {}
_STORES: dict[str, type] = {}


@runtime_checkable
class EmbeddingSink(Protocol):
    name: ClassVar[str]

    def open(self) -> AbstractContextManager[None]: ...
    def write(self, rec: EmbeddingRecord) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class EmbeddingStore(Protocol):
    def __iter__(self) -> Iterator[EmbeddingRecord]: ...
    def __getitem__(self, segment_id: SegmentID) -> EmbeddingRecord: ...
    def __len__(self) -> int: ...
    def ids(self) -> list[SegmentID]: ...
    def matrix(
        self, ids: list[SegmentID] | None = None
    ) -> tuple[list[SegmentID], np.ndarray]: ...


def register_sink(cls):
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(f"{cls} must declare 'name'")
    _SINKS[name] = cls
    return cls


def register_store(name: str):
    def deco(cls):
        _STORES[name] = cls
        return cls
    return deco


def get_sink(name: str, **kwargs) -> EmbeddingSink:
    if name not in _SINKS:
        raise KeyError(f"unknown sink {name!r}; known: {sorted(_SINKS)}")
    return _SINKS[name](**kwargs)


def _store_class_for(name: str) -> type:
    if name not in _STORES:
        raise KeyError(f"no store registered for sink {name!r}")
    return _STORES[name]
