"""Source protocol, registry, and shared helpers."""
from __future__ import annotations

from typing import ClassVar, Iterator, Protocol, runtime_checkable

from sonar_probe.errors import BadSegmentError, BadSpecError
from sonar_probe.types import Segment

_REGISTRY: dict[str, type] = {}


@runtime_checkable
class SegmentSource(Protocol):
    name: ClassVar[str]

    def __iter__(self) -> Iterator[Segment]: ...
    def __len__(self) -> int: ...


def register_source(cls):
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(f"{cls} must declare a 'name' class attribute")
    if name in _REGISTRY:
        raise RuntimeError(f"source {name!r} already registered")
    _REGISTRY[name] = cls
    return cls


def get_source(name: str, **kwargs) -> SegmentSource:
    if name not in _REGISTRY:
        raise KeyError(f"unknown source {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


_CHANNEL_LABELS = {"A": 0, "B": 1, "C": 2, "D": 3}


def coerce_channel(raw: str) -> int:
    s = raw.strip()
    if s in _CHANNEL_LABELS:
        return _CHANNEL_LABELS[s]
    try:
        return int(s)
    except ValueError as e:
        raise BadSpecError(f"cannot interpret channel {raw!r}") from e


def validate_segment(seg: Segment) -> None:
    # Pydantic already enforces end>start and channel>=0; this is a seam for
    # future cross-field checks (e.g., max_seg_seconds, if we want them here).
    if seg.audio_ref is None:
        raise BadSegmentError(f"segment {seg.segment_id} has no audio_ref")
