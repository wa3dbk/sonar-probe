"""Core data model for sonar-probe."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EMBEDDING_DIM = 1024

SegmentID = str
Lang = str

Modality = Literal["speech", "text"]


class AudioRef(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path | None = None
    bytes_: bytes | None = None
    sr: int | None = None

    @model_validator(mode="after")
    def _check_one_source(self) -> "AudioRef":
        if self.path is None and self.bytes_ is None:
            raise ValueError("AudioRef requires either path or bytes_")
        return self


class Segment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    file_id: str
    channel: int = Field(ge=0)
    spkid: str
    start: float
    end: float
    lang: Lang | None = None
    audio_ref: AudioRef | None = None
    # Optional override of the composite ID (used by the kaldi source to keep utt_id).
    _id_override: str | None = None

    @model_validator(mode="after")
    def _check_times(self) -> "Segment":
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be > start ({self.start})")
        return self

    @property
    def segment_id(self) -> SegmentID:
        if self._id_override is not None:
            return self._id_override
        start_ms = round(self.start * 1000)
        end_ms = round(self.end * 1000)
        return f"{self.file_id}-{self.channel}-{self.spkid}-{start_ms}-{end_ms}"

    @property
    def duration(self) -> float:
        return self.end - self.start

    def with_id(self, override: str) -> "Segment":
        s = self.model_copy()
        s._id_override = override
        return s


class EmbeddingRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    segment_id: SegmentID
    file_id: str
    channel: int
    spkid: str
    start: float
    end: float
    lang: Lang | None
    modality: Modality
    embedding: Any  # np.ndarray; validated below

    @field_validator("embedding")
    @classmethod
    def _check_embedding(cls, v: Any) -> np.ndarray:
        if not isinstance(v, np.ndarray):
            raise ValueError(f"embedding must be np.ndarray, got {type(v)}")
        if v.shape != (EMBEDDING_DIM,):
            raise ValueError(f"embedding must have shape ({EMBEDDING_DIM},), got {v.shape}")
        if v.dtype != np.float32:
            v = v.astype(np.float32)
        return v


class TextItem(BaseModel):
    text_id: str
    text: str
    lang: Lang
