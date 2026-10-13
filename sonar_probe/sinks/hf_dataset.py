"""HuggingFace datasets sink/store."""
from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar, Iterator

import numpy as np
from datasets import Dataset, concatenate_datasets, load_from_disk

from sonar_probe.errors import StoreError
from sonar_probe.sinks.base import (
    EmbeddingStore,
    register_sink,
    register_store,
)
from sonar_probe.types import EMBEDDING_DIM, EmbeddingRecord, SegmentID


_COLUMNS = (
    "segment_id", "file_id", "channel", "spkid",
    "start", "end", "lang", "modality", "embedding",
)


def _records_to_dict(buf: list[EmbeddingRecord]) -> dict[str, list]:
    return {
        "segment_id": [r.segment_id for r in buf],
        "file_id":    [r.file_id for r in buf],
        "channel":    [int(r.channel) for r in buf],
        "spkid":      [r.spkid for r in buf],
        "start":      [float(r.start) for r in buf],
        "end":        [float(r.end) for r in buf],
        "lang":       [r.lang for r in buf],
        "modality":   [r.modality for r in buf],
        "embedding":  [r.embedding.tolist() for r in buf],
    }


@register_sink
class HfDatasetSink:
    name: ClassVar[str] = "hf_dataset"

    def __init__(self, output: Path, append: bool = False, buffer_size: int = 256):
        self._out = Path(output)
        self._append = append
        self._buf_size = buffer_size
        self._buf: list[EmbeddingRecord] = []
        self._shards: list[Dataset] = []

    @contextmanager
    def open(self):
        if self._out.exists() and not self._append:
            shutil.rmtree(self._out)
        if self._append and self._out.exists():
            self._shards.append(load_from_disk(str(self._out)))
        self._out.parent.mkdir(parents=True, exist_ok=True)
        try:
            yield
        finally:
            self.close()

    def write(self, rec: EmbeddingRecord) -> None:
        self._buf.append(rec)
        if len(self._buf) >= self._buf_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        ds = Dataset.from_dict(_records_to_dict(self._buf))
        self._shards.append(ds)
        self._buf.clear()

    def close(self) -> None:
        self._flush()
        if not self._shards:
            # Still produce an empty valid dataset
            Dataset.from_dict({c: [] for c in _COLUMNS}).save_to_disk(str(self._out))
            return
        combined = (
            self._shards[0] if len(self._shards) == 1
            else concatenate_datasets(self._shards)
        )
        if self._out.exists():
            shutil.rmtree(self._out)
        combined.save_to_disk(str(self._out))
        self._shards.clear()


@register_store("hf_dataset")
class HfDatasetStore(EmbeddingStore):
    def __init__(self, path: Path):
        self._root = Path(path)
        try:
            self._ds = load_from_disk(str(self._root))
        except Exception as e:
            raise StoreError(f"{self._root}: not an hf_dataset store ({e})") from e
        self._index: dict[str, int] | None = None

    def _ensure_index(self) -> dict[str, int]:
        if self._index is None:
            self._index = {sid: i for i, sid in enumerate(self._ds["segment_id"])}
        return self._index

    def __iter__(self) -> Iterator[EmbeddingRecord]:
        for i in range(len(self._ds)):
            yield self._row_to_record(i)

    def __len__(self) -> int:
        return len(self._ds)

    def ids(self) -> list[SegmentID]:
        return list(self._ds["segment_id"])

    def __getitem__(self, segment_id: SegmentID) -> EmbeddingRecord:
        idx = self._ensure_index().get(segment_id)
        if idx is None:
            raise StoreError(f"missing segment_id {segment_id!r}")
        return self._row_to_record(idx)

    def _row_to_record(self, i: int) -> EmbeddingRecord:
        row = self._ds[i]
        emb = np.asarray(row["embedding"], dtype=np.float32)
        if emb.shape != (EMBEDDING_DIM,):
            raise StoreError(f"row {i}: bad embedding shape {emb.shape}")
        return EmbeddingRecord(
            segment_id=row["segment_id"], file_id=row["file_id"],
            channel=row["channel"], spkid=row["spkid"],
            start=row["start"], end=row["end"],
            lang=row.get("lang"), modality=row["modality"],
            embedding=emb,
        )

    def matrix(
        self, ids: list[SegmentID] | None = None
    ) -> tuple[list[SegmentID], np.ndarray]:
        if ids is None:
            arr = np.asarray(self._ds.with_format("numpy")["embedding"], dtype=np.float32)
            return list(self._ds["segment_id"]), arr
        idx_map = self._ensure_index()
        order = [idx_map[s] for s in ids]
        arr = np.asarray(self._ds.select(order).with_format("numpy")["embedding"],
                         dtype=np.float32)
        return list(ids), arr
