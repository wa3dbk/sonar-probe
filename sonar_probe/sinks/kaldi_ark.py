"""Kaldi ark+scp sink and store, with JSONL metadata sidecar."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar, Iterator

import kaldiio
import numpy as np

from sonar_probe.errors import StoreError
from sonar_probe.sinks.base import (
    EmbeddingStore,
    register_sink,
    register_store,
)
from sonar_probe.types import EmbeddingRecord, SegmentID


@register_sink
class KaldiArkSink:
    name: ClassVar[str] = "kaldi_ark"

    def __init__(self, output: Path, append: bool = False):
        self._out = Path(output)
        self._append = append
        self._writer: kaldiio.WriteHelper | None = None
        self._meta_fh = None

    @contextmanager
    def open(self):
        self._out.mkdir(parents=True, exist_ok=True)
        ark = self._out / "embeddings.ark"
        scp = self._out / "embeddings.scp"
        meta = self._out / "metadata.jsonl"
        mode = "a" if self._append else "w"
        if not self._append:
            # overwrite semantics: truncate ark/scp/meta
            for p in (ark, scp, meta):
                if p.exists():
                    p.unlink()
        spec = f"ark,scp:{ark},{scp}"
        self._writer = kaldiio.WriteHelper(spec, write_function="numpy")
        self._meta_fh = meta.open(mode, encoding="utf-8")
        try:
            yield
        finally:
            self.close()

    def write(self, rec: EmbeddingRecord) -> None:
        if self._writer is None or self._meta_fh is None:
            raise StoreError("sink is not open")
        try:
            self._writer(rec.segment_id, rec.embedding)
            self._meta_fh.write(
                json.dumps({
                    "segment_id": rec.segment_id,
                    "file_id": rec.file_id,
                    "channel": rec.channel,
                    "spkid": rec.spkid,
                    "start": rec.start,
                    "end": rec.end,
                    "lang": rec.lang,
                    "modality": rec.modality,
                }) + "\n"
            )
        except Exception as e:
            raise StoreError(f"write failed for {rec.segment_id}: {e}") from e

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._meta_fh is not None:
            self._meta_fh.close()
            self._meta_fh = None


@register_store("kaldi_ark")
class KaldiArkStore(EmbeddingStore):
    def __init__(self, path: Path):
        self._root = Path(path)
        scp = self._root / "embeddings.scp"
        meta = self._root / "metadata.jsonl"
        if not scp.is_file() or not meta.is_file():
            raise StoreError(f"{self._root}: not a kaldi_ark store")
        self._vectors: dict[str, np.ndarray] = dict(kaldiio.load_scp(str(scp)))
        self._meta: dict[str, dict] = {}
        for raw in meta.read_text().splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            self._meta[row["segment_id"]] = row
        self._order = list(self._meta.keys())

    def __iter__(self) -> Iterator[EmbeddingRecord]:
        for sid in self._order:
            yield self[sid]

    def __len__(self) -> int:
        return len(self._order)

    def ids(self) -> list[str]:
        return list(self._order)

    def __getitem__(self, segment_id: SegmentID) -> EmbeddingRecord:
        if segment_id not in self._meta:
            raise StoreError(f"missing segment_id {segment_id!r}")
        m = self._meta[segment_id]
        vec = self._vectors[segment_id].astype(np.float32, copy=False)
        return EmbeddingRecord(
            segment_id=m["segment_id"],
            file_id=m["file_id"],
            channel=m["channel"],
            spkid=m["spkid"],
            start=m["start"],
            end=m["end"],
            lang=m.get("lang"),
            modality=m["modality"],
            embedding=vec,
        )

    def matrix(
        self, ids: list[SegmentID] | None = None
    ) -> tuple[list[SegmentID], np.ndarray]:
        keys = ids if ids is not None else self._order
        M = np.stack([self._vectors[k].astype(np.float32, copy=False) for k in keys])
        return list(keys), M
