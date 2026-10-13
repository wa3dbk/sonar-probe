"""Sink backends."""
from pathlib import Path

from sonar_probe.errors import StoreError
from sonar_probe.sinks.base import (
    EmbeddingSink,
    EmbeddingStore,
    get_sink,
    register_sink,
    register_store,
)
from sonar_probe.sinks import hf_dataset, kaldi_ark  # noqa: F401


def load_store(path: Path) -> EmbeddingStore:
    p = Path(path)
    if (p / "embeddings.scp").is_file() and (p / "metadata.jsonl").is_file():
        from sonar_probe.sinks.kaldi_ark import KaldiArkStore
        return KaldiArkStore(p)
    if (p / "dataset_info.json").is_file() or (p / "state.json").is_file():
        from sonar_probe.sinks.hf_dataset import HfDatasetStore
        return HfDatasetStore(p)
    raise StoreError(f"{p}: unrecognized store layout")


__all__ = [
    "EmbeddingSink", "EmbeddingStore",
    "get_sink", "register_sink", "register_store",
    "load_store",
]
