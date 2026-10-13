"""Sink backends."""
from sonar_probe.sinks.base import (
    EmbeddingSink,
    EmbeddingStore,
    get_sink,
    register_sink,
    register_store,
)
from sonar_probe.sinks import hf_dataset, kaldi_ark  # noqa: F401

__all__ = [
    "EmbeddingSink", "EmbeddingStore",
    "get_sink", "register_sink", "register_store",
]
