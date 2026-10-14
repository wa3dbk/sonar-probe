"""sonar-probe: extract and compare SONAR speech/text embeddings."""

__version__ = "0.1.0"

from sonar_probe.config import CompareConfig, ExtractConfig
from sonar_probe.driver import ExtractResult, extract, extract_text
from sonar_probe.sinks import load_store
from sonar_probe.text_input import read_text_tsv
from sonar_probe.types import (
    EMBEDDING_DIM,
    AudioRef,
    EmbeddingRecord,
    Segment,
    TextItem,
)

__all__ = [
    "__version__",
    "extract", "extract_text", "load_store", "read_text_tsv",
    "Segment", "EmbeddingRecord", "AudioRef", "TextItem",
    "ExtractConfig", "CompareConfig", "ExtractResult",
    "EMBEDDING_DIM",
]
