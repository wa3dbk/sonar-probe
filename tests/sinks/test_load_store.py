from pathlib import Path

import numpy as np

from sonar_probe.sinks import get_sink, load_store
from sonar_probe.types import EmbeddingRecord


def _r(sid: str) -> EmbeddingRecord:
    return EmbeddingRecord(
        segment_id=sid, file_id="r", channel=0, spkid="s",
        start=0.0, end=1.0, lang="eng", modality="speech",
        embedding=np.zeros(1024, dtype=np.float32),
    )


def test_load_store_detects_kaldi(tmp_path: Path):
    out = tmp_path / "k"
    sink = get_sink("kaldi_ark", output=out)
    with sink.open():
        sink.write(_r("a"))
    store = load_store(out)
    assert store.ids() == ["a"]


def test_load_store_detects_hf(tmp_path: Path):
    out = tmp_path / "h"
    sink = get_sink("hf_dataset", output=out)
    with sink.open():
        sink.write(_r("a"))
    store = load_store(out)
    assert store.ids() == ["a"]
