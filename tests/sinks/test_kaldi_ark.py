from pathlib import Path

import numpy as np
import pytest

from sonar_probe.errors import StoreError
from sonar_probe.sinks import get_sink
from sonar_probe.sinks.kaldi_ark import KaldiArkStore
from sonar_probe.types import EmbeddingRecord


def _rec(seg_id: str, modality: str = "speech") -> EmbeddingRecord:
    rng = np.random.default_rng(hash(seg_id) & 0xFFFFFFFF)
    return EmbeddingRecord(
        segment_id=seg_id, file_id="rec1", channel=0, spkid="spkA",
        start=0.0, end=1.0, lang="eng", modality=modality,
        embedding=rng.standard_normal(1024).astype(np.float32),
    )


def test_write_then_read_roundtrip(tmp_path: Path):
    out = tmp_path / "store"
    sink = get_sink("kaldi_ark", output=out)
    r1, r2 = _rec("a"), _rec("b")
    with sink.open():
        sink.write(r1)
        sink.write(r2)
    store = KaldiArkStore(out)
    assert len(store) == 2
    assert set(store.ids()) == {"a", "b"}
    got = store["a"]
    assert got.file_id == "rec1" and got.lang == "eng"
    assert np.allclose(got.embedding, r1.embedding)


def test_matrix_shape(tmp_path: Path):
    out = tmp_path / "store"
    sink = get_sink("kaldi_ark", output=out)
    with sink.open():
        for i in range(3):
            sink.write(_rec(f"s{i}"))
    ids, M = KaldiArkStore(out).matrix()
    assert M.shape == (3, 1024) and M.dtype == np.float32
    assert ids == ["s0", "s1", "s2"]


def test_text_modality_roundtrip(tmp_path: Path):
    out = tmp_path / "store"
    sink = get_sink("kaldi_ark", output=out)
    with sink.open():
        sink.write(_rec("t1", modality="text"))
    rec = KaldiArkStore(out)["t1"]
    assert rec.modality == "text"


def test_missing_id_raises(tmp_path: Path):
    out = tmp_path / "store"
    sink = get_sink("kaldi_ark", output=out)
    with sink.open():
        sink.write(_rec("only"))
    with pytest.raises(StoreError, match="missing"):
        _ = KaldiArkStore(out)["ghost"]
