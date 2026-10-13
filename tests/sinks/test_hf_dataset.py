from pathlib import Path

import numpy as np

from sonar_probe.sinks import get_sink
from sonar_probe.sinks.hf_dataset import HfDatasetStore
from sonar_probe.types import EmbeddingRecord


def _rec(sid: str) -> EmbeddingRecord:
    rng = np.random.default_rng(hash(sid) & 0xFFFFFFFF)
    return EmbeddingRecord(
        segment_id=sid, file_id="r", channel=0, spkid="s",
        start=0.0, end=1.0, lang="eng", modality="speech",
        embedding=rng.standard_normal(1024).astype(np.float32),
    )


def test_hf_sink_roundtrip(tmp_path: Path):
    out = tmp_path / "hf_store"
    sink = get_sink("hf_dataset", output=out)
    r1, r2 = _rec("a"), _rec("b")
    with sink.open():
        sink.write(r1); sink.write(r2)
    store = HfDatasetStore(out)
    assert len(store) == 2
    got = store["a"]
    assert np.allclose(got.embedding, r1.embedding)
    assert got.modality == "speech"


def test_hf_sink_append(tmp_path: Path):
    out = tmp_path / "hf_store"
    with get_sink("hf_dataset", output=out).open() as _:
        get_sink("hf_dataset", output=out).__enter__ if False else None
    # First write
    with get_sink("hf_dataset", output=out).open():
        pass
    s1 = get_sink("hf_dataset", output=out)
    with s1.open():
        s1.write(_rec("a"))
    s2 = get_sink("hf_dataset", output=out, append=True)
    with s2.open():
        s2.write(_rec("b"))
    store = HfDatasetStore(out)
    assert set(store.ids()) == {"a", "b"}


def test_hf_matrix(tmp_path: Path):
    out = tmp_path / "hf_store"
    sink = get_sink("hf_dataset", output=out)
    with sink.open():
        for i in range(3):
            sink.write(_rec(f"x{i}"))
    ids, M = HfDatasetStore(out).matrix()
    assert M.shape == (3, 1024) and M.dtype == np.float32
