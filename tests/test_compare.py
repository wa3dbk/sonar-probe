import hashlib
from pathlib import Path

import numpy as np
import pytest

from sonar_probe import CompareConfig, compare, load_store
from sonar_probe.compare import OperandRef, parse_operand
from sonar_probe.engine.speech import SpeechEncoderPool
from sonar_probe.engine.text import TextBackend, TextEncoder
from sonar_probe.sinks import get_sink
from sonar_probe.types import EmbeddingRecord


def _vec(seed_text: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "little")
    v = np.random.default_rng(seed).standard_normal(1024).astype(np.float32)
    return v


def _store(tmp_path: Path, name: str, ids: list[str]) -> Path:
    out = tmp_path / name
    sink = get_sink("kaldi_ark", output=out)
    with sink.open():
        for sid in ids:
            sink.write(EmbeddingRecord(
                segment_id=sid, file_id="r", channel=0, spkid="s",
                start=0.0, end=1.0, lang="eng", modality="speech",
                embedding=_vec(sid),
            ))
    return out


def test_parse_operand_inline():
    r = parse_operand("rec1,0,spkA,0.0,1.0")
    assert r.kind == "inline_segment"


def test_parse_operand_store_id():
    r = parse_operand("store:/tmp/x#utt1")
    assert r.kind == "store_id"
    assert r.payload == ("/tmp/x", "utt1")


def test_parse_operand_text():
    r = parse_operand("text:hello world@eng_Latn")
    assert r.kind == "text" and r.payload == ("hello world", "eng_Latn")


def test_parse_operand_text_file():
    r = parse_operand("text-file:/tmp/t.txt@fra_Latn")
    assert r.kind == "text_file" and r.payload == ("/tmp/t.txt", "fra_Latn")


def test_parse_operand_path(tmp_path: Path):
    r = parse_operand(str(tmp_path))
    assert r.kind == "path"


def test_compare_store_vs_store_scalar(tmp_path: Path):
    a = _store(tmp_path, "a", ["x"])
    b = _store(tmp_path, "b", ["x"])
    res = compare(f"store:{a}#x", f"store:{b}#x", CompareConfig(device="cpu"))
    assert res.scores.shape == (1, 1)
    # Both stores wrote identical vectors (same seed) → cosine ≈ 1
    assert res.scores[0, 0] == pytest.approx(1.0, abs=1e-5)


def test_compare_store_vs_store_matrix(tmp_path: Path):
    a = _store(tmp_path, "a", ["x", "y"])
    b = _store(tmp_path, "b", ["x", "z"])
    res = compare(str(a), str(b), CompareConfig(device="cpu"))
    assert res.scores.shape == (2, 2)
    # (x,x) ≈ 1
    i = res.a_ids.index("x"); j = res.b_ids.index("x")
    assert res.scores[i, j] == pytest.approx(1.0, abs=1e-5)


def test_compare_cross_modal_shape(tmp_path: Path):
    a = _store(tmp_path, "a", ["x"])
    class _T(TextBackend):
        def encode(self, texts, lang):
            return np.stack([_vec(t) for t in texts])
    tenc = TextEncoder(device="cpu", encoder_factory=lambda d: _T())
    res = compare(f"store:{a}#x", "text:hello@eng_Latn",
                  CompareConfig(device="cpu"), text_encoder=tenc)
    assert res.scores.shape == (1, 1)
    assert res.modality_a == "speech" and res.modality_b == "text"


def test_compare_topk_filter(tmp_path: Path):
    a = _store(tmp_path, "a", ["x", "y"])
    b = _store(tmp_path, "b", ["x", "y", "z"])
    res = compare(str(a), str(b),
                  CompareConfig(device="cpu", top_k=1))
    # top_k filter is applied via helper on the result; verify shape still (2,3)
    # and top_k helper yields 2 rows.
    pairs = list(res.iter_pairs(top_k=1))
    assert len(pairs) == 2  # one top per a_id
