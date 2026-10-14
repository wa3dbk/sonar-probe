from pathlib import Path

import hashlib
import numpy as np
import pytest

from sonar_probe import ExtractConfig, extract_text, load_store
from sonar_probe.engine.text import TextBackend, TextEncoder
from sonar_probe.errors import BadSegmentError
from sonar_probe.sinks import get_sink
from sonar_probe.text_input import read_text_tsv
from sonar_probe.types import TextItem


class FakeText(TextBackend):
    def encode(self, texts, lang):
        out = []
        for t in texts:
            h = hashlib.sha256((t + lang).encode()).digest()
            seed = int.from_bytes(h[:8], "little")
            out.append(np.random.default_rng(seed).standard_normal(1024).astype(np.float32))
        return np.stack(out)


class FakeTextRaises(TextBackend):
    """Raises BadSegmentError for the lang 'bad_lang', succeeds otherwise."""

    def encode(self, texts, lang):
        if lang == "bad_lang":
            raise BadSegmentError(f"unsupported lang: {lang}")
        out = []
        for t in texts:
            h = hashlib.sha256((t + lang).encode()).digest()
            seed = int.from_bytes(h[:8], "little")
            out.append(np.random.default_rng(seed).standard_normal(1024).astype(np.float32))
        return np.stack(out)


def test_read_text_tsv_with_lang(tmp_path: Path):
    p = tmp_path / "t.tsv"
    p.write_text("text_id\ttext\tlang\n" "t1\thello\teng_Latn\n" "t2\tbonjour\tfra_Latn\n")
    items = list(read_text_tsv(p))
    assert [i.text_id for i in items] == ["t1", "t2"]
    assert items[0].lang == "eng_Latn"


def test_read_text_tsv_default_lang(tmp_path: Path):
    p = tmp_path / "t.tsv"
    p.write_text("text_id\ttext\n" "t1\thello\n")
    (item,) = list(read_text_tsv(p, default_lang="eng_Latn"))
    assert item.lang == "eng_Latn"


def test_extract_text_to_hf(tmp_path: Path):
    p = tmp_path / "t.tsv"
    p.write_text("text_id\ttext\tlang\n" "t1\thi\teng_Latn\n" "t2\tsalut\tfra_Latn\n")
    out = tmp_path / "store"
    enc = TextEncoder(device="cpu", encoder_factory=lambda d: FakeText())
    sink = get_sink("hf_dataset", output=out)
    res = extract_text(read_text_tsv(p), sink, ExtractConfig(device="cpu"), encoder=enc)
    assert res.written == 2 and res.skipped == 0
    store = load_store(out)
    assert len(store) == 2
    rec = store["t1"]
    assert rec.modality == "text" and rec.lang == "eng_Latn"


def test_extract_text_non_strict_logs_failure(tmp_path: Path):
    """Encode error in non-strict mode → failures.jsonl written, item skipped."""
    items = [
        TextItem(text_id="ok1", text="hello", lang="eng_Latn"),
        TextItem(text_id="bad1", text="bonjour", lang="bad_lang"),
    ]
    out = tmp_path / "store"
    enc = TextEncoder(device="cpu", encoder_factory=lambda d: FakeTextRaises())
    sink = get_sink("hf_dataset", output=out)
    res = extract_text(items, sink, ExtractConfig(device="cpu"), encoder=enc)
    assert res.skipped == 1
    assert res.written == 1
    assert res.failures_path is not None and res.failures_path.exists()
    lines = res.failures_path.read_text().splitlines()
    assert len(lines) == 1
    import json
    entry = json.loads(lines[0])
    assert entry["item_id"] == "bad1"
    assert entry["modality"] == "text"
    assert entry["lang"] == "bad_lang"


def test_extract_text_strict_raises_on_failure(tmp_path: Path):
    """Encode error in strict mode → exception propagates."""
    items = [
        TextItem(text_id="bad1", text="bonjour", lang="bad_lang"),
    ]
    out = tmp_path / "store"
    enc = TextEncoder(device="cpu", encoder_factory=lambda d: FakeTextRaises())
    sink = get_sink("hf_dataset", output=out)
    with pytest.raises(BadSegmentError):
        extract_text(items, sink, ExtractConfig(device="cpu", strict=True), encoder=enc)
