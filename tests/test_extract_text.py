from pathlib import Path

import hashlib
import numpy as np

from sonar_probe import ExtractConfig, extract_text, load_store
from sonar_probe.engine.text import TextBackend, TextEncoder
from sonar_probe.sinks import get_sink
from sonar_probe.text_input import read_text_tsv


class FakeText(TextBackend):
    def encode(self, texts, lang):
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
