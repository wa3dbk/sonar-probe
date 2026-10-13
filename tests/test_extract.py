from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sonar_probe import ExtractConfig, extract
from sonar_probe.engine.speech import SpeechEncoder, SpeechEncoderPool
from sonar_probe.errors import MissingLangError
from sonar_probe.sinks import get_sink, load_store
from sonar_probe.sources import get_source


class FakeEncoder(SpeechEncoder):
    def __init__(self, lang):
        self.lang = lang
    def encode(self, waves):
        out = []
        for w in waves:
            seed = int.from_bytes(hashlib.sha256(w.tobytes()).digest()[:8], "little")
            out.append(np.random.default_rng(seed).standard_normal(1024).astype(np.float32))
        return np.stack(out)


def _pool():
    return SpeechEncoderPool(device="cpu", encoder_factory=lambda lg, dev: FakeEncoder(lg))


def test_extract_plain_text_to_kaldi(plain_text_manifest, tmp_path: Path):
    manifest, scp = plain_text_manifest
    out = tmp_path / "store"
    src = get_source("plain_text", input=manifest, wav_scp=scp)
    sink = get_sink("kaldi_ark", output=out)
    cfg = ExtractConfig(device="cpu", speech_lang="eng")
    res = extract(src, sink, cfg, pool=_pool())
    assert res.written == 2 and res.skipped == 0
    store = load_store(out)
    assert len(store) == 2
    rec = next(iter(store))
    assert rec.modality == "speech"
    assert rec.embedding.shape == (1024,)


def test_extract_requires_lang_somewhere(plain_text_manifest, tmp_path: Path):
    manifest, scp = plain_text_manifest
    src = get_source("plain_text", input=manifest, wav_scp=scp)
    sink = get_sink("kaldi_ark", output=tmp_path / "s")
    with pytest.raises(MissingLangError):
        extract(src, sink, ExtractConfig(device="cpu"), pool=_pool())


def test_extract_skip_logs_failures(tmp_path: Path, wav_mono_16k: Path):
    # Manifest with one bad timestamp (out of bounds)
    scp = tmp_path / "wav.scp"; scp.write_text(f"rec1 {wav_mono_16k}\n")
    m = tmp_path / "m.tsv"
    m.write_text(
        "file_id\tchannel\tspkid\tstart\tend\n"
        "rec1\t0\ts\t0.0\t1.0\n"
        "rec1\t0\ts\t0.0\t10.0\n"   # too long
    )
    out = tmp_path / "store"
    src = get_source("plain_text", input=m, wav_scp=scp)
    sink = get_sink("kaldi_ark", output=out)
    res = extract(src, sink, ExtractConfig(device="cpu", speech_lang="eng"), pool=_pool())
    assert res.written == 1 and res.skipped == 1
    assert res.failures_path is not None and res.failures_path.exists()
    fails = [json.loads(l) for l in res.failures_path.read_text().splitlines() if l.strip()]
    assert len(fails) == 1 and "bounds" in fails[0]["message"]


def test_extract_strict_aborts(tmp_path: Path, wav_mono_16k: Path):
    scp = tmp_path / "wav.scp"; scp.write_text(f"rec1 {wav_mono_16k}\n")
    m = tmp_path / "m.tsv"
    m.write_text(
        "file_id\tchannel\tspkid\tstart\tend\n"
        "rec1\t0\ts\t0.0\t10.0\n"
    )
    src = get_source("plain_text", input=m, wav_scp=scp)
    sink = get_sink("kaldi_ark", output=tmp_path / "s")
    from sonar_probe.errors import BadSegmentError
    with pytest.raises(BadSegmentError):
        extract(src, sink, ExtractConfig(device="cpu", speech_lang="eng", strict=True),
                pool=_pool())
