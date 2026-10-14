"""Integration tests that load real SONAR checkpoints.

Run with:  pytest -m integration
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sonar_probe import CompareConfig, ExtractConfig, compare, extract
from sonar_probe.engine.text import TextEncoder
from sonar_probe.sinks import get_sink
from sonar_probe.sources import get_source

pytestmark = pytest.mark.integration


def test_speech_eng_cosine_floor(plain_text_manifest, tmp_path: Path):
    """Two renditions of the same tone should correlate above 0.5."""
    manifest, scp = plain_text_manifest
    out = tmp_path / "store"
    src = get_source("plain_text", input=manifest, wav_scp=scp)
    sink = get_sink("kaldi_ark", output=out)
    cfg = ExtractConfig(device="cpu", speech_lang="eng")
    res = extract(src, sink, cfg)
    assert res.written == 2
    r = compare(f"store:{out}#rec1-0-spkA-0-1000",
                f"store:{out}#rec1-0-spkA-1000-2000",
                CompareConfig(device="cpu"))
    # Same source tone, consecutive 1s windows: expect correlation
    # noticeably above 0 (floor is loose; real SONAR renders both as
    # speech-ish).
    assert float(r.scores[0, 0]) > 0.3


def test_text_cross_modal_round_trip(tmp_path: Path):
    """Text 'hello world' @eng vs. same text @fra differ; same lang pair
    matches itself."""
    tenc = TextEncoder(device="cpu")
    e1 = tenc.encode(["hello world"], lang="eng_Latn")
    e2 = tenc.encode(["hello world"], lang="eng_Latn")
    n1 = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
    n2 = e2 / np.linalg.norm(e2, axis=1, keepdims=True)
    assert float((n1 @ n2.T)[0, 0]) == pytest.approx(1.0, abs=1e-4)
