from __future__ import annotations

import hashlib

import numpy as np
import pytest

from sonar_probe.engine.speech import SpeechEncoder, SpeechEncoderPool


class FakeEncoder(SpeechEncoder):
    """Deterministic encoder: hashes each waveform's bytes to a 1024-d vector."""

    def __init__(self, lang: str):
        self.lang = lang
        self.calls = 0

    def encode(self, waveforms):
        self.calls += 1
        out = []
        for w in waveforms:
            h = hashlib.sha256(w.tobytes()).digest()
            seed = int.from_bytes(h[:8], "little")
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(1024).astype(np.float32)
            # Mark lang in the first dim so lang routing is observable.
            v[0] = {"eng": 1.0, "fra": 2.0, "spa": 3.0}.get(self.lang, 0.0)
            out.append(v)
        return np.stack(out)


def _factory(created):
    def make(lang, device):
        enc = FakeEncoder(lang)
        created.append(lang)
        return enc
    return make


def test_pool_routes_by_language():
    created: list[str] = []
    pool = SpeechEncoderPool(device="cpu", encoder_factory=_factory(created))
    waves = [np.ones(1600, dtype=np.float32), np.zeros(1600, dtype=np.float32)]
    out = pool.encode(waves, ["eng", "fra"])
    assert out.shape == (2, 1024)
    assert out[0, 0] == pytest.approx(1.0)
    assert out[1, 0] == pytest.approx(2.0)
    assert set(created) == {"eng", "fra"}


def test_pool_preserves_input_order():
    created: list[str] = []
    pool = SpeechEncoderPool(device="cpu", encoder_factory=_factory(created))
    # Interleaved langs: eng, fra, eng, fra
    waves = [np.full(1600, i, dtype=np.float32) for i in range(4)]
    langs = ["eng", "fra", "eng", "fra"]
    out = pool.encode(waves, langs)
    assert out.shape == (4, 1024)
    # Row i's lang marker:
    for i, lg in enumerate(langs):
        expected = {"eng": 1.0, "fra": 2.0}[lg]
        assert out[i, 0] == pytest.approx(expected), f"row {i}"


def test_pool_caches_encoders():
    created: list[str] = []
    pool = SpeechEncoderPool(device="cpu", encoder_factory=_factory(created))
    waves = [np.ones(1600, dtype=np.float32)]
    pool.encode(waves, ["eng"])
    pool.encode(waves, ["eng"])
    assert created == ["eng"]  # eng loaded once


def test_pool_evicts_lru():
    created: list[str] = []
    pool = SpeechEncoderPool(device="cpu", encoder_factory=_factory(created), cache_size=2)
    w = np.ones(1600, dtype=np.float32)
    for lg in ["eng", "fra", "spa"]:
        pool.encode([w], [lg])
    # eng was evicted when spa loaded; re-encoding eng loads again
    pool.encode([w], ["eng"])
    assert created == ["eng", "fra", "spa", "eng"]
