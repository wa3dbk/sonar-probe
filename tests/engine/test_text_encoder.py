import hashlib

import numpy as np

from sonar_probe.engine.text import TextBackend, TextEncoder


class FakeTextBackend(TextBackend):
    def encode(self, texts, lang):
        out = []
        for t in texts:
            h = hashlib.sha256((t + lang).encode()).digest()
            rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
            out.append(rng.standard_normal(1024).astype(np.float32))
        return np.stack(out)


def test_text_encoder_shapes():
    enc = TextEncoder(device="cpu", encoder_factory=lambda dev: FakeTextBackend())
    out = enc.encode(["hello", "world"], lang="eng_Latn")
    assert out.shape == (2, 1024) and out.dtype == np.float32


def test_text_encoder_deterministic_per_lang():
    enc = TextEncoder(device="cpu", encoder_factory=lambda dev: FakeTextBackend())
    a = enc.encode(["x"], lang="eng_Latn")
    b = enc.encode(["x"], lang="fra_Latn")
    assert not np.allclose(a, b)
