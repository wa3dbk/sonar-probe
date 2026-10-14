from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner

from sonar_probe.cli import app
from sonar_probe.engine.speech import SpeechEncoder
from sonar_probe.engine.text import TextBackend


class FakeSpeech(SpeechEncoder):
    def __init__(self, lang):
        self.lang = lang

    def encode(self, waves):
        out = []
        for w in waves:
            seed = int.from_bytes(
                hashlib.sha256(w.tobytes()).digest()[:8], "little"
            )
            out.append(
                np.random.default_rng(seed).standard_normal(1024).astype(np.float32)
            )
        return np.stack(out)


class FakeText(TextBackend):
    def encode(self, texts, lang):
        return np.stack(
            [
                np.random.default_rng(
                    int.from_bytes(
                        hashlib.sha256((t + lang).encode()).digest()[:8], "little"
                    )
                ).standard_normal(1024).astype(np.float32)
                for t in texts
            ]
        )


@pytest.fixture
def fake_encoders():
    with patch(
        "sonar_probe.engine.speech._default_factory",
        lambda lang, device: FakeSpeech(lang),
    ), patch(
        "sonar_probe.engine.text._default_factory", lambda device: FakeText()
    ):
        yield


def test_cli_extract_plain_to_kaldi(plain_text_manifest, tmp_path: Path, fake_encoders):
    manifest, scp = plain_text_manifest
    out = tmp_path / "store"
    runner = CliRunner()
    r = runner.invoke(
        app,
        [
            "extract",
            "--source",
            "plain_text",
            "--input",
            str(manifest),
            "--wav-scp",
            str(scp),
            "--sink",
            "kaldi_ark",
            "--output",
            str(out),
            "--speech-lang",
            "eng",
            "--device",
            "cpu",
        ],
    )
    assert r.exit_code == 0, r.output
    assert (out / "embeddings.scp").exists()
    assert "written=2" in r.output


def test_cli_extract_text(tmp_path: Path, fake_encoders):
    inp = tmp_path / "t.tsv"
    inp.write_text("text_id\ttext\tlang\n" "t1\thi\teng_Latn\n")
    out = tmp_path / "store"
    r = CliRunner().invoke(
        app,
        [
            "extract-text",
            "--input",
            str(inp),
            "--sink",
            "hf_dataset",
            "--output",
            str(out),
            "--device",
            "cpu",
        ],
    )
    assert r.exit_code == 0, r.output


def test_cli_compare_scalar(tmp_path: Path, fake_encoders):
    # Build two stores with the same key via the API, then compare via CLI.
    from sonar_probe.sinks import get_sink
    from sonar_probe.types import EmbeddingRecord

    def _mk(path: Path):
        s = get_sink("kaldi_ark", output=path)
        with s.open():
            s.write(
                EmbeddingRecord(
                    segment_id="x",
                    file_id="r",
                    channel=0,
                    spkid="s",
                    start=0.0,
                    end=1.0,
                    lang="eng",
                    modality="speech",
                    embedding=np.ones(1024, dtype=np.float32),
                )
            )

    a = tmp_path / "a"
    b = tmp_path / "b"
    _mk(a)
    _mk(b)
    r = CliRunner().invoke(
        app,
        [
            "compare",
            "--a",
            f"store:{a}#x",
            "--b",
            f"store:{b}#x",
            "--device",
            "cpu",
        ],
    )
    assert r.exit_code == 0
    assert "cosine=" in r.output


def test_cli_inspect(tmp_path: Path, fake_encoders):
    from sonar_probe.sinks import get_sink
    from sonar_probe.types import EmbeddingRecord

    out = tmp_path / "s"
    sink = get_sink("kaldi_ark", output=out)
    with sink.open():
        sink.write(
            EmbeddingRecord(
                segment_id="x",
                file_id="r",
                channel=0,
                spkid="s",
                start=0.0,
                end=1.0,
                lang="eng",
                modality="speech",
                embedding=np.ones(1024, dtype=np.float32),
            )
        )
    r = CliRunner().invoke(app, ["inspect", str(out)])
    assert r.exit_code == 0
    assert "count=1" in r.output
