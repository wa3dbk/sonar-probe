from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def _sine(freq: float, duration: float, sr: int) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)


@pytest.fixture
def wav_mono_16k(tmp_path: Path) -> Path:
    """2-second 16 kHz mono WAV, single tone."""
    path = tmp_path / "mono_16k.wav"
    sf.write(path, _sine(440.0, 2.0, 16_000), 16_000, subtype="FLOAT")
    return path


@pytest.fixture
def wav_stereo_44k(tmp_path: Path) -> Path:
    """2-second 44.1 kHz stereo WAV; channel 0 is 440 Hz, channel 1 is 880 Hz."""
    path = tmp_path / "stereo_44k.wav"
    left = _sine(440.0, 2.0, 44_100)
    right = _sine(880.0, 2.0, 44_100)
    stereo = np.stack([left, right], axis=1)
    sf.write(path, stereo, 44_100, subtype="FLOAT")
    return path


@pytest.fixture
def wav_stereo_16k(tmp_path: Path) -> Path:
    """2-second 16 kHz stereo WAV; channel 0 is 440 Hz, channel 1 is 880 Hz."""
    path = tmp_path / "stereo_16k.wav"
    left = _sine(440.0, 2.0, 16_000)
    right = _sine(880.0, 2.0, 16_000)
    stereo = np.stack([left, right], axis=1)
    sf.write(path, stereo, 16_000, subtype="FLOAT")
    return path


@pytest.fixture
def kaldi_mini_dir(tmp_path: Path, wav_stereo_16k: Path) -> Path:
    """Minimal Kaldi data dir with 2 utts over the stereo file."""
    d = tmp_path / "kaldi_data"
    d.mkdir()
    (d / "wav.scp").write_text(f"rec1 {wav_stereo_16k}\n")
    (d / "segments").write_text(
        "utt1 rec1 0.0 1.0\n"
        "utt2 rec1 1.0 2.0\n"
    )
    (d / "utt2spk").write_text("utt1 spkA\nutt2 spkA\n")
    (d / "reco2file_and_channel").write_text("rec1 rec1 A\n")  # channel A → 0
    return d


@pytest.fixture
def plain_text_manifest(tmp_path: Path, wav_mono_16k: Path) -> tuple[Path, Path]:
    """Returns (manifest_tsv, wav_scp)."""
    scp = tmp_path / "wav.scp"
    scp.write_text(f"rec1 {wav_mono_16k}\n")
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "file_id\tchannel\tspkid\tstart\tend\n"
        "rec1\t0\tspkA\t0.0\t1.0\n"
        "rec1\t0\tspkA\t1.0\t2.0\n"
    )
    return manifest, scp


@pytest.fixture
def hf_mini_dataset(tmp_path: Path, wav_mono_16k: Path):
    """Minimal HF dataset saved to disk; metadata-only, uses --wav-scp fallback."""
    from datasets import Dataset
    scp = tmp_path / "wav.scp"
    scp.write_text(f"rec1 {wav_mono_16k}\n")
    ds = Dataset.from_dict({
        "file_id": ["rec1", "rec1"],
        "channel": [0, 0],
        "spkid":   ["spkA", "spkA"],
        "start":   [0.0, 1.0],
        "end":     [1.0, 2.0],
    })
    out = tmp_path / "hf_data"
    ds.save_to_disk(str(out))
    return out, scp
