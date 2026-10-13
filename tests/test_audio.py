from pathlib import Path

import numpy as np
import pytest

from sonar_probe.audio import load_waveform, parse_wav_scp
from sonar_probe.errors import BadSegmentError
from sonar_probe.types import AudioRef


def test_parse_wav_scp(tmp_path: Path):
    scp = tmp_path / "wav.scp"
    scp.write_text("rec1 /tmp/a.wav\nrec2 sox /tmp/b.flac -t wav - |\n")
    m = parse_wav_scp(scp)
    assert m == {"rec1": "/tmp/a.wav", "rec2": "sox /tmp/b.flac -t wav - |"}


def test_load_waveform_mono_16k_passthrough(wav_mono_16k: Path):
    w = load_waveform(AudioRef(path=wav_mono_16k), channel=0, start=0.0, end=1.0)
    assert w.dtype == np.float32
    assert w.ndim == 1
    assert w.shape[0] == 16_000  # 1 second @ 16 kHz


def test_load_waveform_stereo_channel_select(wav_stereo_16k: Path):
    left = load_waveform(AudioRef(path=wav_stereo_16k), channel=0, start=0.0, end=1.0)
    right = load_waveform(AudioRef(path=wav_stereo_16k), channel=1, start=0.0, end=1.0)
    # Left is 440 Hz, right 880 Hz: different signals.
    assert not np.allclose(left, right)
    assert left.shape == (16_000,)
    assert right.shape == (16_000,)


def test_load_waveform_resamples_to_16k(wav_stereo_44k: Path):
    w = load_waveform(AudioRef(path=wav_stereo_44k), channel=0, start=0.0, end=1.0)
    assert w.shape == (16_000,)


def test_load_waveform_from_bytes(wav_mono_16k: Path):
    data = wav_mono_16k.read_bytes()
    w = load_waveform(AudioRef(bytes_=data, sr=16_000), channel=0, start=0.0, end=0.5)
    assert w.shape == (8_000,)


def test_load_waveform_bad_channel(wav_mono_16k: Path):
    with pytest.raises(BadSegmentError, match="channel"):
        load_waveform(AudioRef(path=wav_mono_16k), channel=1, start=0.0, end=1.0)


def test_load_waveform_out_of_bounds(wav_mono_16k: Path):
    with pytest.raises(BadSegmentError, match="bounds"):
        load_waveform(AudioRef(path=wav_mono_16k), channel=0, start=0.0, end=5.0)
