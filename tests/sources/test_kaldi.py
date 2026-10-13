from pathlib import Path

import pytest

from sonar_probe.errors import BadSpecError
from sonar_probe.sources import get_source


def test_kaldi_yields_segments(kaldi_mini_dir: Path):
    src = get_source("kaldi", input=kaldi_mini_dir)
    segs = list(src)
    assert [s.segment_id for s in segs] == ["utt1", "utt2"]
    assert all(s.spkid == "spkA" for s in segs)
    assert segs[0].channel == 0   # from reco2file_and_channel "A" → 0
    assert segs[0].start == 0.0 and segs[0].end == 1.0


def test_kaldi_default_channel_zero(tmp_path: Path, wav_mono_16k: Path):
    d = tmp_path / "k"; d.mkdir()
    (d / "wav.scp").write_text(f"rec1 {wav_mono_16k}\n")
    (d / "segments").write_text("u1 rec1 0.0 1.0\n")
    (d / "utt2spk").write_text("u1 spkA\n")
    (seg,) = list(get_source("kaldi", input=d))
    assert seg.channel == 0


def test_kaldi_utt2lang_override(tmp_path: Path, wav_mono_16k: Path):
    d = tmp_path / "k"; d.mkdir()
    (d / "wav.scp").write_text(f"rec1 {wav_mono_16k}\n")
    (d / "segments").write_text("u1 rec1 0.0 1.0\n")
    (d / "utt2spk").write_text("u1 spkA\n")
    (d / "utt2lang").write_text("u1 fra\n")
    (seg,) = list(get_source("kaldi", input=d))
    assert seg.lang == "fra"


def test_kaldi_missing_file(tmp_path: Path):
    d = tmp_path / "k"; d.mkdir()
    (d / "wav.scp").write_text("rec1 /tmp/x.wav\n")
    # No segments file
    with pytest.raises(BadSpecError, match="segments"):
        list(get_source("kaldi", input=d))
