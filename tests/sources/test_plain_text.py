from pathlib import Path

import pytest

from sonar_probe.errors import BadSpecError
from sonar_probe.sources import get_source


def test_plain_text_yields_segments(plain_text_manifest):
    manifest, scp = plain_text_manifest
    src = get_source("plain_text", input=manifest, wav_scp=scp)
    segs = list(src)
    assert len(segs) == 2
    assert segs[0].file_id == "rec1"
    assert segs[0].channel == 0
    assert segs[0].spkid == "spkA"
    assert segs[0].start == 0.0
    assert segs[0].end == 1.0
    assert segs[0].audio_ref is not None
    assert segs[0].audio_ref.path is not None


def test_plain_text_len(plain_text_manifest):
    manifest, scp = plain_text_manifest
    src = get_source("plain_text", input=manifest, wav_scp=scp)
    assert len(src) == 2


def test_plain_text_optional_lang_column(tmp_path: Path, wav_mono_16k: Path):
    scp = tmp_path / "wav.scp"; scp.write_text(f"rec1 {wav_mono_16k}\n")
    m = tmp_path / "m.tsv"
    m.write_text(
        "file_id\tchannel\tspkid\tstart\tend\tlang\n"
        "rec1\t0\tspkA\t0.0\t1.0\teng\n"
    )
    (seg,) = list(get_source("plain_text", input=m, wav_scp=scp))
    assert seg.lang == "eng"


def test_plain_text_missing_column(tmp_path: Path):
    scp = tmp_path / "wav.scp"; scp.write_text("rec1 /tmp/x.wav\n")
    m = tmp_path / "m.tsv"
    m.write_text("file_id\tchannel\tspkid\tstart\n" "rec1\t0\tspkA\t0.0\n")
    with pytest.raises(BadSpecError, match="end"):
        list(get_source("plain_text", input=m, wav_scp=scp))


def test_plain_text_unknown_file_id(tmp_path: Path, wav_mono_16k: Path):
    scp = tmp_path / "wav.scp"; scp.write_text(f"other {wav_mono_16k}\n")
    m = tmp_path / "m.tsv"
    m.write_text("file_id\tchannel\tspkid\tstart\tend\n" "rec1\t0\ts\t0.0\t1.0\n")
    with pytest.raises(BadSpecError, match="rec1"):
        list(get_source("plain_text", input=m, wav_scp=scp))


def test_get_source_unknown_name():
    with pytest.raises(KeyError):
        get_source("nope")
