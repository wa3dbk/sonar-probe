from pathlib import Path

import numpy as np
import pytest
from datasets import Audio, Dataset

from sonar_probe.errors import BadSpecError
from sonar_probe.sources import get_source


def test_hf_metadata_only_with_wav_scp(hf_mini_dataset):
    path, scp = hf_mini_dataset
    src = get_source("hf", input=path, wav_scp=scp)
    segs = list(src)
    assert len(segs) == 2
    assert segs[0].audio_ref is not None
    assert segs[0].audio_ref.path is not None


def test_hf_with_audio_column(tmp_path: Path, wav_mono_16k: Path):
    ds = Dataset.from_dict({
        "file_id": ["rec1"],
        "channel": [0],
        "spkid":   ["spkA"],
        "start":   [0.0],
        "end":     [1.0],
        "audio":   [str(wav_mono_16k)],
    }).cast_column("audio", Audio(sampling_rate=16_000))
    out = tmp_path / "ds"; ds.save_to_disk(str(out))
    src = get_source("hf", input=out)
    (seg,) = list(src)
    assert seg.audio_ref is not None
    assert seg.audio_ref.bytes_ is not None or seg.audio_ref.path is not None


def test_hf_missing_required_column(tmp_path: Path):
    ds = Dataset.from_dict({"file_id": ["x"]})
    out = tmp_path / "ds"; ds.save_to_disk(str(out))
    with pytest.raises(BadSpecError, match="channel"):
        list(get_source("hf", input=out))
