import numpy as np
import pytest

from sonar_probe.types import (
    EMBEDDING_DIM,
    AudioRef,
    EmbeddingRecord,
    Segment,
    TextItem,
)


def test_segment_id_is_deterministic():
    s = Segment(file_id="rec1", channel=0, spkid="spkA", start=0.0, end=1.0)
    assert s.segment_id == "rec1-0-spkA-0-1000"


def test_segment_id_handles_fractional_seconds():
    s = Segment(file_id="rec1", channel=1, spkid="spkA", start=0.123, end=0.456)
    assert s.segment_id == "rec1-1-spkA-123-456"


def test_segment_duration():
    s = Segment(file_id="r", channel=0, spkid="s", start=0.5, end=1.75)
    assert s.duration == pytest.approx(1.25)


def test_segment_rejects_end_leq_start():
    with pytest.raises(ValueError):
        Segment(file_id="r", channel=0, spkid="s", start=1.0, end=1.0)


def test_segment_rejects_negative_channel():
    with pytest.raises(ValueError):
        Segment(file_id="r", channel=-1, spkid="s", start=0.0, end=1.0)


def test_embedding_record_shape_enforced():
    emb = np.zeros((EMBEDDING_DIM,), dtype=np.float32)
    rec = EmbeddingRecord(
        segment_id="x", file_id="r", channel=0, spkid="s",
        start=0.0, end=1.0, lang=None, modality="speech", embedding=emb,
    )
    assert rec.embedding.shape == (1024,)
    assert rec.embedding.dtype == np.float32


def test_embedding_record_rejects_wrong_dim():
    with pytest.raises(ValueError):
        EmbeddingRecord(
            segment_id="x", file_id="r", channel=0, spkid="s",
            start=0.0, end=1.0, lang=None, modality="speech",
            embedding=np.zeros((512,), dtype=np.float32),
        )


def test_audio_ref_requires_path_or_bytes():
    with pytest.raises(ValueError):
        AudioRef()
    AudioRef(path="/tmp/x.wav")  # ok
    AudioRef(bytes_=b"\x00", sr=16_000)  # ok


def test_text_item_minimal():
    t = TextItem(text_id="t1", text="hello", lang="eng_Latn")
    assert t.text_id == "t1"
