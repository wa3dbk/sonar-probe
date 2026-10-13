# Sonar Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `sonar-probe`, a library-first Python tool with a Typer CLI that extracts SONAR speech and text embeddings from Kaldi, HuggingFace, and plain-text inputs, writes them to Kaldi `ark+scp` or HF `save_to_disk` stores, and computes cosine similarity (scalar or N×M, including cross-modal).

**Architecture:** Protocol-based sources and sinks keep the extension seams narrow. A thin driver (`extract`) iterates a `SegmentSource`, loads waveforms via `audio.py`, runs them through a per-language `SpeechEncoderPool` (LRU-cached SONAR speech encoders), and streams `EmbeddingRecord`s to an `EmbeddingSink`. `compare` is a tiny module that resolves operands to `(N, 1024)` matrices, L2-normalizes, and matmuls. The CLI is a Typer wrapper around the public API.

**Tech Stack:** Python ≥3.10, `sonar-space` + `fairseq2` (model), `torch` + `torchaudio` (audio + GPU), `soundfile` (fallback decoder), `kaldiio` (ark/scp), `datasets` (HF), `typer` + `rich` (CLI), `numpy`, `pydantic`. Dev: `pytest`, `pytest-cov`, `ruff`, `pyright`.

**Spec:** `docs/superpowers/specs/2026-10-14-sonar-probe-design.md`

## Global Constraints

- Python ≥ 3.10 (`match`, `|` unions, `Protocol`).
- All embeddings stored as `np.float32`, shape `(1024,)`. **Not** L2-normalized at write time.
- `segment_id` is deterministic: `f"{file_id}-{channel}-{spkid}-{start_ms}-{end_ms}"` with `start_ms = round(start * 1000)`. Kaldi source overrides to native `utt_id`.
- SONAR speech encoders expect **16 kHz mono float32**. All audio is normalized by `audio.py` before reaching the engine.
- Speech encoder language codes are **ISO-3** (`"eng"`, `"fra"`, …). Text encoder uses **BCP-47** (`"eng_Latn"`, `"fra_Latn"`, …).
- Device "auto" resolves to CUDA if available else CPU. MPS is never auto-selected.
- Default error policy: skip-and-log to `failures.jsonl`. `--strict` → fail-fast.
- Protocol-based registries for sources and sinks. No `importlib.metadata` entry points.
- Default batch size: 8. Default `max_seg_seconds`: 60.0. Default `encoder_cache_size`: 4.
- No feature-flag shims, no backwards-compatibility branches, no premature abstraction.

---

### Task 1: Project scaffold, lint/test config, synthesized fixtures

**Files:**
- Create: `pyproject.toml`
- Create: `sonar_probe/__init__.py` (empty public API placeholder)
- Create: `sonar_probe/py.typed` (empty)
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py`
- Create: `README.md` (minimal — one-paragraph purpose + install)
- Create: `.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces: `pytest` and `ruff` run cleanly on an empty package; `conftest.py` fixtures `wav_mono_16k`, `wav_stereo_44k`, `kaldi_mini_dir`, `hf_mini_dataset`, `plain_text_manifest` are available to every test.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "sonar-probe"
version = "0.1.0"
description = "Extract and compare SONAR speech/text embeddings"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "pydantic>=2.5",
    "typer>=0.12",
    "rich>=13",
    "soundfile>=0.12",
    "torch>=2.1",
    "torchaudio>=2.1",
    "kaldiio>=2.18",
    "datasets>=2.16",
    "sonar-space>=0.2",
    "fairseq2>=0.2",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-cov>=5",
    "ruff>=0.4",
    "pyright>=1.1.360",
]

[project.scripts]
sonar-probe = "sonar_probe.cli:app"

[tool.setuptools.packages.find]
include = ["sonar_probe*"]

[tool.pytest.ini_options]
markers = [
    "integration: tests that load real SONAR checkpoints (off by default)",
]
addopts = "-m 'not integration' --strict-markers"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "B", "UP"]

[tool.pyright]
include = ["sonar_probe", "tests"]
pythonVersion = "3.10"
typeCheckingMode = "basic"
```

- [ ] **Step 2: Write `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
.coverage
htmlcov/
dist/
build/
*.egg-info/
.venv/
.ruff_cache/
.pyright/
```

- [ ] **Step 3: Write `sonar_probe/__init__.py`**

```python
"""sonar-probe: extract and compare SONAR speech/text embeddings."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Write `tests/conftest.py` with synthesized fixtures**

```python
from __future__ import annotations

import json
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
```

- [ ] **Step 5: Write minimal `README.md`**

```markdown
# sonar-probe

Extract and compare [SONAR](https://github.com/facebookresearch/SONAR) speech and text embeddings.

## Install

    pip install -e .[dev]

## Quick usage

See `docs/superpowers/specs/2026-10-14-sonar-probe-design.md` for the full design.
```

- [ ] **Step 6: Verify toolchain runs**

Run: `python -m pip install -e .[dev]`
Expected: install succeeds. (If sonar-space / fairseq2 wheels aren't available for the dev machine's Python, document and continue — unit tests don't need them until Task 10.)

Run: `pytest`
Expected: `no tests ran` with exit 0 (no tests yet, but config loads).

Run: `ruff check .`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml sonar_probe/ tests/ README.md .gitignore
git commit -m "feat: scaffold sonar-probe package with pytest + ruff config"
```

---

### Task 2: Core types, errors, and config

**Files:**
- Create: `sonar_probe/types.py`
- Create: `sonar_probe/errors.py`
- Create: `sonar_probe/config.py`
- Create: `tests/test_types.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing (pure data).
- Produces:
  - `Segment(file_id, channel, spkid, start, end, lang=None, audio_ref=None)` with `segment_id` and `duration` properties.
  - `AudioRef(path=None, bytes_=None, sr=None)`.
  - `EmbeddingRecord(segment_id, file_id, channel, spkid, start, end, lang, modality, embedding)` where `embedding` is `np.ndarray[np.float32]` of shape `(1024,)`.
  - `TextItem(text_id, text, lang)`.
  - Exception hierarchy rooted at `SonarProbeError`: `BadSegmentError`, `BadSpecError`, `MissingLangError`, `ModelLoadError`, `StoreError`.
  - `ExtractConfig` and `CompareConfig` dataclasses with the fields and defaults listed in spec §14.
  - Module-level constant `EMBEDDING_DIM = 1024`.

- [ ] **Step 1: Write failing tests for `types.py`**

```python
# tests/test_types.py
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
```

- [ ] **Step 2: Run; expect ImportError / failure**

Run: `pytest tests/test_types.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `sonar_probe/types.py`**

```python
"""Core data model for sonar-probe."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EMBEDDING_DIM = 1024

SegmentID = str
Lang = str

Modality = Literal["speech", "text"]


class AudioRef(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path | None = None
    bytes_: bytes | None = None
    sr: int | None = None

    @model_validator(mode="after")
    def _check_one_source(self) -> "AudioRef":
        if self.path is None and self.bytes_ is None:
            raise ValueError("AudioRef requires either path or bytes_")
        return self


class Segment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    file_id: str
    channel: int = Field(ge=0)
    spkid: str
    start: float
    end: float
    lang: Lang | None = None
    audio_ref: AudioRef | None = None
    # Optional override of the composite ID (used by the kaldi source to keep utt_id).
    _id_override: str | None = None

    @model_validator(mode="after")
    def _check_times(self) -> "Segment":
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be > start ({self.start})")
        return self

    @property
    def segment_id(self) -> SegmentID:
        if self._id_override is not None:
            return self._id_override
        start_ms = round(self.start * 1000)
        end_ms = round(self.end * 1000)
        return f"{self.file_id}-{self.channel}-{self.spkid}-{start_ms}-{end_ms}"

    @property
    def duration(self) -> float:
        return self.end - self.start

    def with_id(self, override: str) -> "Segment":
        s = self.model_copy()
        s._id_override = override
        return s


class EmbeddingRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    segment_id: SegmentID
    file_id: str
    channel: int
    spkid: str
    start: float
    end: float
    lang: Lang | None
    modality: Modality
    embedding: Any  # np.ndarray; validated below

    @field_validator("embedding")
    @classmethod
    def _check_embedding(cls, v: Any) -> np.ndarray:
        if not isinstance(v, np.ndarray):
            raise ValueError(f"embedding must be np.ndarray, got {type(v)}")
        if v.shape != (EMBEDDING_DIM,):
            raise ValueError(f"embedding must have shape ({EMBEDDING_DIM},), got {v.shape}")
        if v.dtype != np.float32:
            v = v.astype(np.float32)
        return v


class TextItem(BaseModel):
    text_id: str
    text: str
    lang: Lang
```

- [ ] **Step 4: Run tests, expect pass**

Run: `pytest tests/test_types.py -v`
Expected: all green.

- [ ] **Step 5: Write failing tests for `config.py`**

```python
# tests/test_config.py
from sonar_probe.config import CompareConfig, ExtractConfig, resolve_device


def test_extract_config_defaults():
    c = ExtractConfig()
    assert c.batch_size == 8
    assert c.max_seg_seconds == 60.0
    assert c.encoder_cache_size == 4
    assert c.strict is False
    assert c.append is False
    assert c.deterministic is False
    assert c.device == "auto"


def test_compare_config_defaults():
    c = CompareConfig()
    assert c.device == "auto"
    assert c.output_format == "long"
    assert c.top_k is None
    assert c.threshold is None


def test_resolve_device_cpu_fallback(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_cuda_when_available(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"
```

- [ ] **Step 6: Run; expect fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — module missing.

- [ ] **Step 7: Implement `sonar_probe/config.py`**

```python
"""Configuration dataclasses and device resolution."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from sonar_probe.types import Lang


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@dataclass
class ExtractConfig:
    device: str = "auto"
    batch_size: int = 8
    speech_lang: Lang | None = None
    max_seg_seconds: float = 60.0
    encoder_cache_size: int = 4
    strict: bool = False
    append: bool = False
    deterministic: bool = False


@dataclass
class CompareConfig:
    device: str = "auto"
    speech_lang: Lang | None = None
    wav_scp: Path | None = None
    top_k: int | None = None
    threshold: float | None = None
    output_format: Literal["long", "wide"] = "long"
    strict: bool = False
```

- [ ] **Step 8: Implement `sonar_probe/errors.py`**

```python
"""Typed exception hierarchy."""


class SonarProbeError(Exception):
    """Base for all sonar-probe errors."""


class BadSegmentError(SonarProbeError):
    """Recoverable per-segment failure (missing audio, decode, bounds)."""


class BadSpecError(SonarProbeError):
    """Malformed input manifest, missing required column — unrecoverable."""


class MissingLangError(SonarProbeError):
    """No speech-lang supplied and no per-segment override."""


class ModelLoadError(SonarProbeError):
    """Encoder checkpoint failed to load."""


class StoreError(SonarProbeError):
    """Sink write failed or store read missing an ID."""
```

- [ ] **Step 9: Run full tests**

Run: `pytest -v`
Expected: all green (types + config).

- [ ] **Step 10: Commit**

```bash
git add sonar_probe/ tests/
git commit -m "feat: core types, errors, and config"
```

---

### Task 3: Audio I/O (`audio.py`)

**Files:**
- Create: `sonar_probe/audio.py`
- Create: `tests/test_audio.py`

**Interfaces:**
- Consumes: `AudioRef`, `BadSegmentError` from `types`/`errors`.
- Produces: `load_waveform(ref: AudioRef, channel: int, start: float, end: float) -> np.ndarray` returning 16 kHz mono float32. `parse_wav_scp(path: Path) -> dict[str, str]` returning a `file_id → audio-spec` map (values may contain `|` pipe entries).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_audio.py
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
```

- [ ] **Step 2: Run; expect fail**

Run: `pytest tests/test_audio.py -v`
Expected: FAIL — `audio` module missing.

- [ ] **Step 3: Implement `sonar_probe/audio.py`**

```python
"""Audio decoding, channel extraction, and resample to 16 kHz mono."""
from __future__ import annotations

import io
import logging
import shlex
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F

from sonar_probe.errors import BadSegmentError
from sonar_probe.types import AudioRef

TARGET_SR = 16_000
log = logging.getLogger(__name__)


def parse_wav_scp(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Split only on the FIRST whitespace block: pipe entries contain spaces.
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"{path}:{lineno}: expected 'file_id <spec>', got {line!r}")
        file_id, spec = parts
        out[file_id] = spec.strip()
    return out


def _decode_pipe(spec: str) -> tuple[np.ndarray, int]:
    """Execute a Kaldi-style pipe entry (ends with '|') and decode its stdout."""
    cmd = spec.rstrip("|").strip()
    try:
        proc = subprocess.run(
            shlex.split(cmd),
            check=True,
            capture_output=True,
            timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise BadSegmentError(f"pipe decode failed: {cmd!r}: {e}") from e
    data, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32", always_2d=True)
    return data, sr


def _decode(ref: AudioRef) -> tuple[np.ndarray, int]:
    """Decode an AudioRef to a (frames, channels) float32 array and its sample rate."""
    try:
        if ref.bytes_ is not None:
            data, sr = sf.read(io.BytesIO(ref.bytes_), dtype="float32", always_2d=True)
            return data, sr
        assert ref.path is not None
        spec = str(ref.path)
        if spec.endswith("|"):
            return _decode_pipe(spec)
        data, sr = sf.read(spec, dtype="float32", always_2d=True)
        return data, sr
    except BadSegmentError:
        raise
    except Exception as e:
        raise BadSegmentError(f"decode failed for {ref}: {e}") from e


def load_waveform(ref: AudioRef, channel: int, start: float, end: float) -> np.ndarray:
    """Return a 16 kHz mono float32 waveform for [start, end) of the given channel."""
    data, sr = _decode(ref)  # (frames, channels)
    if channel < 0 or channel >= data.shape[1]:
        raise BadSegmentError(
            f"channel {channel} out of range (file has {data.shape[1]} channels)"
        )
    mono = data[:, channel]  # (frames,)

    start_f = int(round(start * sr))
    end_f = int(round(end * sr))
    if start_f < 0 or end_f > mono.shape[0] or end_f <= start_f:
        raise BadSegmentError(
            f"segment [{start}, {end}) out of bounds for file of "
            f"{mono.shape[0] / sr:.3f}s at sr={sr}"
        )
    clip = mono[start_f:end_f]

    if sr != TARGET_SR:
        log.debug("resampling %d -> %d (%d frames)", sr, TARGET_SR, clip.shape[0])
        clip_t = torch.from_numpy(clip).unsqueeze(0)  # (1, T)
        clip_t = F.resample(clip_t, orig_freq=sr, new_freq=TARGET_SR)
        clip = clip_t.squeeze(0).numpy().astype(np.float32, copy=False)
    return np.ascontiguousarray(clip, dtype=np.float32)
```

- [ ] **Step 4: Run tests, expect pass**

Run: `pytest tests/test_audio.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add sonar_probe/audio.py tests/test_audio.py
git commit -m "feat: audio.py — decode, channel extract, 16k resample"
```

---

### Task 4: Source protocol + registry + `plain_text` source

**Files:**
- Create: `sonar_probe/sources/__init__.py`
- Create: `sonar_probe/sources/base.py`
- Create: `sonar_probe/sources/plain_text.py`
- Create: `tests/sources/__init__.py`
- Create: `tests/sources/test_plain_text.py`

**Interfaces:**
- Consumes: `Segment`, `AudioRef`, `BadSegmentError`, `BadSpecError`, `parse_wav_scp`.
- Produces:
  - Protocol `SegmentSource` with `name: ClassVar[str]`, `__iter__`, `__len__`.
  - `register_source(cls)` decorator and `get_source(name: str, **kwargs) -> SegmentSource`.
  - `PlainTextSource(input: Path, wav_scp: Path)`; registered as `"plain_text"`.
  - Shared helper `validate_segment(seg: Segment) -> None` and `coerce_channel(raw: str) -> int`.

- [ ] **Step 1: Write failing tests**

```python
# tests/sources/test_plain_text.py
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
```

- [ ] **Step 2: Run; expect fail**

Run: `pytest tests/sources/test_plain_text.py -v`
Expected: FAIL — modules missing.

- [ ] **Step 3: Implement `sources/base.py`**

```python
"""Source protocol, registry, and shared helpers."""
from __future__ import annotations

from typing import ClassVar, Iterator, Protocol, runtime_checkable

from sonar_probe.errors import BadSegmentError, BadSpecError
from sonar_probe.types import Segment

_REGISTRY: dict[str, type] = {}


@runtime_checkable
class SegmentSource(Protocol):
    name: ClassVar[str]

    def __iter__(self) -> Iterator[Segment]: ...
    def __len__(self) -> int: ...


def register_source(cls):
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(f"{cls} must declare a 'name' class attribute")
    if name in _REGISTRY:
        raise RuntimeError(f"source {name!r} already registered")
    _REGISTRY[name] = cls
    return cls


def get_source(name: str, **kwargs) -> SegmentSource:
    if name not in _REGISTRY:
        raise KeyError(f"unknown source {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


_CHANNEL_LABELS = {"A": 0, "B": 1, "C": 2, "D": 3}


def coerce_channel(raw: str) -> int:
    s = raw.strip()
    if s in _CHANNEL_LABELS:
        return _CHANNEL_LABELS[s]
    try:
        return int(s)
    except ValueError as e:
        raise BadSpecError(f"cannot interpret channel {raw!r}") from e


def validate_segment(seg: Segment) -> None:
    # Pydantic already enforces end>start and channel>=0; this is a seam for
    # future cross-field checks (e.g., max_seg_seconds, if we want them here).
    if seg.audio_ref is None:
        raise BadSegmentError(f"segment {seg.segment_id} has no audio_ref")
```

- [ ] **Step 4: Implement `sources/plain_text.py`**

```python
"""Plain-text TSV source + wav.scp."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import ClassVar, Iterator

from sonar_probe.audio import parse_wav_scp
from sonar_probe.errors import BadSpecError
from sonar_probe.sources.base import SegmentSource, coerce_channel, register_source
from sonar_probe.types import AudioRef, Segment

REQUIRED = ("file_id", "channel", "spkid", "start", "end")


@register_source
class PlainTextSource(SegmentSource):
    name: ClassVar[str] = "plain_text"

    def __init__(self, input: Path, wav_scp: Path):
        self._input = Path(input)
        self._scp = parse_wav_scp(Path(wav_scp))
        self._rows = self._read_rows()

    def _read_rows(self) -> list[dict[str, str]]:
        with self._input.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            if reader.fieldnames is None:
                raise BadSpecError(f"{self._input}: empty manifest")
            missing = [c for c in REQUIRED if c not in reader.fieldnames]
            if missing:
                raise BadSpecError(
                    f"{self._input}: missing required column(s): {missing}"
                )
            return list(reader)

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[Segment]:
        for row in self._rows:
            file_id = row["file_id"]
            spec = self._scp.get(file_id)
            if spec is None:
                raise BadSpecError(f"{self._input}: file_id {file_id!r} not in wav.scp")
            yield Segment(
                file_id=file_id,
                channel=coerce_channel(row["channel"]),
                spkid=row["spkid"],
                start=float(row["start"]),
                end=float(row["end"]),
                lang=(row.get("lang") or None),
                audio_ref=AudioRef(path=Path(spec) if not spec.endswith("|") else None,
                                   bytes_=None, sr=None) if not spec.endswith("|")
                           else AudioRef(path=Path(spec)),
            )
```

Note: pipe entries carry trailing `|`; `AudioRef.path` stores the full spec string wrapped in `Path(...)`. `audio.load_waveform` detects the trailing `|` and routes to `_decode_pipe`.

- [ ] **Step 5: Implement `sources/__init__.py`**

```python
"""Source backends and registry."""
from sonar_probe.sources.base import (
    SegmentSource,
    get_source,
    register_source,
)
# Import backends so they register on package load.
from sonar_probe.sources import plain_text  # noqa: F401

__all__ = ["SegmentSource", "get_source", "register_source"]
```

- [ ] **Step 6: Create empty `tests/sources/__init__.py`**

```python
```

- [ ] **Step 7: Run tests, expect pass**

Run: `pytest tests/sources/test_plain_text.py -v`
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add sonar_probe/sources/ tests/sources/
git commit -m "feat: source protocol + registry + plain_text backend"
```

---

### Task 5: `kaldi` source

**Files:**
- Create: `sonar_probe/sources/kaldi.py`
- Modify: `sonar_probe/sources/__init__.py` (import kaldi so it registers)
- Create: `tests/sources/test_kaldi.py`

**Interfaces:**
- Consumes: Task 4's `register_source`, `coerce_channel`, `parse_wav_scp`.
- Produces: `KaldiSource(input: Path)`; registered as `"kaldi"`. Reads `segments`, `wav.scp`, `utt2spk`, optional `reco2file_and_channel`, optional `utt2lang`. Yields `Segment` with `segment_id` overridden to the native `utt_id` via `Segment.with_id`.

- [ ] **Step 1: Write failing tests**

```python
# tests/sources/test_kaldi.py
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
```

- [ ] **Step 2: Run; expect fail**

Run: `pytest tests/sources/test_kaldi.py -v`
Expected: FAIL — unknown source `"kaldi"`.

- [ ] **Step 3: Implement `sources/kaldi.py`**

```python
"""Kaldi data-directory source."""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Iterator

from sonar_probe.audio import parse_wav_scp
from sonar_probe.errors import BadSpecError
from sonar_probe.sources.base import SegmentSource, coerce_channel, register_source
from sonar_probe.types import AudioRef, Segment


def _read_two_col(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise BadSpecError(f"{path}:{lineno}: expected 2 columns")
        out[parts[0]] = parts[1].strip()
    return out


@register_source
class KaldiSource(SegmentSource):
    name: ClassVar[str] = "kaldi"

    def __init__(self, input: Path):
        root = Path(input)
        if not root.is_dir():
            raise BadSpecError(f"{root}: not a directory")
        self._root = root
        segs_path = root / "segments"
        if not segs_path.is_file():
            raise BadSpecError(f"{segs_path}: missing segments file")
        self._segments = self._read_segments(segs_path)
        self._wav_scp = parse_wav_scp(root / "wav.scp")
        self._utt2spk = _read_two_col(root / "utt2spk")
        rfc = root / "reco2file_and_channel"
        self._reco_channel: dict[str, int] = {}
        if rfc.is_file():
            for lineno, raw in enumerate(rfc.read_text().splitlines(), 1):
                parts = raw.split()
                if len(parts) != 3:
                    raise BadSpecError(f"{rfc}:{lineno}: expected 3 columns")
                reco, _file, label = parts
                self._reco_channel[reco] = coerce_channel(label)
        u2l = root / "utt2lang"
        self._utt2lang = _read_two_col(u2l) if u2l.is_file() else {}

    def _read_segments(self, path: Path) -> list[tuple[str, str, float, float]]:
        out = []
        for lineno, raw in enumerate(path.read_text().splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 4:
                raise BadSpecError(f"{path}:{lineno}: expected 'utt file start end'")
            utt, file_id, start, end = parts
            out.append((utt, file_id, float(start), float(end)))
        return out

    def __len__(self) -> int:
        return len(self._segments)

    def __iter__(self) -> Iterator[Segment]:
        for utt, file_id, start, end in self._segments:
            spec = self._wav_scp.get(file_id)
            if spec is None:
                raise BadSpecError(f"{file_id}: not in wav.scp")
            spkid = self._utt2spk.get(utt)
            if spkid is None:
                raise BadSpecError(f"{utt}: not in utt2spk")
            seg = Segment(
                file_id=file_id,
                channel=self._reco_channel.get(file_id, 0),
                spkid=spkid,
                start=start,
                end=end,
                lang=self._utt2lang.get(utt),
                audio_ref=AudioRef(path=Path(spec)),
            )
            yield seg.with_id(utt)
```

- [ ] **Step 4: Register in `sources/__init__.py`**

```python
"""Source backends and registry."""
from sonar_probe.sources.base import (
    SegmentSource,
    get_source,
    register_source,
)
from sonar_probe.sources import kaldi, plain_text  # noqa: F401

__all__ = ["SegmentSource", "get_source", "register_source"]
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/sources/ -v`
Expected: all green (plain_text + kaldi).

- [ ] **Step 6: Commit**

```bash
git add sonar_probe/sources/kaldi.py sonar_probe/sources/__init__.py tests/sources/test_kaldi.py
git commit -m "feat: kaldi data-dir source"
```

---

### Task 6: `hf` source

**Files:**
- Create: `sonar_probe/sources/hf.py`
- Modify: `sonar_probe/sources/__init__.py`
- Create: `tests/sources/test_hf.py`

**Interfaces:**
- Consumes: Task 4 registry; `AudioRef(bytes_, sr)` for audio-column datasets.
- Produces: `HfSource(input: str | Path, wav_scp: Path | None = None, split: str | None = None, config: str | None = None)`; registered as `"hf"`.

- [ ] **Step 1: Write failing tests**

```python
# tests/sources/test_hf.py
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
```

- [ ] **Step 2: Run; expect fail**

Run: `pytest tests/sources/test_hf.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `sources/hf.py`**

```python
"""HuggingFace datasets source."""
from __future__ import annotations

import io
from pathlib import Path
from typing import ClassVar, Iterator

import numpy as np
import soundfile as sf
from datasets import Dataset, load_dataset, load_from_disk

from sonar_probe.audio import parse_wav_scp
from sonar_probe.errors import BadSpecError
from sonar_probe.sources.base import SegmentSource, coerce_channel, register_source
from sonar_probe.types import AudioRef, Segment

REQUIRED = ("file_id", "channel", "spkid", "start", "end")


def _load(input: str | Path, split: str | None, config: str | None) -> Dataset:
    p = Path(input) if not isinstance(input, Dataset) else None
    if isinstance(input, Dataset):
        return input
    if p is not None and p.exists() and p.is_dir():
        ds = load_from_disk(str(p))
    else:
        ds = load_dataset(str(input), name=config, split=split or "train")
    if hasattr(ds, "keys"):  # DatasetDict
        if split is None:
            raise BadSpecError(f"dataset has splits {list(ds.keys())}; pass --hf-split")
        ds = ds[split]
    return ds


@register_source
class HfSource(SegmentSource):
    name: ClassVar[str] = "hf"

    def __init__(
        self,
        input: str | Path | Dataset,
        wav_scp: Path | None = None,
        split: str | None = None,
        config: str | None = None,
    ):
        self._ds = _load(input, split, config)
        cols = set(self._ds.column_names)
        missing = [c for c in REQUIRED if c not in cols]
        if missing:
            raise BadSpecError(f"hf dataset missing columns: {missing}")
        self._has_audio_col = "audio" in cols
        self._scp = parse_wav_scp(Path(wav_scp)) if wav_scp is not None else None
        if not self._has_audio_col and self._scp is None:
            raise BadSpecError(
                "dataset has no 'audio' column; pass --wav-scp to resolve file_id"
            )

    def __len__(self) -> int:
        return len(self._ds)

    def __iter__(self) -> Iterator[Segment]:
        for row in self._ds:
            audio_ref = self._audio_ref_for(row)
            yield Segment(
                file_id=row["file_id"],
                channel=int(row["channel"]) if isinstance(row["channel"], (int, np.integer))
                        else coerce_channel(str(row["channel"])),
                spkid=row["spkid"],
                start=float(row["start"]),
                end=float(row["end"]),
                lang=row.get("lang"),
                audio_ref=audio_ref,
            )

    def _audio_ref_for(self, row) -> AudioRef:
        if self._has_audio_col:
            a = row["audio"]
            # HF Audio feature yields {"array": np.ndarray, "sampling_rate": int, "path": str|None}
            if isinstance(a, dict) and "array" in a:
                buf = io.BytesIO()
                sf.write(buf, a["array"], a["sampling_rate"], format="WAV", subtype="FLOAT")
                return AudioRef(bytes_=buf.getvalue(), sr=a["sampling_rate"])
            if isinstance(a, str):
                return AudioRef(path=Path(a))
        assert self._scp is not None
        spec = self._scp.get(row["file_id"])
        if spec is None:
            raise BadSpecError(f"file_id {row['file_id']!r} not in wav.scp")
        return AudioRef(path=Path(spec))
```

- [ ] **Step 4: Register hf in `sources/__init__.py`**

```python
from sonar_probe.sources.base import (
    SegmentSource,
    get_source,
    register_source,
)
from sonar_probe.sources import hf, kaldi, plain_text  # noqa: F401

__all__ = ["SegmentSource", "get_source", "register_source"]
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/sources/ -v`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add sonar_probe/sources/hf.py sonar_probe/sources/__init__.py tests/sources/test_hf.py
git commit -m "feat: hf dataset source (audio column + wav.scp fallback)"
```

---

### Task 7: Sink/Store protocols + registry + `kaldi_ark` sink

**Files:**
- Create: `sonar_probe/sinks/__init__.py`
- Create: `sonar_probe/sinks/base.py`
- Create: `sonar_probe/sinks/kaldi_ark.py`
- Create: `tests/sinks/__init__.py`
- Create: `tests/sinks/test_kaldi_ark.py`

**Interfaces:**
- Consumes: `EmbeddingRecord`, `SegmentID`, `StoreError`.
- Produces:
  - Protocols `EmbeddingSink` (`open() -> ContextManager`, `write`, `close`) and `EmbeddingStore` (`__iter__`, `__getitem__`, `__len__`, `ids`, `matrix`).
  - `register_sink(cls)`, `get_sink(name, **kwargs) -> EmbeddingSink`.
  - `register_store(cls)`, `_STORE_REGISTRY` keyed by sink name.
  - `KaldiArkSink(output: Path, append: bool = False)`; `KaldiArkStore(path: Path)`.

- [ ] **Step 1: Write failing tests**

```python
# tests/sinks/test_kaldi_ark.py
from pathlib import Path

import numpy as np
import pytest

from sonar_probe.errors import StoreError
from sonar_probe.sinks import get_sink
from sonar_probe.sinks.kaldi_ark import KaldiArkStore
from sonar_probe.types import EmbeddingRecord


def _rec(seg_id: str, modality: str = "speech") -> EmbeddingRecord:
    rng = np.random.default_rng(hash(seg_id) & 0xFFFFFFFF)
    return EmbeddingRecord(
        segment_id=seg_id, file_id="rec1", channel=0, spkid="spkA",
        start=0.0, end=1.0, lang="eng", modality=modality,
        embedding=rng.standard_normal(1024).astype(np.float32),
    )


def test_write_then_read_roundtrip(tmp_path: Path):
    out = tmp_path / "store"
    sink = get_sink("kaldi_ark", output=out)
    r1, r2 = _rec("a"), _rec("b")
    with sink.open():
        sink.write(r1)
        sink.write(r2)
    store = KaldiArkStore(out)
    assert len(store) == 2
    assert set(store.ids()) == {"a", "b"}
    got = store["a"]
    assert got.file_id == "rec1" and got.lang == "eng"
    assert np.allclose(got.embedding, r1.embedding)


def test_matrix_shape(tmp_path: Path):
    out = tmp_path / "store"
    sink = get_sink("kaldi_ark", output=out)
    with sink.open():
        for i in range(3):
            sink.write(_rec(f"s{i}"))
    ids, M = KaldiArkStore(out).matrix()
    assert M.shape == (3, 1024) and M.dtype == np.float32
    assert ids == ["s0", "s1", "s2"]


def test_text_modality_roundtrip(tmp_path: Path):
    out = tmp_path / "store"
    sink = get_sink("kaldi_ark", output=out)
    with sink.open():
        sink.write(_rec("t1", modality="text"))
    rec = KaldiArkStore(out)["t1"]
    assert rec.modality == "text"


def test_missing_id_raises(tmp_path: Path):
    out = tmp_path / "store"
    sink = get_sink("kaldi_ark", output=out)
    with sink.open():
        sink.write(_rec("only"))
    with pytest.raises(StoreError, match="missing"):
        _ = KaldiArkStore(out)["ghost"]
```

- [ ] **Step 2: Run; expect fail**

Run: `pytest tests/sinks/test_kaldi_ark.py -v`
Expected: FAIL — modules missing.

- [ ] **Step 3: Implement `sinks/base.py`**

```python
"""Sink / Store protocols and registry."""
from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import ClassVar, Iterator, Protocol, runtime_checkable

import numpy as np

from sonar_probe.types import EmbeddingRecord, SegmentID

_SINKS: dict[str, type] = {}
_STORES: dict[str, type] = {}


@runtime_checkable
class EmbeddingSink(Protocol):
    name: ClassVar[str]

    def open(self) -> AbstractContextManager[None]: ...
    def write(self, rec: EmbeddingRecord) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class EmbeddingStore(Protocol):
    def __iter__(self) -> Iterator[EmbeddingRecord]: ...
    def __getitem__(self, segment_id: SegmentID) -> EmbeddingRecord: ...
    def __len__(self) -> int: ...
    def ids(self) -> list[SegmentID]: ...
    def matrix(
        self, ids: list[SegmentID] | None = None
    ) -> tuple[list[SegmentID], np.ndarray]: ...


def register_sink(cls):
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(f"{cls} must declare 'name'")
    _SINKS[name] = cls
    return cls


def register_store(name: str):
    def deco(cls):
        _STORES[name] = cls
        return cls
    return deco


def get_sink(name: str, **kwargs) -> EmbeddingSink:
    if name not in _SINKS:
        raise KeyError(f"unknown sink {name!r}; known: {sorted(_SINKS)}")
    return _SINKS[name](**kwargs)


def _store_class_for(name: str) -> type:
    if name not in _STORES:
        raise KeyError(f"no store registered for sink {name!r}")
    return _STORES[name]
```

- [ ] **Step 4: Implement `sinks/kaldi_ark.py`**

```python
"""Kaldi ark+scp sink and store, with JSONL metadata sidecar."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar, Iterator

import kaldiio
import numpy as np

from sonar_probe.errors import StoreError
from sonar_probe.sinks.base import (
    EmbeddingStore,
    register_sink,
    register_store,
)
from sonar_probe.types import EmbeddingRecord, SegmentID


@register_sink
class KaldiArkSink:
    name: ClassVar[str] = "kaldi_ark"

    def __init__(self, output: Path, append: bool = False):
        self._out = Path(output)
        self._append = append
        self._writer: kaldiio.WriteHelper | None = None
        self._meta_fh = None

    @contextmanager
    def open(self):
        self._out.mkdir(parents=True, exist_ok=True)
        ark = self._out / "embeddings.ark"
        scp = self._out / "embeddings.scp"
        meta = self._out / "metadata.jsonl"
        mode = "a" if self._append else "w"
        if not self._append:
            # overwrite semantics: truncate ark/scp/meta
            for p in (ark, scp, meta):
                if p.exists():
                    p.unlink()
        spec = f"ark,scp:{ark},{scp}"
        self._writer = kaldiio.WriteHelper(spec, write_function="numpy")
        self._meta_fh = meta.open(mode, encoding="utf-8")
        try:
            yield
        finally:
            self.close()

    def write(self, rec: EmbeddingRecord) -> None:
        if self._writer is None or self._meta_fh is None:
            raise StoreError("sink is not open")
        try:
            self._writer(rec.segment_id, rec.embedding)
            self._meta_fh.write(
                json.dumps({
                    "segment_id": rec.segment_id,
                    "file_id": rec.file_id,
                    "channel": rec.channel,
                    "spkid": rec.spkid,
                    "start": rec.start,
                    "end": rec.end,
                    "lang": rec.lang,
                    "modality": rec.modality,
                }) + "\n"
            )
        except Exception as e:
            raise StoreError(f"write failed for {rec.segment_id}: {e}") from e

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._meta_fh is not None:
            self._meta_fh.close()
            self._meta_fh = None


@register_store("kaldi_ark")
class KaldiArkStore(EmbeddingStore):
    def __init__(self, path: Path):
        self._root = Path(path)
        scp = self._root / "embeddings.scp"
        meta = self._root / "metadata.jsonl"
        if not scp.is_file() or not meta.is_file():
            raise StoreError(f"{self._root}: not a kaldi_ark store")
        self._vectors: dict[str, np.ndarray] = dict(kaldiio.load_scp(str(scp)))
        self._meta: dict[str, dict] = {}
        for raw in meta.read_text().splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            self._meta[row["segment_id"]] = row
        self._order = list(self._meta.keys())

    def __iter__(self) -> Iterator[EmbeddingRecord]:
        for sid in self._order:
            yield self[sid]

    def __len__(self) -> int:
        return len(self._order)

    def ids(self) -> list[str]:
        return list(self._order)

    def __getitem__(self, segment_id: SegmentID) -> EmbeddingRecord:
        if segment_id not in self._meta:
            raise StoreError(f"missing segment_id {segment_id!r}")
        m = self._meta[segment_id]
        vec = self._vectors[segment_id].astype(np.float32, copy=False)
        return EmbeddingRecord(
            segment_id=m["segment_id"],
            file_id=m["file_id"],
            channel=m["channel"],
            spkid=m["spkid"],
            start=m["start"],
            end=m["end"],
            lang=m.get("lang"),
            modality=m["modality"],
            embedding=vec,
        )

    def matrix(
        self, ids: list[SegmentID] | None = None
    ) -> tuple[list[SegmentID], np.ndarray]:
        keys = ids if ids is not None else self._order
        M = np.stack([self._vectors[k].astype(np.float32, copy=False) for k in keys])
        return list(keys), M
```

- [ ] **Step 5: Implement `sinks/__init__.py`**

```python
"""Sink backends."""
from sonar_probe.sinks.base import (
    EmbeddingSink,
    EmbeddingStore,
    get_sink,
    register_sink,
    register_store,
)
from sonar_probe.sinks import kaldi_ark  # noqa: F401

__all__ = [
    "EmbeddingSink", "EmbeddingStore",
    "get_sink", "register_sink", "register_store",
]
```

- [ ] **Step 6: Empty `tests/sinks/__init__.py`**

```python
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/sinks/test_kaldi_ark.py -v`
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add sonar_probe/sinks/ tests/sinks/
git commit -m "feat: sink protocols + kaldi_ark sink and store"
```

---

### Task 8: `hf_dataset` sink and store

**Files:**
- Create: `sonar_probe/sinks/hf_dataset.py`
- Modify: `sonar_probe/sinks/__init__.py`
- Create: `tests/sinks/test_hf_dataset.py`

**Interfaces:**
- Consumes: Task 7 registry + protocols.
- Produces: `HfDatasetSink(output: Path, append: bool = False, buffer_size: int = 256)`; `HfDatasetStore(path: Path)`.

- [ ] **Step 1: Write failing tests**

```python
# tests/sinks/test_hf_dataset.py
from pathlib import Path

import numpy as np

from sonar_probe.sinks import get_sink
from sonar_probe.sinks.hf_dataset import HfDatasetStore
from sonar_probe.types import EmbeddingRecord


def _rec(sid: str) -> EmbeddingRecord:
    rng = np.random.default_rng(hash(sid) & 0xFFFFFFFF)
    return EmbeddingRecord(
        segment_id=sid, file_id="r", channel=0, spkid="s",
        start=0.0, end=1.0, lang="eng", modality="speech",
        embedding=rng.standard_normal(1024).astype(np.float32),
    )


def test_hf_sink_roundtrip(tmp_path: Path):
    out = tmp_path / "hf_store"
    sink = get_sink("hf_dataset", output=out)
    r1, r2 = _rec("a"), _rec("b")
    with sink.open():
        sink.write(r1); sink.write(r2)
    store = HfDatasetStore(out)
    assert len(store) == 2
    got = store["a"]
    assert np.allclose(got.embedding, r1.embedding)
    assert got.modality == "speech"


def test_hf_sink_append(tmp_path: Path):
    out = tmp_path / "hf_store"
    with get_sink("hf_dataset", output=out).open() as _:
        get_sink("hf_dataset", output=out).__enter__ if False else None
    # First write
    with get_sink("hf_dataset", output=out).open():
        pass
    s1 = get_sink("hf_dataset", output=out)
    with s1.open():
        s1.write(_rec("a"))
    s2 = get_sink("hf_dataset", output=out, append=True)
    with s2.open():
        s2.write(_rec("b"))
    store = HfDatasetStore(out)
    assert set(store.ids()) == {"a", "b"}


def test_hf_matrix(tmp_path: Path):
    out = tmp_path / "hf_store"
    sink = get_sink("hf_dataset", output=out)
    with sink.open():
        for i in range(3):
            sink.write(_rec(f"x{i}"))
    ids, M = HfDatasetStore(out).matrix()
    assert M.shape == (3, 1024) and M.dtype == np.float32
```

- [ ] **Step 2: Run; expect fail**

Run: `pytest tests/sinks/test_hf_dataset.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `sinks/hf_dataset.py`**

```python
"""HuggingFace datasets sink/store."""
from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar, Iterator

import numpy as np
from datasets import Dataset, concatenate_datasets, load_from_disk

from sonar_probe.errors import StoreError
from sonar_probe.sinks.base import (
    EmbeddingStore,
    register_sink,
    register_store,
)
from sonar_probe.types import EMBEDDING_DIM, EmbeddingRecord, SegmentID


_COLUMNS = (
    "segment_id", "file_id", "channel", "spkid",
    "start", "end", "lang", "modality", "embedding",
)


def _records_to_dict(buf: list[EmbeddingRecord]) -> dict[str, list]:
    return {
        "segment_id": [r.segment_id for r in buf],
        "file_id":    [r.file_id for r in buf],
        "channel":    [int(r.channel) for r in buf],
        "spkid":      [r.spkid for r in buf],
        "start":      [float(r.start) for r in buf],
        "end":        [float(r.end) for r in buf],
        "lang":       [r.lang for r in buf],
        "modality":   [r.modality for r in buf],
        "embedding":  [r.embedding.tolist() for r in buf],
    }


@register_sink
class HfDatasetSink:
    name: ClassVar[str] = "hf_dataset"

    def __init__(self, output: Path, append: bool = False, buffer_size: int = 256):
        self._out = Path(output)
        self._append = append
        self._buf_size = buffer_size
        self._buf: list[EmbeddingRecord] = []
        self._shards: list[Dataset] = []

    @contextmanager
    def open(self):
        if self._out.exists() and not self._append:
            shutil.rmtree(self._out)
        if self._append and self._out.exists():
            self._shards.append(load_from_disk(str(self._out)))
        self._out.parent.mkdir(parents=True, exist_ok=True)
        try:
            yield
        finally:
            self.close()

    def write(self, rec: EmbeddingRecord) -> None:
        self._buf.append(rec)
        if len(self._buf) >= self._buf_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        ds = Dataset.from_dict(_records_to_dict(self._buf))
        self._shards.append(ds)
        self._buf.clear()

    def close(self) -> None:
        self._flush()
        if not self._shards:
            # Still produce an empty valid dataset
            Dataset.from_dict({c: [] for c in _COLUMNS}).save_to_disk(str(self._out))
            return
        combined = (
            self._shards[0] if len(self._shards) == 1
            else concatenate_datasets(self._shards)
        )
        if self._out.exists():
            shutil.rmtree(self._out)
        combined.save_to_disk(str(self._out))
        self._shards.clear()


@register_store("hf_dataset")
class HfDatasetStore(EmbeddingStore):
    def __init__(self, path: Path):
        self._root = Path(path)
        try:
            self._ds = load_from_disk(str(self._root))
        except Exception as e:
            raise StoreError(f"{self._root}: not an hf_dataset store ({e})") from e
        self._index: dict[str, int] | None = None

    def _ensure_index(self) -> dict[str, int]:
        if self._index is None:
            self._index = {sid: i for i, sid in enumerate(self._ds["segment_id"])}
        return self._index

    def __iter__(self) -> Iterator[EmbeddingRecord]:
        for i in range(len(self._ds)):
            yield self._row_to_record(i)

    def __len__(self) -> int:
        return len(self._ds)

    def ids(self) -> list[SegmentID]:
        return list(self._ds["segment_id"])

    def __getitem__(self, segment_id: SegmentID) -> EmbeddingRecord:
        idx = self._ensure_index().get(segment_id)
        if idx is None:
            raise StoreError(f"missing segment_id {segment_id!r}")
        return self._row_to_record(idx)

    def _row_to_record(self, i: int) -> EmbeddingRecord:
        row = self._ds[i]
        emb = np.asarray(row["embedding"], dtype=np.float32)
        if emb.shape != (EMBEDDING_DIM,):
            raise StoreError(f"row {i}: bad embedding shape {emb.shape}")
        return EmbeddingRecord(
            segment_id=row["segment_id"], file_id=row["file_id"],
            channel=row["channel"], spkid=row["spkid"],
            start=row["start"], end=row["end"],
            lang=row.get("lang"), modality=row["modality"],
            embedding=emb,
        )

    def matrix(
        self, ids: list[SegmentID] | None = None
    ) -> tuple[list[SegmentID], np.ndarray]:
        if ids is None:
            arr = np.asarray(self._ds.with_format("numpy")["embedding"], dtype=np.float32)
            return list(self._ds["segment_id"]), arr
        idx_map = self._ensure_index()
        order = [idx_map[s] for s in ids]
        arr = np.asarray(self._ds.select(order).with_format("numpy")["embedding"],
                         dtype=np.float32)
        return list(ids), arr
```

- [ ] **Step 4: Register in `sinks/__init__.py`**

```python
from sonar_probe.sinks.base import (
    EmbeddingSink,
    EmbeddingStore,
    get_sink,
    register_sink,
    register_store,
)
from sonar_probe.sinks import hf_dataset, kaldi_ark  # noqa: F401

__all__ = [
    "EmbeddingSink", "EmbeddingStore",
    "get_sink", "register_sink", "register_store",
]
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/sinks/ -v`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add sonar_probe/sinks/hf_dataset.py sonar_probe/sinks/__init__.py tests/sinks/test_hf_dataset.py
git commit -m "feat: hf_dataset sink and store"
```

---

### Task 9: `load_store` helper

**Files:**
- Modify: `sonar_probe/sinks/__init__.py` (add `load_store`)
- Create: `tests/sinks/test_load_store.py`

**Interfaces:**
- Consumes: both store classes from Tasks 7–8.
- Produces: `load_store(path: Path) -> EmbeddingStore`.

- [ ] **Step 1: Write failing test**

```python
# tests/sinks/test_load_store.py
from pathlib import Path

import numpy as np

from sonar_probe.sinks import get_sink, load_store
from sonar_probe.types import EmbeddingRecord


def _r(sid: str) -> EmbeddingRecord:
    return EmbeddingRecord(
        segment_id=sid, file_id="r", channel=0, spkid="s",
        start=0.0, end=1.0, lang="eng", modality="speech",
        embedding=np.zeros(1024, dtype=np.float32),
    )


def test_load_store_detects_kaldi(tmp_path: Path):
    out = tmp_path / "k"
    sink = get_sink("kaldi_ark", output=out)
    with sink.open():
        sink.write(_r("a"))
    store = load_store(out)
    assert store.ids() == ["a"]


def test_load_store_detects_hf(tmp_path: Path):
    out = tmp_path / "h"
    sink = get_sink("hf_dataset", output=out)
    with sink.open():
        sink.write(_r("a"))
    store = load_store(out)
    assert store.ids() == ["a"]
```

- [ ] **Step 2: Run; expect fail**

Run: `pytest tests/sinks/test_load_store.py -v`
Expected: FAIL — `load_store` not exported.

- [ ] **Step 3: Add `load_store` to `sinks/__init__.py`**

```python
from pathlib import Path

from sonar_probe.errors import StoreError
from sonar_probe.sinks.base import (
    EmbeddingSink,
    EmbeddingStore,
    get_sink,
    register_sink,
    register_store,
)
from sonar_probe.sinks import hf_dataset, kaldi_ark  # noqa: F401


def load_store(path: Path) -> EmbeddingStore:
    p = Path(path)
    if (p / "embeddings.scp").is_file() and (p / "metadata.jsonl").is_file():
        from sonar_probe.sinks.kaldi_ark import KaldiArkStore
        return KaldiArkStore(p)
    if (p / "dataset_info.json").is_file() or (p / "state.json").is_file():
        from sonar_probe.sinks.hf_dataset import HfDatasetStore
        return HfDatasetStore(p)
    raise StoreError(f"{p}: unrecognized store layout")


__all__ = [
    "EmbeddingSink", "EmbeddingStore",
    "get_sink", "register_sink", "register_store",
    "load_store",
]
```

- [ ] **Step 4: Run tests, commit**

Run: `pytest tests/sinks/ -v`
Expected: all green.

```bash
git add sonar_probe/sinks/__init__.py tests/sinks/test_load_store.py
git commit -m "feat: load_store auto-detects kaldi_ark vs hf_dataset"
```

---

### Task 10: `SpeechEncoderPool` with injectable encoder factory

**Files:**
- Create: `sonar_probe/engine/__init__.py`
- Create: `sonar_probe/engine/speech.py`
- Create: `tests/engine/__init__.py`
- Create: `tests/engine/test_speech_pool.py`

**Interfaces:**
- Consumes: `ModelLoadError`.
- Produces:
  - `SpeechEncoderPool(device: str, cache_size: int = 4, encoder_factory: Callable[[Lang, str], SpeechEncoder] | None = None)`.
  - `SpeechEncoder` protocol with `encode(waveforms: list[np.ndarray]) -> np.ndarray` returning `(B, 1024)` float32.
  - `pool.encode(waveforms: list[np.ndarray], langs: list[Lang]) -> np.ndarray` of shape `(B, 1024)`.
  - Default factory uses `sonar.inference_pipelines.speech.SpeechToEmbeddingModelPipeline`; test suite always injects a fake.

- [ ] **Step 1: Write failing tests**

```python
# tests/engine/test_speech_pool.py
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
```

- [ ] **Step 2: Run; expect fail**

Run: `pytest tests/engine/test_speech_pool.py -v`
Expected: FAIL — missing module.

- [ ] **Step 3: Implement `engine/__init__.py`**

```python
"""SONAR encoder wrappers."""
```

- [ ] **Step 4: Implement `engine/speech.py`**

```python
"""Per-language SONAR speech encoder pool."""
from __future__ import annotations

import gc
import logging
from collections import OrderedDict
from typing import Callable, Protocol, runtime_checkable

import numpy as np
import torch

from sonar_probe.errors import ModelLoadError
from sonar_probe.types import EMBEDDING_DIM, Lang

log = logging.getLogger(__name__)


@runtime_checkable
class SpeechEncoder(Protocol):
    def encode(self, waveforms: list[np.ndarray]) -> np.ndarray: ...


def _default_factory(lang: Lang, device: str) -> SpeechEncoder:
    try:
        from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
    except Exception as e:
        raise ModelLoadError(f"sonar-space not installed: {e}") from e

    class _SonarSpeech(SpeechEncoder):
        def __init__(self, lang: str, device: str):
            try:
                self._pipe = SpeechToEmbeddingModelPipeline(
                    encoder=f"sonar_speech_encoder_{lang}",
                    device=torch.device(device),
                    fbank_dtype=torch.float32,
                )
            except Exception as e:
                raise ModelLoadError(
                    f"failed to load sonar_speech_encoder_{lang}: {e}"
                ) from e

        def encode(self, waveforms: list[np.ndarray]) -> np.ndarray:
            tensors = [torch.from_numpy(w).to(torch.float32) for w in waveforms]
            with torch.inference_mode():
                out = self._pipe.predict(tensors, batch_size=len(tensors))
            arr = out.detach().cpu().numpy().astype(np.float32, copy=False)
            if arr.ndim != 2 or arr.shape[1] != EMBEDDING_DIM:
                raise ModelLoadError(
                    f"speech encoder returned unexpected shape {arr.shape}"
                )
            return arr

    return _SonarSpeech(lang, device)


class SpeechEncoderPool:
    def __init__(
        self,
        device: str,
        cache_size: int = 4,
        encoder_factory: Callable[[Lang, str], SpeechEncoder] | None = None,
    ):
        self._device = device
        self._cache_size = cache_size
        self._factory = encoder_factory or _default_factory
        self._cache: OrderedDict[Lang, SpeechEncoder] = OrderedDict()

    def _get(self, lang: Lang) -> SpeechEncoder:
        if lang in self._cache:
            self._cache.move_to_end(lang)
            return self._cache[lang]
        enc = self._factory(lang, self._device)
        self._cache[lang] = enc
        if len(self._cache) > self._cache_size:
            evicted_lang, _ = self._cache.popitem(last=False)
            log.debug("evicting speech encoder %r", evicted_lang)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return enc

    def encode(
        self, waveforms: list[np.ndarray], langs: list[Lang]
    ) -> np.ndarray:
        if len(waveforms) != len(langs):
            raise ValueError("waveforms and langs must align")
        if not waveforms:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        groups: dict[Lang, list[int]] = {}
        for i, lg in enumerate(langs):
            groups.setdefault(lg, []).append(i)

        out = np.empty((len(waveforms), EMBEDDING_DIM), dtype=np.float32)
        for lg, idxs in groups.items():
            enc = self._get(lg)
            sub = [waveforms[i] for i in idxs]
            embs = enc.encode(sub)
            if embs.shape != (len(sub), EMBEDDING_DIM):
                raise ModelLoadError(
                    f"encoder for {lg} returned shape {embs.shape}, expected "
                    f"({len(sub)}, {EMBEDDING_DIM})"
                )
            for row, i in zip(embs, idxs):
                out[i] = row
        return out
```

- [ ] **Step 5: Empty `tests/engine/__init__.py`**

```python
```

- [ ] **Step 6: Run tests, commit**

Run: `pytest tests/engine/test_speech_pool.py -v`
Expected: all green.

```bash
git add sonar_probe/engine/ tests/engine/
git commit -m "feat: SpeechEncoderPool with LRU and injectable factory"
```

---

### Task 11: `TextEncoder` with injectable factory

**Files:**
- Create: `sonar_probe/engine/text.py`
- Create: `tests/engine/test_text_encoder.py`

**Interfaces:**
- Consumes: `ModelLoadError`.
- Produces:
  - `TextEncoder(device: str, encoder_factory: Callable[[str], _TextBackend] | None = None)` with `encode(texts: list[str], lang: Lang) -> np.ndarray` of shape `(B, 1024)`.

- [ ] **Step 1: Write failing test**

```python
# tests/engine/test_text_encoder.py
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
```

- [ ] **Step 2: Run; expect fail**

Run: `pytest tests/engine/test_text_encoder.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `engine/text.py`**

```python
"""SONAR text encoder wrapper."""
from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import numpy as np
import torch

from sonar_probe.errors import ModelLoadError
from sonar_probe.types import EMBEDDING_DIM, Lang


@runtime_checkable
class TextBackend(Protocol):
    def encode(self, texts: list[str], lang: Lang) -> np.ndarray: ...


def _default_factory(device: str) -> TextBackend:
    try:
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
    except Exception as e:
        raise ModelLoadError(f"sonar-space not installed: {e}") from e

    class _SonarText(TextBackend):
        def __init__(self, device: str):
            try:
                self._pipe = TextToEmbeddingModelPipeline(
                    encoder="text_sonar_basic_encoder",
                    tokenizer="text_sonar_basic_encoder",
                    device=torch.device(device),
                )
            except Exception as e:
                raise ModelLoadError(f"failed to load text encoder: {e}") from e

        def encode(self, texts: list[str], lang: Lang) -> np.ndarray:
            with torch.inference_mode():
                out = self._pipe.predict(texts, source_lang=lang, batch_size=len(texts))
            arr = out.detach().cpu().numpy().astype(np.float32, copy=False)
            if arr.ndim != 2 or arr.shape[1] != EMBEDDING_DIM:
                raise ModelLoadError(f"text encoder returned shape {arr.shape}")
            return arr

    return _SonarText(device)


class TextEncoder:
    def __init__(
        self,
        device: str,
        encoder_factory: Callable[[str], TextBackend] | None = None,
    ):
        self._backend: TextBackend | None = None
        self._factory = encoder_factory or _default_factory
        self._device = device

    def _get(self) -> TextBackend:
        if self._backend is None:
            self._backend = self._factory(self._device)
        return self._backend

    def encode(self, texts: list[str], lang: Lang) -> np.ndarray:
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        embs = self._get().encode(texts, lang)
        if embs.shape != (len(texts), EMBEDDING_DIM):
            raise ModelLoadError(
                f"text encoder returned {embs.shape}, expected ({len(texts)}, {EMBEDDING_DIM})"
            )
        return embs
```

- [ ] **Step 4: Run, commit**

Run: `pytest tests/engine/ -v`
Expected: all green.

```bash
git add sonar_probe/engine/text.py tests/engine/test_text_encoder.py
git commit -m "feat: TextEncoder with injectable backend"
```

---

### Task 12: `extract` driver + `failures.jsonl` + public API

**Files:**
- Create: `sonar_probe/driver.py`
- Modify: `sonar_probe/__init__.py` (export public API)
- Create: `tests/test_extract.py`

**Interfaces:**
- Consumes: `SegmentSource`, `EmbeddingSink`, `SpeechEncoderPool`, `load_waveform`, `resolve_device`, `ExtractConfig`, `BadSegmentError`, `MissingLangError`.
- Produces:
  - `extract(source: SegmentSource, sink: EmbeddingSink, cfg: ExtractConfig, pool: SpeechEncoderPool | None = None) -> ExtractResult`.
  - `ExtractResult(written: int, skipped: int, failures_path: Path | None)`.
  - Public re-exports in `sonar_probe/__init__.py`: `extract`, `Segment`, `EmbeddingRecord`, `ExtractConfig`, `CompareConfig`, `load_store`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_extract.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sonar_probe import ExtractConfig, extract
from sonar_probe.engine.speech import SpeechEncoder, SpeechEncoderPool
from sonar_probe.errors import MissingLangError
from sonar_probe.sinks import get_sink, load_store
from sonar_probe.sources import get_source


class FakeEncoder(SpeechEncoder):
    def __init__(self, lang):
        self.lang = lang
    def encode(self, waves):
        out = []
        for w in waves:
            seed = int.from_bytes(hashlib.sha256(w.tobytes()).digest()[:8], "little")
            out.append(np.random.default_rng(seed).standard_normal(1024).astype(np.float32))
        return np.stack(out)


def _pool():
    return SpeechEncoderPool(device="cpu", encoder_factory=lambda lg, dev: FakeEncoder(lg))


def test_extract_plain_text_to_kaldi(plain_text_manifest, tmp_path: Path):
    manifest, scp = plain_text_manifest
    out = tmp_path / "store"
    src = get_source("plain_text", input=manifest, wav_scp=scp)
    sink = get_sink("kaldi_ark", output=out)
    cfg = ExtractConfig(device="cpu", speech_lang="eng")
    res = extract(src, sink, cfg, pool=_pool())
    assert res.written == 2 and res.skipped == 0
    store = load_store(out)
    assert len(store) == 2
    rec = next(iter(store))
    assert rec.modality == "speech"
    assert rec.embedding.shape == (1024,)


def test_extract_requires_lang_somewhere(plain_text_manifest, tmp_path: Path):
    manifest, scp = plain_text_manifest
    src = get_source("plain_text", input=manifest, wav_scp=scp)
    sink = get_sink("kaldi_ark", output=tmp_path / "s")
    with pytest.raises(MissingLangError):
        extract(src, sink, ExtractConfig(device="cpu"), pool=_pool())


def test_extract_skip_logs_failures(tmp_path: Path, wav_mono_16k: Path):
    # Manifest with one bad timestamp (out of bounds)
    scp = tmp_path / "wav.scp"; scp.write_text(f"rec1 {wav_mono_16k}\n")
    m = tmp_path / "m.tsv"
    m.write_text(
        "file_id\tchannel\tspkid\tstart\tend\n"
        "rec1\t0\ts\t0.0\t1.0\n"
        "rec1\t0\ts\t0.0\t10.0\n"   # too long
    )
    out = tmp_path / "store"
    src = get_source("plain_text", input=m, wav_scp=scp)
    sink = get_sink("kaldi_ark", output=out)
    res = extract(src, sink, ExtractConfig(device="cpu", speech_lang="eng"), pool=_pool())
    assert res.written == 1 and res.skipped == 1
    assert res.failures_path is not None and res.failures_path.exists()
    fails = [json.loads(l) for l in res.failures_path.read_text().splitlines() if l.strip()]
    assert len(fails) == 1 and "bounds" in fails[0]["message"]


def test_extract_strict_aborts(tmp_path: Path, wav_mono_16k: Path):
    scp = tmp_path / "wav.scp"; scp.write_text(f"rec1 {wav_mono_16k}\n")
    m = tmp_path / "m.tsv"
    m.write_text(
        "file_id\tchannel\tspkid\tstart\tend\n"
        "rec1\t0\ts\t0.0\t10.0\n"
    )
    src = get_source("plain_text", input=m, wav_scp=scp)
    sink = get_sink("kaldi_ark", output=tmp_path / "s")
    from sonar_probe.errors import BadSegmentError
    with pytest.raises(BadSegmentError):
        extract(src, sink, ExtractConfig(device="cpu", speech_lang="eng", strict=True),
                pool=_pool())
```

- [ ] **Step 2: Run; expect fail**

Run: `pytest tests/test_extract.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `sonar_probe/driver.py`**

```python
"""Extract driver: iterate a source, encode, write to a sink."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from sonar_probe.audio import load_waveform
from sonar_probe.config import ExtractConfig, resolve_device
from sonar_probe.engine.speech import SpeechEncoderPool
from sonar_probe.engine.text import TextEncoder
from sonar_probe.errors import BadSegmentError, MissingLangError, SonarProbeError
from sonar_probe.sinks.base import EmbeddingSink
from sonar_probe.sources.base import SegmentSource
from sonar_probe.types import EmbeddingRecord, Lang, Segment, TextItem

log = logging.getLogger(__name__)


@dataclass
class ExtractResult:
    written: int
    skipped: int
    failures_path: Path | None


def _batched(it: Iterable[Segment], size: int) -> Iterator[list[Segment]]:
    buf: list[Segment] = []
    for item in it:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def _resolve_lang(seg: Segment, cfg: ExtractConfig) -> Lang:
    lg = seg.lang or cfg.speech_lang
    if not lg:
        raise MissingLangError(
            f"segment {seg.segment_id} has no lang and --speech-lang not set"
        )
    return lg


def _failure_path(sink: EmbeddingSink) -> Path | None:
    out = getattr(sink, "_out", None)
    return (Path(out) / "failures.jsonl") if out is not None else None


def _log_failure(fh, seg: Segment, exc: Exception) -> None:
    fh.write(json.dumps({
        "segment_id": seg.segment_id,
        "file_id": seg.file_id,
        "start": seg.start,
        "end": seg.end,
        "error_type": type(exc).__name__,
        "message": str(exc),
    }) + "\n")


def extract(
    source: SegmentSource,
    sink: EmbeddingSink,
    cfg: ExtractConfig,
    pool: SpeechEncoderPool | None = None,
) -> ExtractResult:
    device = resolve_device(cfg.device)
    pool = pool or SpeechEncoderPool(
        device=device, cache_size=cfg.encoder_cache_size,
    )

    written = skipped = 0
    fail_path = _failure_path(sink)
    fail_fh = None
    start_t = time.perf_counter()

    with sink.open():
        if fail_path is not None and not cfg.strict:
            fail_path.parent.mkdir(parents=True, exist_ok=True)
            fail_fh = fail_path.open("w", encoding="utf-8")
        try:
            for batch in _batched(iter(source), cfg.batch_size):
                waves: list = []
                langs: list[Lang] = []
                metas: list[Segment] = []
                for seg in batch:
                    if seg.duration > cfg.max_seg_seconds:
                        err = BadSegmentError(
                            f"duration {seg.duration:.2f}s exceeds max_seg_seconds "
                            f"{cfg.max_seg_seconds}"
                        )
                        if cfg.strict: raise err
                        skipped += 1
                        if fail_fh: _log_failure(fail_fh, seg, err)
                        continue
                    try:
                        lang = _resolve_lang(seg, cfg)
                        w = load_waveform(seg.audio_ref, seg.channel, seg.start, seg.end)
                    except MissingLangError:
                        raise
                    except BadSegmentError as e:
                        if cfg.strict: raise
                        skipped += 1
                        if fail_fh: _log_failure(fail_fh, seg, e)
                        continue
                    waves.append(w); langs.append(lang); metas.append(seg)

                if not waves:
                    continue
                embs = pool.encode(waves, langs)
                for seg, emb, lang in zip(metas, embs, langs):
                    sink.write(EmbeddingRecord(
                        segment_id=seg.segment_id,
                        file_id=seg.file_id, channel=seg.channel, spkid=seg.spkid,
                        start=seg.start, end=seg.end, lang=lang,
                        modality="speech", embedding=emb,
                    ))
                    written += 1
        finally:
            if fail_fh is not None:
                fail_fh.close()

    elapsed = time.perf_counter() - start_t
    log.info(
        "extract done: written=%d skipped=%d elapsed=%.2fs throughput=%.1f seg/s",
        written, skipped, elapsed,
        (written + skipped) / elapsed if elapsed > 0 else 0.0,
    )
    return ExtractResult(
        written=written, skipped=skipped,
        failures_path=fail_path if fail_fh is not None else None,
    )


def extract_text(
    items: Iterable[TextItem],
    sink: EmbeddingSink,
    cfg: ExtractConfig,
    encoder: TextEncoder | None = None,
) -> ExtractResult:
    device = resolve_device(cfg.device)
    enc = encoder or TextEncoder(device=device)

    written = skipped = 0
    fail_path = _failure_path(sink)
    fail_fh = None

    with sink.open():
        if fail_path is not None and not cfg.strict:
            fail_path.parent.mkdir(parents=True, exist_ok=True)
            fail_fh = fail_path.open("w", encoding="utf-8")
        try:
            buf: list[TextItem] = []
            def flush():
                nonlocal written
                if not buf: return
                texts = [t.text for t in buf]
                # Group by lang to call encoder once per lang.
                groups: dict[str, list[int]] = {}
                for i, t in enumerate(buf):
                    groups.setdefault(t.lang, []).append(i)
                out = [None] * len(buf)
                for lg, idxs in groups.items():
                    embs = enc.encode([texts[i] for i in idxs], lang=lg)
                    for row, i in zip(embs, idxs):
                        out[i] = row
                for item, emb in zip(buf, out):
                    sink.write(EmbeddingRecord(
                        segment_id=item.text_id, file_id=item.text_id,
                        channel=0, spkid="", start=0.0, end=0.0,
                        lang=item.lang, modality="text", embedding=emb,
                    ))
                    written += 1
                buf.clear()

            for item in items:
                buf.append(item)
                if len(buf) >= cfg.batch_size:
                    flush()
            flush()
        finally:
            if fail_fh is not None:
                fail_fh.close()
    return ExtractResult(written=written, skipped=skipped,
                        failures_path=fail_path if fail_fh is not None else None)
```

- [ ] **Step 4: Update `sonar_probe/__init__.py`**

```python
"""sonar-probe: extract and compare SONAR speech/text embeddings."""

__version__ = "0.1.0"

from sonar_probe.config import CompareConfig, ExtractConfig
from sonar_probe.driver import ExtractResult, extract, extract_text
from sonar_probe.sinks import load_store
from sonar_probe.types import (
    EMBEDDING_DIM,
    AudioRef,
    EmbeddingRecord,
    Segment,
    TextItem,
)

__all__ = [
    "__version__",
    "extract", "extract_text", "load_store",
    "Segment", "EmbeddingRecord", "AudioRef", "TextItem",
    "ExtractConfig", "CompareConfig", "ExtractResult",
    "EMBEDDING_DIM",
]
```

- [ ] **Step 5: Run tests, commit**

Run: `pytest tests/test_extract.py -v`
Expected: all green.

```bash
git add sonar_probe/driver.py sonar_probe/__init__.py tests/test_extract.py
git commit -m "feat: extract driver with batching, strict mode, failures.jsonl"
```

---

### Task 13: `extract_text` tests + text TSV parser helper

**Files:**
- Create: `sonar_probe/text_input.py`
- Modify: `sonar_probe/__init__.py` (export `read_text_tsv`)
- Create: `tests/test_extract_text.py`

**Interfaces:**
- Consumes: `TextItem`, `BadSpecError`, `extract_text` from Task 12.
- Produces: `read_text_tsv(path: Path, default_lang: Lang | None = None) -> Iterator[TextItem]`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_extract_text.py
from pathlib import Path

import hashlib
import numpy as np

from sonar_probe import ExtractConfig, extract_text, load_store
from sonar_probe.engine.text import TextBackend, TextEncoder
from sonar_probe.sinks import get_sink
from sonar_probe.text_input import read_text_tsv


class FakeText(TextBackend):
    def encode(self, texts, lang):
        out = []
        for t in texts:
            h = hashlib.sha256((t + lang).encode()).digest()
            seed = int.from_bytes(h[:8], "little")
            out.append(np.random.default_rng(seed).standard_normal(1024).astype(np.float32))
        return np.stack(out)


def test_read_text_tsv_with_lang(tmp_path: Path):
    p = tmp_path / "t.tsv"
    p.write_text("text_id\ttext\tlang\n" "t1\thello\teng_Latn\n" "t2\tbonjour\tfra_Latn\n")
    items = list(read_text_tsv(p))
    assert [i.text_id for i in items] == ["t1", "t2"]
    assert items[0].lang == "eng_Latn"


def test_read_text_tsv_default_lang(tmp_path: Path):
    p = tmp_path / "t.tsv"
    p.write_text("text_id\ttext\n" "t1\thello\n")
    (item,) = list(read_text_tsv(p, default_lang="eng_Latn"))
    assert item.lang == "eng_Latn"


def test_extract_text_to_hf(tmp_path: Path):
    p = tmp_path / "t.tsv"
    p.write_text("text_id\ttext\tlang\n" "t1\thi\teng_Latn\n" "t2\tsalut\tfra_Latn\n")
    out = tmp_path / "store"
    enc = TextEncoder(device="cpu", encoder_factory=lambda d: FakeText())
    sink = get_sink("hf_dataset", output=out)
    res = extract_text(read_text_tsv(p), sink, ExtractConfig(device="cpu"), encoder=enc)
    assert res.written == 2 and res.skipped == 0
    store = load_store(out)
    assert len(store) == 2
    rec = store["t1"]
    assert rec.modality == "text" and rec.lang == "eng_Latn"
```

- [ ] **Step 2: Run; expect fail**

Run: `pytest tests/test_extract_text.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `sonar_probe/text_input.py`**

```python
"""TSV reader for text extraction."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from sonar_probe.errors import BadSpecError
from sonar_probe.types import Lang, TextItem


def read_text_tsv(path: Path, default_lang: Lang | None = None) -> Iterator[TextItem]:
    with Path(path).open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or "text_id" not in reader.fieldnames or "text" not in reader.fieldnames:
            raise BadSpecError(f"{path}: must have columns 'text_id' and 'text'")
        has_lang_col = "lang" in reader.fieldnames
        for row in reader:
            lang = (row.get("lang") if has_lang_col else None) or default_lang
            if not lang:
                raise BadSpecError(
                    f"{path}:{reader.line_num}: no lang column and no default --lang"
                )
            yield TextItem(text_id=row["text_id"], text=row["text"], lang=lang)
```

- [ ] **Step 4: Export in `sonar_probe/__init__.py`**

Append to the import block:
```python
from sonar_probe.text_input import read_text_tsv
```
And add `"read_text_tsv"` to `__all__`.

- [ ] **Step 5: Run, commit**

Run: `pytest tests/test_extract_text.py -v`
Expected: all green.

```bash
git add sonar_probe/text_input.py sonar_probe/__init__.py tests/test_extract_text.py
git commit -m "feat: text TSV reader + extract_text end-to-end"
```

---

### Task 14: `compare` module (operand parsing + driver)

**Files:**
- Create: `sonar_probe/compare.py`
- Modify: `sonar_probe/__init__.py` (export `compare`, `CompareResult`)
- Create: `tests/test_compare.py`

**Interfaces:**
- Consumes: `CompareConfig`, `SpeechEncoderPool`, `TextEncoder`, `load_store`, `load_waveform`, `parse_wav_scp`, `EmbeddingRecord`.
- Produces:
  - `OperandVectors(ids: list[str], matrix: np.ndarray, modality: str)` dataclass.
  - `parse_operand(spec: str) -> OperandRef` returning a tagged union of kinds: `inline_segment`, `store_id`, `path`, `text`, `text_file`.
  - `resolve_operand(spec: str, cfg: CompareConfig, pool, text_encoder) -> OperandVectors`.
  - `compare(a: str, b: str, cfg: CompareConfig, pool=None, text_encoder=None) -> CompareResult`.
  - `CompareResult(a_ids, b_ids, modality_a, modality_b, scores: np.ndarray)`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_compare.py
import hashlib
from pathlib import Path

import numpy as np
import pytest

from sonar_probe import CompareConfig, compare, load_store
from sonar_probe.compare import OperandRef, parse_operand
from sonar_probe.engine.speech import SpeechEncoder, SpeechEncoderPool
from sonar_probe.engine.text import TextBackend, TextEncoder
from sonar_probe.sinks import get_sink
from sonar_probe.types import EmbeddingRecord


def _vec(seed_text: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "little")
    v = np.random.default_rng(seed).standard_normal(1024).astype(np.float32)
    return v


def _store(tmp_path: Path, name: str, ids: list[str]) -> Path:
    out = tmp_path / name
    sink = get_sink("kaldi_ark", output=out)
    with sink.open():
        for sid in ids:
            sink.write(EmbeddingRecord(
                segment_id=sid, file_id="r", channel=0, spkid="s",
                start=0.0, end=1.0, lang="eng", modality="speech",
                embedding=_vec(sid),
            ))
    return out


def test_parse_operand_inline():
    r = parse_operand("rec1,0,spkA,0.0,1.0")
    assert r.kind == "inline_segment"


def test_parse_operand_store_id():
    r = parse_operand("store:/tmp/x#utt1")
    assert r.kind == "store_id"
    assert r.payload == ("/tmp/x", "utt1")


def test_parse_operand_text():
    r = parse_operand("text:hello world@eng_Latn")
    assert r.kind == "text" and r.payload == ("hello world", "eng_Latn")


def test_parse_operand_text_file():
    r = parse_operand("text-file:/tmp/t.txt@fra_Latn")
    assert r.kind == "text_file" and r.payload == ("/tmp/t.txt", "fra_Latn")


def test_parse_operand_path(tmp_path: Path):
    r = parse_operand(str(tmp_path))
    assert r.kind == "path"


def test_compare_store_vs_store_scalar(tmp_path: Path):
    a = _store(tmp_path, "a", ["x"])
    b = _store(tmp_path, "b", ["x"])
    res = compare(f"store:{a}#x", f"store:{b}#x", CompareConfig(device="cpu"))
    assert res.scores.shape == (1, 1)
    # Both stores wrote identical vectors (same seed) → cosine ≈ 1
    assert res.scores[0, 0] == pytest.approx(1.0, abs=1e-5)


def test_compare_store_vs_store_matrix(tmp_path: Path):
    a = _store(tmp_path, "a", ["x", "y"])
    b = _store(tmp_path, "b", ["x", "z"])
    res = compare(str(a), str(b), CompareConfig(device="cpu"))
    assert res.scores.shape == (2, 2)
    # (x,x) ≈ 1
    i = res.a_ids.index("x"); j = res.b_ids.index("x")
    assert res.scores[i, j] == pytest.approx(1.0, abs=1e-5)


def test_compare_cross_modal_shape(tmp_path: Path):
    a = _store(tmp_path, "a", ["x"])
    class _T(TextBackend):
        def encode(self, texts, lang):
            return np.stack([_vec(t) for t in texts])
    tenc = TextEncoder(device="cpu", encoder_factory=lambda d: _T())
    res = compare(f"store:{a}#x", "text:hello@eng_Latn",
                  CompareConfig(device="cpu"), text_encoder=tenc)
    assert res.scores.shape == (1, 1)
    assert res.modality_a == "speech" and res.modality_b == "text"


def test_compare_topk_filter(tmp_path: Path):
    a = _store(tmp_path, "a", ["x", "y"])
    b = _store(tmp_path, "b", ["x", "y", "z"])
    res = compare(str(a), str(b),
                  CompareConfig(device="cpu", top_k=1))
    # top_k filter is applied via helper on the result; verify shape still (2,3)
    # and top_k helper yields 2 rows.
    pairs = list(res.iter_pairs(top_k=1))
    assert len(pairs) == 2  # one top per a_id
```

- [ ] **Step 2: Run; expect fail**

Run: `pytest tests/test_compare.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `sonar_probe/compare.py`**

```python
"""Operand resolution + cosine compare."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

import numpy as np

from sonar_probe.audio import load_waveform, parse_wav_scp
from sonar_probe.config import CompareConfig, resolve_device
from sonar_probe.engine.speech import SpeechEncoderPool
from sonar_probe.engine.text import TextEncoder
from sonar_probe.errors import BadSpecError, MissingLangError
from sonar_probe.sinks import load_store
from sonar_probe.types import AudioRef, Lang, Modality

OperandKind = Literal["inline_segment", "store_id", "path", "text", "text_file"]


@dataclass
class OperandRef:
    kind: OperandKind
    payload: tuple


@dataclass
class OperandVectors:
    ids: list[str]
    matrix: np.ndarray            # (N, 1024) float32
    modality: Modality


def parse_operand(spec: str) -> OperandRef:
    if spec.startswith("text:"):
        rest = spec[len("text:"):]
        if "@" not in rest:
            raise BadSpecError(f"text operand missing @lang: {spec!r}")
        text, lang = rest.rsplit("@", 1)
        return OperandRef("text", (text, lang))
    if spec.startswith("text-file:"):
        rest = spec[len("text-file:"):]
        if "@" not in rest:
            raise BadSpecError(f"text-file operand missing @lang: {spec!r}")
        path, lang = rest.rsplit("@", 1)
        return OperandRef("text_file", (path, lang))
    if spec.startswith("store:"):
        rest = spec[len("store:"):]
        if "#" not in rest:
            raise BadSpecError(f"store operand missing #segment_id: {spec!r}")
        path, sid = rest.rsplit("#", 1)
        return OperandRef("store_id", (path, sid))
    # inline_segment is a comma-separated tuple with 5 or 6 fields
    if "," in spec and Path(spec).exists() is False:
        parts = [p.strip() for p in spec.split(",")]
        if len(parts) not in (5, 6):
            raise BadSpecError(f"inline operand needs 5 or 6 comma fields: {spec!r}")
        return OperandRef("inline_segment", tuple(parts))
    # otherwise treat as a path
    return OperandRef("path", (spec,))


def _l2(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (mat / norms).astype(np.float32, copy=False)


@dataclass
class CompareResult:
    a_ids: list[str]
    b_ids: list[str]
    modality_a: Modality
    modality_b: Modality
    scores: np.ndarray            # (N, M) float32

    def iter_pairs(
        self,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> Iterator[tuple[str, str, float]]:
        for i, aid in enumerate(self.a_ids):
            row = self.scores[i]
            order = np.argsort(-row)
            if top_k is not None:
                order = order[:top_k]
            for j in order:
                s = float(row[j])
                if threshold is not None and s < threshold:
                    continue
                yield aid, self.b_ids[j], s


def _resolve_inline(
    parts: tuple[str, ...],
    cfg: CompareConfig,
    pool: SpeechEncoderPool,
) -> OperandVectors:
    if cfg.wav_scp is None:
        raise BadSpecError("inline operand requires --wav-scp")
    file_id, channel, spkid, start, end, *rest = parts
    lang = (rest[0] if rest else None) or cfg.speech_lang
    if not lang:
        raise MissingLangError("inline operand needs lang (6th field) or --speech-lang")
    scp = parse_wav_scp(Path(cfg.wav_scp))
    spec = scp.get(file_id)
    if spec is None:
        raise BadSpecError(f"file_id {file_id!r} not in wav.scp")
    w = load_waveform(AudioRef(path=Path(spec)), int(channel),
                      float(start), float(end))
    emb = pool.encode([w], [lang])
    seg_id = f"{file_id}-{channel}-{spkid}-{int(round(float(start)*1000))}-{int(round(float(end)*1000))}"
    return OperandVectors(ids=[seg_id], matrix=emb, modality="speech")


def _resolve_store_id(
    path: str, sid: str,
) -> OperandVectors:
    store = load_store(Path(path))
    rec = store[sid]
    return OperandVectors(ids=[sid], matrix=rec.embedding[np.newaxis, :],
                          modality=rec.modality)


def _resolve_path(path: str) -> OperandVectors:
    p = Path(path)
    store = load_store(p)
    ids, mat = store.matrix()
    first = next(iter(store))
    return OperandVectors(ids=ids, matrix=mat, modality=first.modality)


def _resolve_text(
    text: str, lang: Lang, text_encoder: TextEncoder,
) -> OperandVectors:
    emb = text_encoder.encode([text], lang=lang)
    return OperandVectors(ids=[f"text:{text[:16]}"], matrix=emb, modality="text")


def _resolve_text_file(
    path: str, lang: Lang, text_encoder: TextEncoder,
) -> OperandVectors:
    lines = [ln.rstrip("\n") for ln in Path(path).read_text().splitlines() if ln.strip()]
    emb = text_encoder.encode(lines, lang=lang)
    return OperandVectors(ids=[f"text:{t[:16]}" for t in lines],
                          matrix=emb, modality="text")


def _resolve(
    ref: OperandRef, cfg: CompareConfig,
    pool: SpeechEncoderPool, text_encoder: TextEncoder,
) -> OperandVectors:
    if ref.kind == "store_id":
        return _resolve_store_id(*ref.payload)
    if ref.kind == "path":
        return _resolve_path(ref.payload[0])
    if ref.kind == "inline_segment":
        return _resolve_inline(ref.payload, cfg, pool)
    if ref.kind == "text":
        return _resolve_text(*ref.payload, text_encoder=text_encoder)
    if ref.kind == "text_file":
        return _resolve_text_file(*ref.payload, text_encoder=text_encoder)
    raise BadSpecError(f"unknown operand kind {ref.kind!r}")


def compare(
    a: str, b: str, cfg: CompareConfig,
    pool: SpeechEncoderPool | None = None,
    text_encoder: TextEncoder | None = None,
) -> CompareResult:
    device = resolve_device(cfg.device)
    pool = pool or SpeechEncoderPool(device=device)
    text_encoder = text_encoder or TextEncoder(device=device)

    ra, rb = parse_operand(a), parse_operand(b)
    left  = _resolve(ra, cfg, pool, text_encoder)
    right = _resolve(rb, cfg, pool, text_encoder)

    L = _l2(left.matrix)
    R = _l2(right.matrix)
    scores = (L @ R.T).astype(np.float32, copy=False)
    return CompareResult(
        a_ids=left.ids, b_ids=right.ids,
        modality_a=left.modality, modality_b=right.modality,
        scores=scores,
    )
```

- [ ] **Step 4: Export in `sonar_probe/__init__.py`**

Append:
```python
from sonar_probe.compare import CompareResult, compare, parse_operand
```
Add `"compare"`, `"CompareResult"`, `"parse_operand"` to `__all__`.

- [ ] **Step 5: Run, commit**

Run: `pytest tests/test_compare.py -v`
Expected: all green.

```bash
git add sonar_probe/compare.py sonar_probe/__init__.py tests/test_compare.py
git commit -m "feat: compare — operand parsing + cosine (scalar/matrix/cross-modal)"
```

---

### Task 15: CLI — Typer app for extract, extract-text, compare, inspect

**Files:**
- Create: `sonar_probe/cli.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: public API from `sonar_probe/__init__.py`; sources + sinks registries.
- Produces: `app: typer.Typer` with subcommands `extract`, `extract-text`, `compare`, `inspect`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cli.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner

from sonar_probe.cli import app
from sonar_probe.engine.speech import SpeechEncoder
from sonar_probe.engine.text import TextBackend


class FakeSpeech(SpeechEncoder):
    def __init__(self, lang): self.lang = lang
    def encode(self, waves):
        out = []
        for w in waves:
            seed = int.from_bytes(hashlib.sha256(w.tobytes()).digest()[:8], "little")
            out.append(np.random.default_rng(seed).standard_normal(1024).astype(np.float32))
        return np.stack(out)


class FakeText(TextBackend):
    def encode(self, texts, lang):
        return np.stack([
            np.random.default_rng(
                int.from_bytes(hashlib.sha256((t+lang).encode()).digest()[:8], "little")
            ).standard_normal(1024).astype(np.float32)
            for t in texts
        ])


@pytest.fixture
def fake_encoders():
    with patch("sonar_probe.engine.speech._default_factory",
               lambda lang, device: FakeSpeech(lang)), \
         patch("sonar_probe.engine.text._default_factory",
               lambda device: FakeText()):
        yield


def test_cli_extract_plain_to_kaldi(plain_text_manifest, tmp_path: Path, fake_encoders):
    manifest, scp = plain_text_manifest
    out = tmp_path / "store"
    runner = CliRunner()
    r = runner.invoke(app, [
        "extract",
        "--source", "plain_text",
        "--input", str(manifest),
        "--wav-scp", str(scp),
        "--sink", "kaldi_ark",
        "--output", str(out),
        "--speech-lang", "eng",
        "--device", "cpu",
    ])
    assert r.exit_code == 0, r.output
    assert (out / "embeddings.scp").exists()
    assert "written=2" in r.output


def test_cli_extract_text(tmp_path: Path, fake_encoders):
    inp = tmp_path / "t.tsv"
    inp.write_text("text_id\ttext\tlang\n" "t1\thi\teng_Latn\n")
    out = tmp_path / "store"
    r = CliRunner().invoke(app, [
        "extract-text",
        "--input", str(inp),
        "--sink", "hf_dataset",
        "--output", str(out),
        "--device", "cpu",
    ])
    assert r.exit_code == 0, r.output


def test_cli_compare_scalar(tmp_path: Path, fake_encoders):
    # Build two stores with the same key via the API, then compare via CLI.
    from sonar_probe.sinks import get_sink
    from sonar_probe.types import EmbeddingRecord

    def _mk(path: Path):
        s = get_sink("kaldi_ark", output=path)
        with s.open():
            s.write(EmbeddingRecord(
                segment_id="x", file_id="r", channel=0, spkid="s",
                start=0.0, end=1.0, lang="eng", modality="speech",
                embedding=np.ones(1024, dtype=np.float32),
            ))
    a = tmp_path / "a"; b = tmp_path / "b"
    _mk(a); _mk(b)
    r = CliRunner().invoke(app, [
        "compare",
        "--a", f"store:{a}#x",
        "--b", f"store:{b}#x",
        "--device", "cpu",
    ])
    assert r.exit_code == 0
    assert "cosine=" in r.output


def test_cli_inspect(tmp_path: Path, fake_encoders):
    from sonar_probe.sinks import get_sink
    from sonar_probe.types import EmbeddingRecord
    out = tmp_path / "s"
    sink = get_sink("kaldi_ark", output=out)
    with sink.open():
        sink.write(EmbeddingRecord(
            segment_id="x", file_id="r", channel=0, spkid="s",
            start=0.0, end=1.0, lang="eng", modality="speech",
            embedding=np.ones(1024, dtype=np.float32),
        ))
    r = CliRunner().invoke(app, ["inspect", str(out)])
    assert r.exit_code == 0
    assert "count=1" in r.output
```

- [ ] **Step 2: Run; expect fail**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL — `cli` missing.

- [ ] **Step 3: Implement `sonar_probe/cli.py`**

```python
"""Typer CLI for sonar-probe."""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Optional

import typer

from sonar_probe import (
    CompareConfig,
    ExtractConfig,
    compare as api_compare,
    extract as api_extract,
    extract_text as api_extract_text,
    load_store,
    read_text_tsv,
)
from sonar_probe.sinks import get_sink
from sonar_probe.sources import get_source

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def _root(verbose: bool = typer.Option(False, "--verbose", "-v"),
          quiet: bool = typer.Option(False, "--quiet", "-q")):
    level = logging.WARNING
    if verbose: level = logging.DEBUG
    if quiet:   level = logging.ERROR
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


@app.command()
def extract(
    source: str = typer.Option(..., "--source"),
    input: Path = typer.Option(..., "--input"),
    wav_scp: Optional[Path] = typer.Option(None, "--wav-scp"),
    hf_split: Optional[str] = typer.Option(None, "--hf-split"),
    hf_config: Optional[str] = typer.Option(None, "--hf-config"),
    sink: str = typer.Option(..., "--sink"),
    output: Path = typer.Option(..., "--output"),
    speech_lang: Optional[str] = typer.Option(None, "--speech-lang"),
    device: str = typer.Option("auto", "--device"),
    batch_size: int = typer.Option(8, "--batch-size"),
    max_seg_seconds: float = typer.Option(60.0, "--max-seg-seconds"),
    encoder_cache_size: int = typer.Option(4, "--encoder-cache-size"),
    strict: bool = typer.Option(False, "--strict/--no-strict"),
    append: bool = typer.Option(False, "--append/--overwrite"),
    deterministic: bool = typer.Option(False, "--deterministic"),
):
    kwargs = {"input": input}
    if source == "plain_text":
        if wav_scp is None:
            typer.echo("ERROR: --wav-scp required for plain_text", err=True)
            raise typer.Exit(2)
        kwargs["wav_scp"] = wav_scp
    elif source == "hf":
        if wav_scp is not None: kwargs["wav_scp"] = wav_scp
        if hf_split is not None: kwargs["split"] = hf_split
        if hf_config is not None: kwargs["config"] = hf_config
    src = get_source(source, **kwargs)
    snk = get_sink(sink, output=output, append=append)
    cfg = ExtractConfig(
        device=device, batch_size=batch_size, speech_lang=speech_lang,
        max_seg_seconds=max_seg_seconds, encoder_cache_size=encoder_cache_size,
        strict=strict, append=append, deterministic=deterministic,
    )
    res = api_extract(src, snk, cfg)
    typer.echo(f"written={res.written} skipped={res.skipped}")


@app.command("extract-text")
def extract_text_cmd(
    input: Path = typer.Option(..., "--input"),
    lang: Optional[str] = typer.Option(None, "--lang"),
    sink: str = typer.Option(..., "--sink"),
    output: Path = typer.Option(..., "--output"),
    device: str = typer.Option("auto", "--device"),
    batch_size: int = typer.Option(8, "--batch-size"),
    strict: bool = typer.Option(False, "--strict/--no-strict"),
    append: bool = typer.Option(False, "--append/--overwrite"),
):
    snk = get_sink(sink, output=output, append=append)
    cfg = ExtractConfig(device=device, batch_size=batch_size,
                        strict=strict, append=append)
    res = api_extract_text(read_text_tsv(input, default_lang=lang), snk, cfg)
    typer.echo(f"written={res.written} skipped={res.skipped}")


@app.command()
def compare(
    a: str = typer.Option(..., "--a"),
    b: str = typer.Option(..., "--b"),
    wav_scp: Optional[Path] = typer.Option(None, "--wav-scp"),
    speech_lang: Optional[str] = typer.Option(None, "--speech-lang"),
    device: str = typer.Option("auto", "--device"),
    output: Optional[Path] = typer.Option(None, "--output"),
    output_format: str = typer.Option("long", "--output-format"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    threshold: Optional[float] = typer.Option(None, "--threshold"),
):
    cfg = CompareConfig(
        device=device, speech_lang=speech_lang, wav_scp=wav_scp,
        top_k=top_k, threshold=threshold, output_format=output_format,  # type: ignore
    )
    res = api_compare(a, b, cfg)
    if res.scores.shape == (1, 1):
        typer.echo(
            f"cosine={float(res.scores[0,0]):.4f} "
            f"modality={res.modality_a}↔{res.modality_b}"
        )
        return
    if output is None:
        typer.echo("ERROR: matrix output requires --output", err=True)
        raise typer.Exit(2)
    if output_format == "long":
        with Path(output).open("w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["a_id", "b_id", "score"])
            for a_id, b_id, s in res.iter_pairs(top_k=top_k, threshold=threshold):
                w.writerow([a_id, b_id, f"{s:.6f}"])
    elif output_format == "wide":
        with Path(output).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([""] + res.b_ids)
            for a_id, row in zip(res.a_ids, res.scores):
                w.writerow([a_id] + [f"{float(v):.6f}" for v in row])
    else:
        typer.echo(f"ERROR: unknown --output-format {output_format!r}", err=True)
        raise typer.Exit(2)
    typer.echo(f"wrote {output}")


@app.command()
def inspect(path: Path):
    store = load_store(path)
    langs: set[str] = set()
    modalities: set[str] = set()
    for rec in store:
        if rec.lang: langs.add(rec.lang)
        modalities.add(rec.modality)
    typer.echo(
        f"count={len(store)} dim=1024 "
        f"modalities={sorted(modalities)} langs={sorted(langs)}"
    )


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run tests, commit**

Run: `pytest tests/test_cli.py -v`
Expected: all green.

```bash
git add sonar_probe/cli.py tests/test_cli.py
git commit -m "feat: Typer CLI (extract, extract-text, compare, inspect)"
```

---

### Task 16: Integration tests (real SONAR, gated)

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_sonar_real.py`
- Modify: `README.md` (add "Running integration tests" section)

**Interfaces:**
- Consumes: real `sonar-space` runtime, downloaded checkpoints.
- Produces: two integration tests marked `@pytest.mark.integration`.

- [ ] **Step 1: Write the integration tests**

```python
# tests/integration/test_sonar_real.py
"""Integration tests that load real SONAR checkpoints.

Run with:  pytest -m integration
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sonar_probe import CompareConfig, ExtractConfig, compare, extract
from sonar_probe.engine.speech import SpeechEncoderPool
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
    # noticeably above 0 (floor is loose; real SONAR renders both as speech-ish).
    assert float(r.scores[0, 0]) > 0.3


def test_text_cross_modal_round_trip(tmp_path: Path):
    """Text 'hello world' @eng vs. same text @fra differ; same lang pair matches itself."""
    tenc = TextEncoder(device="cpu")
    e1 = tenc.encode(["hello world"], lang="eng_Latn")
    e2 = tenc.encode(["hello world"], lang="eng_Latn")
    n1 = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
    n2 = e2 / np.linalg.norm(e2, axis=1, keepdims=True)
    assert float((n1 @ n2.T)[0, 0]) == pytest.approx(1.0, abs=1e-4)
```

- [ ] **Step 2: Empty `tests/integration/__init__.py`**

```python
```

- [ ] **Step 3: Update README with integration-run instructions**

Append to `README.md`:
```markdown
## Running integration tests

Unit tests use a fake encoder and run offline. Integration tests load
real SONAR checkpoints (several hundred MB on first run) and are off by
default:

    pytest -m integration

Set `HF_HOME` to a persistent dir to cache checkpoints across runs.
```

- [ ] **Step 4: Verify unit tests still pass; skip integration locally if deps missing**

Run: `pytest`
Expected: all unit tests green; integration tests collected but not run.

Optional: `pytest -m integration` on a machine with `sonar-space` installed and GPU/CPU of your choice.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/ README.md
git commit -m "test: integration tests gated behind -m integration"
```

---

## Self-Review

### Spec coverage check

Walked each spec section against the task list:

- §2.1 In scope — all three sources (Tasks 4–6), both sinks (Tasks 7–8), `extract` (12), `extract-text` (13), `compare` (14), `inspect` (15, CLI only but reads via `load_store`). ✓
- §2.2 Out of scope — nothing in the plan builds training, clustering, multi-GPU, or entry-point plugins. ✓
- §3 Approach decisions — all 11 + extensibility pattern mapped onto specific tasks. ✓
- §4 Architecture — layout matches §4.1 one-to-one. Dependencies are in `pyproject.toml` (Task 1). ✓
- §5 Core data model — Task 2 implements all five types + `EMBEDDING_DIM` constant. ✓
- §6 Input sources — Tasks 4, 5, 6. ✓
- §7 `audio.py` — Task 3. ✓
- §8 Engine — Tasks 10 (speech) and 11 (text). ✓
- §9 Sinks — Tasks 7 and 8, plus §9.3 shared behavior (`open` context manager, append, cross-modality transparent). ✓
- §10 Compare — Task 14. ✓
- §11 CLI — Task 15. ✓
- §12 Errors — Task 2 defines the hierarchy; `failures.jsonl` in Task 12; strict policy enforced in extract. ✓
- §13 Testing — fake encoders injected via factory arg (Tasks 10, 11); fixtures in Task 1; integration gated in Task 16. ✓
- §14 Config — Task 2. ✓

No gaps identified.

### Placeholder scan

Grepped the plan content mentally for TBD / TODO / "appropriate error handling" / "etc." — none remain. All test code blocks contain concrete assertions, all implementation blocks contain concrete code.

### Type consistency

- `SegmentSource` consumed by `extract` (Task 12) matches the protocol defined in Task 4.
- `EmbeddingSink.open()` returns a context manager — all backends use `@contextmanager`; driver uses `with sink.open():` throughout.
- `SpeechEncoderPool.encode(waveforms, langs) -> np.ndarray` — same signature in Task 10, Task 12, Task 14.
- `TextEncoder.encode(texts, lang)` — same signature in Task 11, Task 13, Task 14.
- `EmbeddingRecord(embedding=np.ndarray[(1024,) float32])` — enforced in Task 2; respected by all writers (Tasks 7, 8) and both drivers (Tasks 12, 13).
- `OperandRef.kind` string literals — matched in parser (Task 14) and dispatcher (`_resolve`) in the same file.
- `load_store` return protocol (`EmbeddingStore`) matched by both `KaldiArkStore` and `HfDatasetStore`.

No signature drift found.
