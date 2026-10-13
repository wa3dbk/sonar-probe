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
