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
