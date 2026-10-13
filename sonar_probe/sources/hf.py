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
