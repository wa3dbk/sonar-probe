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
