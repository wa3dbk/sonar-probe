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
