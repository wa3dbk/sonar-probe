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
from sonar_probe.errors import BadSegmentError, MissingLangError
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
