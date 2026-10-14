"""Operand resolution + cosine compare."""
from __future__ import annotations

from dataclasses import dataclass
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
        msg = "inline operand needs lang (6th field) or --speech-lang"
        raise MissingLangError(msg)
    scp = parse_wav_scp(Path(cfg.wav_scp))
    spec = scp.get(file_id)
    if spec is None:
        raise BadSpecError(f"file_id {file_id!r} not in wav.scp")
    w = load_waveform(AudioRef(path=Path(spec)), int(channel),
                      float(start), float(end))
    emb = pool.encode([w], [lang])
    seg_id = (f"{file_id}-{channel}-{spkid}-"
              f"{int(round(float(start)*1000))}-{int(round(float(end)*1000))}")
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
    return OperandVectors(ids=[f"text:0:{text[:16]}"], matrix=emb, modality="text")


def _resolve_text_file(
    path: str, lang: Lang, text_encoder: TextEncoder,
) -> OperandVectors:
    lines = [ln.rstrip("\n") for ln in Path(path).read_text().splitlines()
             if ln.strip()]
    emb = text_encoder.encode(lines, lang=lang)
    return OperandVectors(ids=[f"text:{i}:{t[:16]}" for i, t in enumerate(lines)],
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
