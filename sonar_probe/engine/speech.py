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
        deterministic: bool = False,
    ):
        self._device = device
        self._cache_size = cache_size
        self._factory = encoder_factory or _default_factory
        self._cache: OrderedDict[Lang, SpeechEncoder] = OrderedDict()
        if deterministic:
            try:
                import torch
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except ImportError:
                pass

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
