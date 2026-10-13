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
