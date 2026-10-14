"""MLS config names → NLLB-200 codes.

Multilingual LibriSpeech (``facebook/multilingual_librispeech``) exposes
one HF config per language, named with the English language name.
"""
from __future__ import annotations

MLS_TO_NLLB: dict[str, str] = {
    "english": "eng_Latn",
    "german": "deu_Latn",
    "dutch": "nld_Latn",
    "french": "fra_Latn",
    "italian": "ita_Latn",
    "spanish": "spa_Latn",
    "portuguese": "por_Latn",
    "polish": "pol_Latn",
}

DEFAULT_LANGS: tuple[str, ...] = tuple(MLS_TO_NLLB.keys())
