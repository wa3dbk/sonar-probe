"""Common Voice ISO 639-1 codes → SONAR / NLLB-200 language codes.

The 44 CV codes requested for this recipe map to NLLB codes below. CV codes
not covered by NLLB (``ia``, ``skr``) are intentionally absent — the recipe
skips them with a warning.
"""
from __future__ import annotations

CV_TO_NLLB: dict[str, str] = {
    "ar": "arb_Arab",     # Modern Standard Arabic
    "ba": "bak_Cyrl",     # Bashkir
    "be": "bel_Cyrl",     # Belarusian
    "bn": "ben_Beng",     # Bengali
    "ca": "cat_Latn",     # Catalan
    "cs": "ces_Latn",     # Czech
    "da": "dan_Latn",     # Danish
    "de": "deu_Latn",     # German
    "en": "eng_Latn",     # English
    "es": "spa_Latn",     # Spanish
    "et": "est_Latn",     # Estonian
    "eu": "eus_Latn",     # Basque
    "fr": "fra_Latn",     # French
    "gl": "glg_Latn",     # Galician
    "hi": "hin_Deva",     # Hindi
    "it": "ita_Latn",     # Italian
    "ka": "kat_Geor",     # Georgian
    "kmr": "kmr_Latn",    # Northern Kurdish
    "lv": "lvs_Latn",     # Standard Latvian
    "mn": "khk_Cyrl",     # Halh Mongolian
    "mr": "mar_Deva",     # Marathi
    "mt": "mlt_Latn",     # Maltese
    "nl": "nld_Latn",     # Dutch
    "pl": "pol_Latn",     # Polish
    "ro": "ron_Latn",     # Romanian
    "ru": "rus_Cyrl",     # Russian
    "rw": "kin_Latn",     # Kinyarwanda
    "sk": "slk_Latn",     # Slovak
    "sl": "slv_Latn",     # Slovenian
    "sr": "srp_Cyrl",     # Serbian
    "sw": "swh_Latn",     # Swahili
    "ta": "tam_Taml",     # Tamil
    "th": "tha_Thai",     # Thai
    "tr": "tur_Latn",     # Turkish
    "uk": "ukr_Cyrl",     # Ukrainian
    "ur": "urd_Arab",     # Urdu
    "vi": "vie_Latn",     # Vietnamese
    "yo": "yor_Latn",     # Yoruba
    "zh-CN": "zho_Hans",  # Simplified Chinese
}

# CV codes requested but not supported by NLLB-200. Skipped by the recipe.
UNSUPPORTED: frozenset[str] = frozenset({"ia", "skr"})
