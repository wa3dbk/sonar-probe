"""OPUS-100 lang-code → NLLB-200 mapping.

OPUS-100 uses ISO-639-1 codes in pair names like ``en-fr``. SONAR speaks
NLLB-200, so we remap each ISO code to its NLLB counterpart.
"""
from __future__ import annotations

ISO_TO_NLLB: dict[str, str] = {
    "en": "eng_Latn", "fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn",
    "it": "ita_Latn", "pt": "por_Latn", "ro": "ron_Latn", "ca": "cat_Latn",
    "nl": "nld_Latn", "sv": "swe_Latn", "da": "dan_Latn", "nb": "nob_Latn",
    "ru": "rus_Cyrl", "uk": "ukr_Cyrl", "pl": "pol_Latn", "cs": "ces_Latn",
    "sk": "slk_Latn", "sr": "srp_Cyrl", "bg": "bul_Cyrl",
    "el": "ell_Grek", "lt": "lit_Latn", "lv": "lvs_Latn",
    "hi": "hin_Deva", "bn": "ben_Beng", "ur": "urd_Arab", "fa": "pes_Arab",
    "ar": "arb_Arab", "he": "heb_Hebr",
    "tr": "tur_Latn", "fi": "fin_Latn", "hu": "hun_Latn",
    "ta": "tam_Taml", "te": "tel_Telu", "ml": "mal_Mlym",
    "id": "ind_Latn", "ms": "zsm_Latn",
    "vi": "vie_Latn", "th": "tha_Thai",
    "zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "sw": "swh_Latn", "zu": "zul_Latn",
}

DEFAULT_PAIRS: tuple[str, ...] = (
    "en-fr", "en-de", "en-es", "en-it", "en-ru", "en-zh",
    "en-ja", "en-ar", "en-hi", "en-tr", "en-ko", "en-vi",
)
