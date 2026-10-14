"""CoVoST 2 config codes → NLLB-200 codes.

CoVoST 2 (``facebook/covost2``) uses ``<src>_<tgt>`` pair codes where one
side is always ``en``. We keep the xx→en configs (21 source languages
transcribed and translated to English) — the natural cross-modal demo is
speech-in-xx + translated text-in-English.
"""
from __future__ import annotations

CV_ISO_TO_NLLB: dict[str, str] = {
    "en": "eng_Latn", "fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn",
    "it": "ita_Latn", "pt": "por_Latn", "ca": "cat_Latn", "nl": "nld_Latn",
    "sv-SE": "swe_Latn",
    "ru": "rus_Cyrl", "sl": "slv_Latn", "lv": "lvs_Latn", "et": "est_Latn",
    "tr": "tur_Latn", "fa": "pes_Arab", "ar": "arb_Arab",
    "ta": "tam_Taml", "zh-CN": "zho_Hans", "ja": "jpn_Jpan", "id": "ind_Latn",
    "mn": "khk_Cyrl", "cy": "cym_Latn",
}

DEFAULT_PAIRS: tuple[str, ...] = (
    "fr_en", "de_en", "es_en", "it_en", "ru_en", "zh-CN_en",
    "ja_en", "ar_en", "tr_en", "fa_en",
)
