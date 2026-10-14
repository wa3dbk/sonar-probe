"""Preset FLORES-200 language codes.

FLORES-200 configs are NLLB-200 codes directly (e.g. ``eng_Latn``), so no
remapping is needed — SONAR consumes them as-is. ``DEFAULT_LANGS`` is a
40-code sample spanning diverse scripts and families.
"""
from __future__ import annotations

DEFAULT_LANGS: tuple[str, ...] = (
    # Romance
    "fra_Latn", "ita_Latn", "spa_Latn", "por_Latn", "ron_Latn", "cat_Latn",
    # Germanic
    "eng_Latn", "deu_Latn", "nld_Latn", "swe_Latn", "dan_Latn", "nob_Latn",
    # Slavic
    "rus_Cyrl", "ukr_Cyrl", "pol_Latn", "ces_Latn", "srp_Cyrl",
    # Hellenic / Baltic
    "ell_Grek", "lit_Latn",
    # Indo-Iranian
    "hin_Deva", "ben_Beng", "urd_Arab", "pes_Arab", "mar_Deva",
    # Afro-Asiatic
    "arb_Arab", "heb_Hebr", "mlt_Latn",
    # Turkic / Uralic
    "tur_Latn", "fin_Latn", "hun_Latn",
    # Dravidian
    "tam_Taml", "tel_Telu", "mal_Mlym",
    # SE / E Asian
    "vie_Latn", "tha_Thai", "khm_Khmr", "mya_Mymr",
    "zho_Hans", "jpn_Jpan", "kor_Hang",
    # Niger-Congo
    "swh_Latn", "yor_Latn", "zul_Latn",
)
