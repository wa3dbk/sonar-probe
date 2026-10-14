"""FLEURS config codes → SONAR / NLLB-200 language codes.

FLEURS uses ``<iso>_<country>`` config codes (``google/fleurs``). The mapping
below covers ~40 configs spanning diverse scripts and language families.
``DEFAULT_LANGS`` is a 25-config subset tuned for download time while
preserving family / script diversity.
"""
from __future__ import annotations

FLEURS_TO_NLLB: dict[str, str] = {
    # Indo-European — Romance
    "fr_fr": "fra_Latn", "it_it": "ita_Latn", "es_419": "spa_Latn",
    "pt_br": "por_Latn", "ro_ro": "ron_Latn", "ca_es": "cat_Latn",
    # Indo-European — Germanic
    "en_us": "eng_Latn", "de_de": "deu_Latn", "nl_nl": "nld_Latn",
    "sv_se": "swe_Latn", "da_dk": "dan_Latn", "nb_no": "nob_Latn",
    # Indo-European — Slavic
    "ru_ru": "rus_Cyrl", "uk_ua": "ukr_Cyrl", "pl_pl": "pol_Latn",
    "cs_cz": "ces_Latn", "sk_sk": "slk_Latn", "sr_rs": "srp_Cyrl",
    # Indo-European — Hellenic / Celtic / Baltic
    "el_gr": "ell_Grek", "lv_lv": "lvs_Latn", "lt_lt": "lit_Latn",
    # Indo-Iranian
    "hi_in": "hin_Deva", "bn_in": "ben_Beng", "ur_pk": "urd_Arab",
    "fa_ir": "pes_Arab", "mr_in": "mar_Deva", "pa_in": "pan_Guru",
    # Afro-Asiatic
    "ar_eg": "arb_Arab", "he_il": "heb_Hebr", "mt_mt": "mlt_Latn",
    "ha_ng": "hau_Latn",
    # Turkic / Uralic
    "tr_tr": "tur_Latn", "fi_fi": "fin_Latn", "hu_hu": "hun_Latn",
    # Dravidian
    "ta_in": "tam_Taml", "te_in": "tel_Telu", "ml_in": "mal_Mlym",
    "kn_in": "kan_Knda",
    # Austronesian
    "id_id": "ind_Latn", "ms_my": "zsm_Latn", "fil_ph": "tgl_Latn",
    # Austroasiatic / Tai-Kadai / Sino-Tibetan / Japonic / Koreanic
    "vi_vn": "vie_Latn", "th_th": "tha_Thai", "km_kh": "khm_Khmr",
    "lo_la": "lao_Laoo", "my_mm": "mya_Mymr",
    "cmn_hans_cn": "zho_Hans", "yue_hant_hk": "yue_Hant",
    "ja_jp": "jpn_Jpan", "ko_kr": "kor_Hang",
    # Niger-Congo / other African
    "sw_ke": "swh_Latn", "yo_ng": "yor_Latn", "ig_ng": "ibo_Latn",
    "zu_za": "zul_Latn", "xh_za": "xho_Latn",
}

DEFAULT_LANGS: tuple[str, ...] = (
    "en_us", "fr_fr", "de_de", "es_419", "it_it", "pt_br",
    "ru_ru", "uk_ua", "cs_cz", "pl_pl", "nl_nl", "el_gr",
    "tr_tr", "ar_eg", "fa_ir", "hi_in", "bn_in", "ta_in",
    "th_th", "vi_vn", "ja_jp", "ko_kr", "cmn_hans_cn",
    "sw_ke", "yo_ng",
)
