"""CoVoST 2 cross-modal recipe.

For each requested CoVoST 2 ``xx_en`` config, samples ``--per-pair`` rows,
encodes the audio through SONAR speech AND the English translation through
SONAR text, then projects both sets together to 2-D so you can see that
speech-in-xx and text-in-English land near each other in the shared SONAR
space.

CoVoST 2's audio comes from Common Voice v4. You must download and extract
the CV v4 TSV + clips tarball for each source language yourself, and pass
``--cv-root`` pointing at the root that contains per-language subdirs with
``clips/`` and ``train.tsv`` / ``test.tsv``. See the HuggingFace dataset
card for the exact layout.

Example:

    python examples/covost/run.py \\
        --cv-root /data/cv4 \\
        --per-pair 10 \\
        --store-out out/covost_store \\
        --viz-out   out/covost_umap.html
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset

from sonar_probe import (
    ExtractConfig,
    extract,
    extract_text,
    load_store,
    read_text_tsv,
)
from sonar_probe.project import project_embeddings, write_scatter_html
from sonar_probe.sinks import get_sink
from sonar_probe.sources import get_source

sys.path.insert(0, str(Path(__file__).parent))
from covost_langs import CV_ISO_TO_NLLB, DEFAULT_PAIRS  # noqa: E402


def materialize(
    pairs: list[str], per_pair: int, split: str, scratch: Path, cv_root: Path,
    seed: int,
) -> tuple[Path, Path, Path, dict[str, str], list[str]]:
    """Build wav.scp + segments.tsv (speech) and a text TSV (translations).

    Returns ``(segments_tsv, wav_scp, text_tsv, utt_to_pair, skipped)``.
    ``utt_to_pair`` maps every emitted id (both speech and text) to its
    ``<pair>:<row_id>`` key so the scatter can color by pair-of-origin.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    clip_dir = scratch / "clips"
    clip_dir.mkdir(exist_ok=True)
    scp_path = scratch / "wav.scp"
    seg_path = scratch / "segments.tsv"
    txt_path = scratch / "translations.tsv"

    rng = random.Random(seed)
    utt_to_pair: dict[str, str] = {}
    skipped: list[str] = []

    with scp_path.open("w", encoding="utf-8") as scp, \
         seg_path.open("w", encoding="utf-8", newline="") as seg_fh, \
         txt_path.open("w", encoding="utf-8", newline="") as txt_fh:
        seg = csv.writer(seg_fh, delimiter="\t")
        seg.writerow(["file_id", "channel", "spkid", "start", "end", "lang"])
        txt = csv.writer(txt_fh, delimiter="\t")
        txt.writerow(["text_id", "text", "lang"])

        for pair in pairs:
            try:
                src_iso, tgt_iso = pair.split("_", 1)
            except ValueError:
                skipped.append(f"{pair} (bad pair spec; want 'xx_en')")
                continue
            if src_iso not in CV_ISO_TO_NLLB or tgt_iso not in CV_ISO_TO_NLLB:
                skipped.append(f"{pair} (no NLLB mapping)")
                continue
            src_nllb = CV_ISO_TO_NLLB[src_iso]
            tgt_nllb = CV_ISO_TO_NLLB[tgt_iso]
            cv_lang_root = cv_root / src_iso
            if not cv_lang_root.is_dir():
                skipped.append(f"{pair} (no {cv_lang_root} — download CV v4)")
                continue
            try:
                ds = load_dataset(
                    "facebook/covost2", pair,
                    data_dir=str(cv_lang_root),
                    split=split,
                )
            except Exception as e:
                skipped.append(f"{pair} ({type(e).__name__}: {e})")
                continue
            picks = rng.sample(range(len(ds)), min(per_pair, len(ds)))
            for idx in picks:
                row = ds[idx]
                row_id = f"{pair}:{idx}"
                audio = row["audio"]
                clip_path = Path(audio.get("path") or "")
                if not clip_path.is_file():
                    clip_path = clip_dir / f"{src_iso}_{idx}.wav"
                    sf.write(clip_path, audio["array"], audio["sampling_rate"])
                dur = len(audio["array"]) / audio["sampling_rate"]
                sp_id = f"sp:{row_id}"
                tx_id = f"tx:{row_id}"
                scp.write(f"{sp_id} {clip_path}\n")
                seg.writerow([sp_id, 0, src_iso, 0.0, f"{dur:.4f}", src_nllb])
                txt.writerow([tx_id, row["translation"], tgt_nllb])
                utt_to_pair[sp_id] = row_id
                utt_to_pair[tx_id] = row_id
    return seg_path, scp_path, txt_path, utt_to_pair, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cv-root", type=Path, required=True,
        help="Common Voice v4 root (per-language subdirs with clips/).",
    )
    parser.add_argument(
        "--pairs", default=",".join(DEFAULT_PAIRS),
        help=f"Comma-separated CoVoST 2 pair codes. Default: {len(DEFAULT_PAIRS)} "
             "preset xx_en pairs.",
    )
    parser.add_argument("--per-pair", type=int, default=10)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--speech-store", type=Path, default=Path("out/covost_speech"),
    )
    parser.add_argument(
        "--text-store", type=Path, default=Path("out/covost_text"),
    )
    parser.add_argument("--store-out", type=Path, default=Path("out/covost_store"))
    parser.add_argument("--viz-out", type=Path, default=Path("out/covost_umap.html"))
    parser.add_argument("--method", choices=["umap", "tsne"], default="umap")
    parser.add_argument(
        "--color-by", choices=["modality", "lang", "pair"], default="modality",
        help="modality: speech vs text; lang: NLLB code; pair: utterance-of-origin.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--scratch", type=Path, default=Path("out/covost_scratch"),
    )
    args = parser.parse_args()

    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    print(f"[1/5] materializing {args.per_pair} rows/pair from CoVoST 2 "
          f"({len(pairs)} configs, split={args.split})")
    seg, scp, txt, utt_to_pair, skipped = materialize(
        pairs, args.per_pair, args.split, args.scratch, args.cv_root, args.seed,
    )
    if skipped:
        print(f"      skipped: {', '.join(skipped)}")
    if not utt_to_pair:
        print("      nothing materialized; aborting.", file=sys.stderr)
        return 2

    print(f"[2/5] encoding speech → {args.speech_store}")
    src = get_source("plain_text", input=seg, wav_scp=scp)
    speech_sink = get_sink("kaldi_ark", output=args.speech_store)
    cfg = ExtractConfig(device=args.device, batch_size=args.batch_size)
    r1 = extract(src, speech_sink, cfg)
    print(f"      written={r1.written} skipped={r1.skipped}")

    print(f"[3/5] encoding text → {args.text_store}")
    text_sink = get_sink("hf_dataset", output=args.text_store)
    r2 = extract_text(read_text_tsv(txt), text_sink, cfg)
    print(f"      written={r2.written} skipped={r2.skipped}")

    print(f"[4/5] combining stores and projecting ({args.method}) → 2-D")
    sp_store = load_store(args.speech_store)
    tx_store = load_store(args.text_store)
    sp_ids, sp_mat = sp_store.matrix()
    tx_ids, tx_mat = tx_store.matrix()
    ids = list(sp_ids) + list(tx_ids)
    mat = np.vstack([sp_mat, tx_mat])
    coords = project_embeddings(mat, method=args.method, random_state=args.seed)

    print(f"[5/5] rendering scatter → {args.viz_out} (color_by={args.color_by})")
    if args.color_by == "modality":
        id_to_label = {i: "speech" for i in sp_ids}
        id_to_label.update({i: "text" for i in tx_ids})
    elif args.color_by == "pair":
        id_to_label = {i: utt_to_pair.get(i, "(none)") for i in ids}
    else:  # lang
        id_to_label = {rec.segment_id: (rec.lang or "(none)") for rec in sp_store}
        id_to_label.update(
            {rec.segment_id: (rec.lang or "(none)") for rec in tx_store}
        )
    write_scatter_html(
        ids,
        coords,
        labels=[id_to_label[i] for i in ids],
        title=f"CoVoST 2 cross-modal · SONAR · {args.method.upper()}",
        output=args.viz_out,
    )
    print(f"done: open {args.viz_out}; try `--color-by pair` to see "
          f"same-utterance speech↔text clusters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
