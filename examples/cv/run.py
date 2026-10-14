"""Common Voice sentence-embedding recipe.

Samples sentences from each requested language's ``validated_sentences.tsv``,
encodes them through SONAR via ``sonar-probe``'s library API, saves the
embeddings to a HuggingFace-dataset store, and renders a UMAP scatter plot
colored by language.

Example:

    python examples/cv/run.py \\
        --cv-root /data/cv-corpus-24.0-2025-12-05 \\
        --per-lang 3 \\
        --store-out  out/cv_text_store \\
        --viz-out    out/cv_text_umap.html
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

from sonar_probe import (
    ExtractConfig,
    extract_text,
    load_store,
    project_store,
    read_text_tsv,
)
from sonar_probe.project import write_scatter_html
from sonar_probe.sinks import get_sink

sys.path.insert(0, str(Path(__file__).parent))
from cv_langs import CV_TO_NLLB, UNSUPPORTED  # noqa: E402


def sample_sentences(
    cv_root: Path, cv_code: str, per_lang: int, rng: random.Random
) -> list[tuple[str, str]]:
    """Return up to ``per_lang`` ``(sentence_id, sentence)`` pairs."""
    tsv = cv_root / cv_code / "validated_sentences.tsv"
    if not tsv.is_file():
        return []
    with tsv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = [
            (r["sentence_id"], r["sentence"])
            for r in reader
            if r.get("sentence_id") and r.get("sentence", "").strip()
        ]
    if len(rows) <= per_lang:
        return rows
    return rng.sample(rows, per_lang)


def build_sonar_tsv(
    cv_root: Path, langs: list[str], per_lang: int, out_tsv: Path, seed: int
) -> tuple[int, list[str]]:
    rng = random.Random(seed)
    skipped: list[str] = []
    written = 0
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["text_id", "text", "lang"])
        for cv_code in langs:
            if cv_code in UNSUPPORTED:
                skipped.append(f"{cv_code} (not in NLLB-200)")
                continue
            if cv_code not in CV_TO_NLLB:
                skipped.append(f"{cv_code} (no NLLB mapping configured)")
                continue
            nllb = CV_TO_NLLB[cv_code]
            picks = sample_sentences(cv_root, cv_code, per_lang, rng)
            if not picks:
                skipped.append(f"{cv_code} (no validated_sentences.tsv found)")
                continue
            for sid, text in picks:
                writer.writerow([f"{cv_code}:{sid[:16]}", text, nllb])
                written += 1
    return written, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cv-root", type=Path, required=True,
        help="Path to the Common Voice release root (e.g. cv-corpus-24.0-2025-12-05).",
    )
    parser.add_argument(
        "--langs", default=",".join(sorted(CV_TO_NLLB.keys()) + sorted(UNSUPPORTED)),
        help="Comma-separated CV lang codes to sample. Default: all 44 preset langs.",
    )
    parser.add_argument(
        "--per-lang", type=int, default=3, help="Sentences to sample per language.",
    )
    parser.add_argument(
        "--store-out", type=Path, default=Path("out/cv_text_store"),
        help="HuggingFace-dataset store output directory.",
    )
    parser.add_argument(
        "--viz-out", type=Path, default=Path("out/cv_text_umap.html"),
        help="Plotly scatter HTML output path.",
    )
    parser.add_argument(
        "--method", choices=["umap", "tsne"], default="umap",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    langs = [c.strip() for c in args.langs.split(",") if c.strip()]
    scratch = args.store_out.parent / "cv_scratch.tsv"
    print(f"[1/4] sampling {args.per_lang} sentences/lang from {args.cv_root}")
    written, skipped = build_sonar_tsv(
        args.cv_root, langs, args.per_lang, scratch, args.seed
    )
    if skipped:
        print(f"      skipped: {', '.join(skipped)}")
    if written == 0:
        print("      no sentences sampled — check --cv-root path.", file=sys.stderr)
        return 2
    print(f"      wrote {written} rows → {scratch}")

    print(f"[2/4] encoding through SONAR text (batch_size={args.batch_size})")
    sink = get_sink("hf_dataset", output=args.store_out)
    cfg = ExtractConfig(device=args.device, batch_size=args.batch_size)
    res = extract_text(read_text_tsv(scratch), sink, cfg)
    print(f"      written={res.written} skipped={res.skipped}")

    print(f"[3/4] projecting ({args.method}) → 2-D")
    store = load_store(args.store_out)
    ids, coords = project_store(
        store, method=args.method, random_state=args.seed,
    )

    print(f"[4/4] rendering scatter → {args.viz_out}")
    labels = [rec.lang or "(none)" for rec in store]
    id_to_label = dict(zip(store.ids(), labels))
    write_scatter_html(
        ids,
        coords,
        labels=[id_to_label[i] for i in ids],
        title=f"Common Voice sentences · SONAR text · {args.method.upper()}",
        output=args.viz_out,
    )
    print(f"done: open {args.viz_out} in a browser")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
