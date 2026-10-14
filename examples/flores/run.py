"""FLORES-200 text-embedding recipe.

Samples ``--per-lang`` sentences from ``facebook/flores`` for each requested
NLLB-200 code, encodes them through SONAR text, saves to a
HuggingFace-dataset store, and renders a UMAP (or t-SNE) scatter colored
by language.

Example:

    python examples/flores/run.py \\
        --per-lang 5 \\
        --store-out out/flores_store \\
        --viz-out   out/flores_umap.html
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

from datasets import load_dataset

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
from flores_langs import DEFAULT_LANGS  # noqa: E402


def build_tsv(
    langs: list[str], per_lang: int, split: str, out_tsv: Path, seed: int
) -> tuple[int, list[str]]:
    rng = random.Random(seed)
    skipped: list[str] = []
    written = 0
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["text_id", "text", "lang"])
        for code in langs:
            try:
                ds = load_dataset("facebook/flores", code, split=split)
            except Exception as e:
                skipped.append(f"{code} ({type(e).__name__}: {e})")
                continue
            picks = rng.sample(range(len(ds)), min(per_lang, len(ds)))
            for idx in picks:
                row = ds[idx]
                text_id = f"{code}:{row['id']}"
                writer.writerow([text_id, row["sentence"], code])
                written += 1
    return written, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--langs", default=",".join(DEFAULT_LANGS),
        help=f"Comma-separated NLLB codes. Default: {len(DEFAULT_LANGS)} preset codes.",
    )
    parser.add_argument("--per-lang", type=int, default=5)
    parser.add_argument(
        "--split", default="devtest",
        help="FLORES-200 split (dev | devtest).",
    )
    parser.add_argument("--store-out", type=Path, default=Path("out/flores_store"))
    parser.add_argument("--viz-out", type=Path, default=Path("out/flores_umap.html"))
    parser.add_argument("--method", choices=["umap", "tsne"], default="umap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    langs = [c.strip() for c in args.langs.split(",") if c.strip()]
    scratch = args.store_out.parent / "flores_scratch.tsv"
    print(f"[1/4] sampling {args.per_lang} sentences/lang from FLORES-200 "
          f"({len(langs)} codes, split={args.split})")
    written, skipped = build_tsv(
        langs, args.per_lang, args.split, scratch, args.seed
    )
    if skipped:
        print(f"      skipped: {', '.join(skipped)}")
    if written == 0:
        print("      nothing sampled; aborting.", file=sys.stderr)
        return 2
    print(f"      wrote {written} rows → {scratch}")

    print(f"[2/4] encoding through SONAR text (batch_size={args.batch_size})")
    sink = get_sink("hf_dataset", output=args.store_out)
    cfg = ExtractConfig(device=args.device, batch_size=args.batch_size)
    res = extract_text(read_text_tsv(scratch), sink, cfg)
    print(f"      written={res.written} skipped={res.skipped}")

    print(f"[3/4] projecting ({args.method}) → 2-D")
    store = load_store(args.store_out)
    ids, coords = project_store(store, method=args.method, random_state=args.seed)

    print(f"[4/4] rendering scatter → {args.viz_out}")
    id_to_label = {rec.segment_id: (rec.lang or "(none)") for rec in store}
    write_scatter_html(
        ids,
        coords,
        labels=[id_to_label[i] for i in ids],
        title=f"FLORES-200 · SONAR text · {args.method.upper()}",
        output=args.viz_out,
    )
    print(f"done: open {args.viz_out} in a browser")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
