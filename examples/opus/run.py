"""OPUS-100 parallel-sentence embedding recipe.

Samples ``--per-pair`` sentence pairs from each requested OPUS-100 config,
embeds each half through SONAR text, and renders a scatter where you can
toggle legend groups to see translations land near each other in the shared
SONAR space. Each pair contributes TWO points to the plot.

Example:

    python examples/opus/run.py \\
        --per-pair 25 \\
        --store-out out/opus_store \\
        --viz-out   out/opus_umap.html
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
from opus_langs import DEFAULT_PAIRS, ISO_TO_NLLB  # noqa: E402


def build_tsv(
    pairs: list[str], per_pair: int, split: str, out_tsv: Path, seed: int,
    color_by: str,
) -> tuple[int, list[str]]:
    """Write a TSV of ``text_id, text, lang`` + ``color`` for the scatter.

    ``color_by`` ∈ {"lang", "pair"} controls what label the scatter groups by.
    """
    rng = random.Random(seed)
    skipped: list[str] = []
    written = 0
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["text_id", "text", "lang", "color"])
        for pair in pairs:
            try:
                a, b = pair.split("-")
            except ValueError:
                skipped.append(f"{pair} (bad pair spec; want 'xx-yy')")
                continue
            if a not in ISO_TO_NLLB or b not in ISO_TO_NLLB:
                skipped.append(f"{pair} (no NLLB mapping for one half)")
                continue
            try:
                ds = load_dataset("Helsinki-NLP/opus-100", pair, split=split)
            except Exception as e:
                skipped.append(f"{pair} ({type(e).__name__}: {e})")
                continue
            picks = rng.sample(range(len(ds)), min(per_pair, len(ds)))
            for idx in picks:
                tr = ds[idx]["translation"]
                sa, sb = tr.get(a, "").strip(), tr.get(b, "").strip()
                if not sa or not sb:
                    continue
                pair_id = f"{pair}:{idx}"
                for iso, text in ((a, sa), (b, sb)):
                    nllb = ISO_TO_NLLB[iso]
                    color = nllb if color_by == "lang" else pair
                    writer.writerow([f"{pair_id}:{iso}", text, nllb, color])
                    written += 1
    return written, skipped


def _color_map(scratch_tsv: Path) -> dict[str, str]:
    with scratch_tsv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return {row["text_id"]: row["color"] for row in reader}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pairs", default=",".join(DEFAULT_PAIRS),
        help=f"Comma-separated OPUS-100 pair codes. Default: {len(DEFAULT_PAIRS)} "
             "preset pairs. Each pair must appear as 'xx-yy' with en on one side.",
    )
    parser.add_argument("--per-pair", type=int, default=25)
    parser.add_argument(
        "--split", default="test",
        help="OPUS-100 split (train | validation | test). test is ~2k rows.",
    )
    parser.add_argument(
        "--color-by", choices=["lang", "pair"], default="lang",
        help="Color points by language (default) or by source pair.",
    )
    parser.add_argument("--store-out", type=Path, default=Path("out/opus_store"))
    parser.add_argument("--viz-out", type=Path, default=Path("out/opus_umap.html"))
    parser.add_argument("--method", choices=["umap", "tsne"], default="umap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    scratch = args.store_out.parent / "opus_scratch.tsv"
    print(f"[1/4] sampling {args.per_pair} pairs each from OPUS-100 "
          f"({len(pairs)} configs, split={args.split})")
    written, skipped = build_tsv(
        pairs, args.per_pair, args.split, scratch, args.seed, args.color_by,
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
    color_map = _color_map(scratch)
    write_scatter_html(
        ids,
        coords,
        labels=[color_map.get(i, "(none)") for i in ids],
        title=f"OPUS-100 parallel sentences · SONAR text · {args.method.upper()}",
        output=args.viz_out,
    )
    print(f"done: open {args.viz_out} in a browser "
          f"(try `--color-by pair` to see translation clusters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
