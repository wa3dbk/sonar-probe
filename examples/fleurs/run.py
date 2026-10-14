"""FLEURS speech-embedding recipe.

Downloads a slice of ``google/fleurs`` across many languages, samples
``--per-lang`` clips from each, encodes them through SONAR speech, saves to
a Kaldi ark/scp store, and renders a UMAP scatter colored by language.

Example:

    python examples/fleurs/run.py \\
        --per-lang 3 \\
        --split test \\
        --store-out out/fleurs_store \\
        --viz-out   out/fleurs_umap.html
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import soundfile as sf
from datasets import load_dataset

from sonar_probe import ExtractConfig, extract, load_store, project_store
from sonar_probe.project import write_scatter_html
from sonar_probe.sinks import get_sink
from sonar_probe.sources import get_source

sys.path.insert(0, str(Path(__file__).parent))
from fleurs_langs import DEFAULT_LANGS, FLEURS_TO_NLLB  # noqa: E402


def materialize(
    cfgs: list[str], per_lang: int, split: str, scratch: Path, seed: int
) -> tuple[Path, Path, list[str]]:
    """Download clips for each FLEURS config, write a wav.scp + segments.tsv.

    Returns ``(segments_tsv, wav_scp, skipped_messages)``.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    clip_dir = scratch / "clips"
    clip_dir.mkdir(exist_ok=True)
    scp_path = scratch / "wav.scp"
    seg_path = scratch / "segments.tsv"

    rng = random.Random(seed)
    skipped: list[str] = []
    with scp_path.open("w", encoding="utf-8") as scp, \
         seg_path.open("w", encoding="utf-8", newline="") as seg_fh:
        seg = csv.writer(seg_fh, delimiter="\t")
        seg.writerow(["file_id", "channel", "spkid", "start", "end", "lang"])

        for cfg in cfgs:
            if cfg not in FLEURS_TO_NLLB:
                skipped.append(f"{cfg} (no NLLB mapping)")
                continue
            nllb = FLEURS_TO_NLLB[cfg]
            try:
                ds = load_dataset("google/fleurs", cfg, split=split)
            except Exception as e:
                skipped.append(f"{cfg} ({type(e).__name__}: {e})")
                continue
            picks = rng.sample(range(len(ds)), min(per_lang, len(ds)))
            for idx in picks:
                row = ds[idx]
                audio = row["audio"]
                clip_path = Path(audio.get("path") or "")
                if not clip_path.is_file():
                    clip_path = clip_dir / f"{cfg}_{row['id']}.wav"
                    sf.write(clip_path, audio["array"], audio["sampling_rate"])
                dur = row["num_samples"] / audio["sampling_rate"]
                file_id = f"{cfg}_{row['id']}"
                scp.write(f"{file_id} {clip_path}\n")
                seg.writerow([file_id, 0, cfg, 0.0, f"{dur:.4f}", nllb])
    return seg_path, scp_path, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--langs", default=",".join(DEFAULT_LANGS),
        help=f"Comma-separated FLEURS configs. Default: {len(DEFAULT_LANGS)} "
             f"preset langs. Pass 'all' to use the full mapping.",
    )
    parser.add_argument("--per-lang", type=int, default=3)
    parser.add_argument(
        "--split", default="test",
        help="FLEURS split (train | validation | test). Smaller is faster.",
    )
    parser.add_argument(
        "--store-out", type=Path, default=Path("out/fleurs_store"),
    )
    parser.add_argument(
        "--viz-out", type=Path, default=Path("out/fleurs_umap.html"),
    )
    parser.add_argument("--method", choices=["umap", "tsne"], default="umap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--scratch", type=Path, default=Path("out/fleurs_scratch"),
    )
    args = parser.parse_args()

    if args.langs.strip() == "all":
        cfgs = list(FLEURS_TO_NLLB.keys())
    else:
        cfgs = [c.strip() for c in args.langs.split(",") if c.strip()]

    print(f"[1/4] materializing {args.per_lang} clips/lang from FLEURS "
          f"({len(cfgs)} configs, split={args.split})")
    seg, scp, skipped = materialize(
        cfgs, args.per_lang, args.split, args.scratch, args.seed
    )
    if skipped:
        print(f"      skipped: {', '.join(skipped)}")

    print(f"[2/4] encoding through SONAR speech (batch_size={args.batch_size})")
    src = get_source("plain_text", input=seg, wav_scp=scp)
    sink = get_sink("kaldi_ark", output=args.store_out)
    cfg = ExtractConfig(device=args.device, batch_size=args.batch_size)
    res = extract(src, sink, cfg)
    print(f"      written={res.written} skipped={res.skipped}")
    if res.written == 0:
        print("      nothing to project; aborting.", file=sys.stderr)
        return 2

    print(f"[3/4] projecting ({args.method}) → 2-D")
    store = load_store(args.store_out)
    ids, coords = project_store(
        store, method=args.method, random_state=args.seed,
    )

    print(f"[4/4] rendering scatter → {args.viz_out}")
    id_to_label = {rec.segment_id: (rec.lang or "(none)") for rec in store}
    write_scatter_html(
        ids,
        coords,
        labels=[id_to_label[i] for i in ids],
        title=f"FLEURS speech · SONAR · {args.method.upper()}",
        output=args.viz_out,
    )
    print(f"done: open {args.viz_out} in a browser")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
