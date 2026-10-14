"""Multilingual LibriSpeech (MLS) speech-embedding recipe.

Samples ``--per-lang`` clips from each MLS language config, encodes them
through SONAR speech, saves to a Kaldi ark/scp store, and renders a UMAP
scatter colored by language.

MLS is large; by default we sample from the ``test`` split (smaller) and
cap per-language clips to keep the run short.

Example:

    python examples/mls/run.py \\
        --per-lang 10 \\
        --store-out out/mls_store \\
        --viz-out   out/mls_umap.html
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
from mls_langs import DEFAULT_LANGS, MLS_TO_NLLB  # noqa: E402


def materialize(
    cfgs: list[str], per_lang: int, split: str, scratch: Path, seed: int,
    streaming: bool,
) -> tuple[Path, Path, list[str]]:
    """Download clips for each MLS config, write wav.scp + segments.tsv.

    When ``streaming`` is True we iterate the dataset lazily and stop after
    ``per_lang`` items per language — useful for the huge train split.
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
            if cfg not in MLS_TO_NLLB:
                skipped.append(f"{cfg} (no NLLB mapping)")
                continue
            nllb = MLS_TO_NLLB[cfg]
            try:
                ds = load_dataset(
                    "facebook/multilingual_librispeech", cfg,
                    split=split, streaming=streaming,
                )
            except Exception as e:
                skipped.append(f"{cfg} ({type(e).__name__}: {e})")
                continue

            if streaming:
                rows = []
                for i, row in enumerate(ds):
                    rows.append(row)
                    if i + 1 >= per_lang:
                        break
            else:
                picks = rng.sample(range(len(ds)), min(per_lang, len(ds)))
                rows = [ds[i] for i in picks]

            for row in rows:
                audio = row["audio"]
                clip_path = Path(audio.get("path") or "")
                if not clip_path.is_file():
                    uid = row.get("id") or row.get("audio_id") or hex(id(row))
                    clip_path = clip_dir / f"{cfg}_{uid}.wav"
                    sf.write(clip_path, audio["array"], audio["sampling_rate"])
                dur = len(audio["array"]) / audio["sampling_rate"]
                file_id = f"{cfg}_{row.get('id') or row.get('audio_id')}"
                spkid = str(row.get("speaker_id", cfg))
                scp.write(f"{file_id} {clip_path}\n")
                seg.writerow([file_id, 0, spkid, 0.0, f"{dur:.4f}", nllb])
    return seg_path, scp_path, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--langs", default=",".join(DEFAULT_LANGS),
        help=f"Comma-separated MLS configs. Default: all {len(DEFAULT_LANGS)}.",
    )
    parser.add_argument("--per-lang", type=int, default=10)
    parser.add_argument(
        "--split", default="test",
        help="MLS split (train | validation | test). test is smallest.",
    )
    parser.add_argument(
        "--streaming", action="store_true",
        help="Stream the dataset instead of downloading. Use for train split.",
    )
    parser.add_argument("--store-out", type=Path, default=Path("out/mls_store"))
    parser.add_argument("--viz-out", type=Path, default=Path("out/mls_umap.html"))
    parser.add_argument("--method", choices=["umap", "tsne"], default="umap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--scratch", type=Path, default=Path("out/mls_scratch"),
    )
    parser.add_argument(
        "--color-by", choices=["lang", "spkid"], default="lang",
    )
    args = parser.parse_args()

    cfgs = [c.strip() for c in args.langs.split(",") if c.strip()]
    print(f"[1/4] materializing {args.per_lang} clips/lang from MLS "
          f"({len(cfgs)} configs, split={args.split}, streaming={args.streaming})")
    seg, scp, skipped = materialize(
        cfgs, args.per_lang, args.split, args.scratch, args.seed, args.streaming,
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
    ids, coords = project_store(store, method=args.method, random_state=args.seed)

    print(f"[4/4] rendering scatter → {args.viz_out}")
    pick = (lambda r: r.lang or "(none)") if args.color_by == "lang" \
        else (lambda r: r.spkid or "(none)")
    id_to_label = {rec.segment_id: pick(rec) for rec in store}
    write_scatter_html(
        ids,
        coords,
        labels=[id_to_label[i] for i in ids],
        title=f"MLS speech · SONAR · {args.method.upper()}",
        output=args.viz_out,
    )
    print(f"done: open {args.viz_out} in a browser")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
