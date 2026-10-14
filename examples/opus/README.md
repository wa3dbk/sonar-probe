# OPUS-100 parallel-sentence recipe

Samples sentence pairs from [`Helsinki-NLP/opus-100`](https://huggingface.co/datasets/Helsinki-NLP/opus-100)
across many English-pivoted language pairs, embeds BOTH halves of each
pair through SONAR text, and renders a scatter you can color either by
language (shows per-script clusters) or by pair (shows that translations
land near each other in the shared SONAR space).

## Prereqs

```bash
pip install -e .[dev,viz]
```

First encode downloads SONAR text weights (~1 GB). Set `HF_HOME` to a
persistent directory.

## Run

```bash
python examples/opus/run.py \
    --per-pair 25 \
    --store-out out/opus_store \
    --viz-out   out/opus_umap.html
```

Defaults produce 600 points (12 preset pairs × 25 sentences × 2 halves).

| Flag | Default | Notes |
|---|---|---|
| `--pairs` | 12 preset `en-*` pairs | Comma-separated OPUS-100 pair codes. |
| `--per-pair` | `25` | Sentence pairs per config (each contributes 2 points). |
| `--split` | `test` | OPUS-100 split (`test` is small; `train` is huge). |
| `--color-by` | `lang` | `lang` colors by NLLB code; `pair` colors by source pair. |
| `--method` | `umap` | `tsne` is the alternative. |
| `--batch-size` | `16` | Lower if GPU-memory bound. |

## What to look for

- `--color-by lang`: expect per-script clusters (Latin / Cyrillic / CJK
  separate cleanly), confirming SONAR groups by language.
- `--color-by pair`: expect each pair to form a tight 50-point cloud of
  translations — this is the signature of SONAR being a shared,
  language-agnostic sentence embedder.

## Language coverage

See `opus_langs.py`. `ISO_TO_NLLB` maps ~42 OPUS-100 ISO codes to NLLB.
`DEFAULT_PAIRS` is a 12-pair sample pivoting through English across diverse
scripts.
