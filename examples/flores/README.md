# FLORES-200 text-embedding recipe

Samples sentences from [`facebook/flores`](https://huggingface.co/datasets/facebook/flores)
across many languages, encodes them through SONAR text, and renders a UMAP
(or t-SNE) scatter colored by language. FLORES-200's config codes are the
same NLLB-200 codes SONAR consumes, so no remapping is needed.

## Prereqs

```bash
pip install -e .[dev,viz]
```

- First run downloads a small FLORES slice (~a few MB per language — much
  lighter than FLEURS).
- First encode downloads SONAR text weights (~1 GB). Set `HF_HOME` to a
  persistent directory to cache across runs.

## Run

```bash
python examples/flores/run.py \
    --per-lang 5 \
    --split devtest \
    --store-out out/flores_store \
    --viz-out   out/flores_umap.html
```

Defaults produce 200 points (40 preset codes × 5 sentences).

| Flag | Default | Notes |
|---|---|---|
| `--langs` | 40 preset NLLB codes | Comma-separated NLLB codes (e.g. `eng_Latn,fra_Latn,…`). |
| `--per-lang` | `5` | Sentences per language. |
| `--split` | `devtest` | FLORES-200 split (`dev` has 997, `devtest` has 1012). |
| `--method` | `umap` | `tsne` is the alternative. |
| `--batch-size` | `16` | Lower if GPU-memory bound. |
| `--device` | `auto` | `cpu` is viable for text (slower but works). |

## What it exercises

- `datasets.load_dataset("facebook/flores", <nllb-code>, split=…)`
- sonar-probe's `read_text_tsv` + `extract_text` + `hf_dataset` sink
- `project_store()` → `write_scatter_html()`
- Scale (200 langs available) — try `--langs <all 200>` for a dense picture.

## Why FLORES vs Common Voice vs FLEURS

| Dataset | Modality | Granularity | Download | Lang coverage |
|---|---|---|---|---|
| FLORES-200 | text | sentences (clean, pro-translated) | tiny | 200 |
| Common Voice | text/speech | crowdsourced sentences + clips | large | ~100 |
| FLEURS | speech | read speech | medium | ~100 |

FLORES is the right pick for a fast, uniform text-only cross-lingual demo.
