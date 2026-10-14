# FLEURS speech-embedding recipe

Downloads a slice of [`google/fleurs`](https://huggingface.co/datasets/google/fleurs)
across many languages, encodes `--per-lang` clips per config through SONAR
speech, and renders a UMAP (or t-SNE) scatter colored by language. Clean
counterpart to the Common Voice *text* recipe — same visualization, other
modality.

## Prereqs

```bash
pip install -e .[dev,viz]
```

- First run per language config downloads the FLEURS slice (hundreds of MB
  to a few GB per config). Set `HF_HOME` to a persistent directory.
- The SONAR speech encoder is loaded per language on first use; expect a
  one-time weight download of ~1 GB.

## Run

```bash
python examples/fleurs/run.py \
    --per-lang 3 \
    --split test \
    --store-out out/fleurs_store \
    --viz-out   out/fleurs_umap.html
```

Defaults produce 75 points (25 preset configs × 3 clips).

| Flag | Default | Notes |
|---|---|---|
| `--langs` | 25 preset FLEURS configs | `all` → the full ~50-config map; or a comma list. |
| `--per-lang` | `3` | Clips per language. |
| `--split` | `test` | FLEURS split (train / validation / test). Smaller = faster. |
| `--method` | `umap` | `tsne` is the alternative. |
| `--batch-size` | `8` | SpeechEncoderPool batch; lower if GPU-memory bound. |
| `--device` | `auto` | `cpu` works but is slow for the speech encoder. |
| `--scratch` | `out/fleurs_scratch` | wav.scp + segments.tsv + any re-encoded clips land here. |

## What it exercises

- `datasets.load_dataset("google/fleurs", <cfg>, split=…)`
- sonar-probe's `plain_text` source + `kaldi_ark` sink
- `extract()` → `project_store()` → `write_scatter_html()`
- Real SONAR speech encoder loading / per-language LRU cache
  (`SpeechEncoderPool`)

## Language coverage

See `fleurs_langs.py`. `FLEURS_TO_NLLB` maps ~50 FLEURS configs to NLLB-200
codes; `DEFAULT_LANGS` is a 25-config subset that spans Romance, Germanic,
Slavic, Hellenic, Indo-Iranian, Afro-Asiatic, Dravidian, Tai-Kadai,
Austroasiatic, Japonic, Koreanic, Sinitic, Niger-Congo. Edit either to add
or remove configs.
