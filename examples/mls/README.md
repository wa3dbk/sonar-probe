# Multilingual LibriSpeech (MLS) speech-embedding recipe

Samples clips from [`facebook/multilingual_librispeech`](https://huggingface.co/datasets/facebook/multilingual_librispeech)
across 8 European languages, encodes them through SONAR speech, and
renders a UMAP (or t-SNE) scatter. Compared to FLEURS, MLS is read
audiobook speech — longer, cleaner clips with speaker metadata.

## Prereqs

```bash
pip install -e .[dev,viz]
```

MLS is **large**: a full language split is tens of GB. For a quick demo,
stick with the defaults (`--per-lang 10 --split test`) or use
`--streaming` to avoid downloading entire splits.

The SONAR speech encoder loads per language; expect a one-time ~1 GB
weight download. Set `HF_HOME` to a persistent directory.

## Run

Sampled (default — downloads the ``test`` split per language):

```bash
python examples/mls/run.py \
    --per-lang 10 \
    --split test \
    --store-out out/mls_store \
    --viz-out   out/mls_umap.html
```

Streaming (recommended for the ``train`` split or when disk is tight):

```bash
python examples/mls/run.py \
    --per-lang 25 \
    --split train \
    --streaming \
    --store-out out/mls_train_store \
    --viz-out   out/mls_train_umap.html
```

Defaults produce 80 points (8 configs × 10 clips).

| Flag | Default | Notes |
|---|---|---|
| `--langs` | all 8 MLS configs | Comma-separated MLS configs. |
| `--per-lang` | `10` | Clips per language. |
| `--split` | `test` | MLS split. `test` is small; `train` is huge — pair with `--streaming`. |
| `--streaming` | off | Stream the dataset lazily; stops after `--per-lang` items. |
| `--color-by` | `lang` | `spkid` colors by speaker for the same-language viz. |
| `--method` | `umap` | `tsne` is the alternative. |
| `--batch-size` | `8` | Lower if GPU-memory bound. |

## What to look for

- `--color-by lang`: expect per-language clusters even though all 8 langs
  share the Latin script, since SONAR encodes phonology, not just text.
- `--color-by spkid` on a single-language run: expect speaker-based
  clusters — MLS has rich per-speaker reads from LibriVox narrators.

## Language coverage

MLS covers 8 languages: English, German, Dutch, French, Italian, Spanish,
Portuguese, Polish. See `mls_langs.py` for the NLLB mapping.
