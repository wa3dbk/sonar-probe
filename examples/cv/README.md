# Common Voice sentence-embedding recipe

Samples sentences from a local
[Common Voice](https://commonvoice.mozilla.org/en/datasets) release across
~40 languages, encodes them through SONAR text, saves to a
HuggingFace-dataset store, and renders a UMAP (or t-SNE) scatter plot
colored by language. Expect clear per-script clusters that collapse when
viewed by language family.

## Prereqs

```bash
pip install -e .[dev,viz]
```

A local Common Voice release laid out as:

```
cv-corpus-24.0-2025-12-05/
  ar/validated_sentences.tsv
  de/validated_sentences.tsv
  en/validated_sentences.tsv
  …
```

The first run downloads SONAR text weights via HuggingFace Hub (~1 GB). Set
`HF_HOME` to a persistent directory to cache across runs.

## Run

```bash
python examples/cv/run.py \
    --cv-root  /path/to/cv-corpus-24.0-2025-12-05 \
    --per-lang 3 \
    --store-out out/cv_text_store \
    --viz-out   out/cv_text_umap.html
```

With the defaults (44 CV codes × 3 sentences) you get ~126 points. Each
step prints progress; the final line names the HTML file to open.

Flags worth knowing:

| Flag | Default | Notes |
|---|---|---|
| `--langs` | all 44 preset codes | Comma-separated CV codes to sample. |
| `--per-lang` | `3` | Sentences per language (~`N_LANGS × per_lang` total). |
| `--method` | `umap` | `tsne` is the alternative. |
| `--batch-size` | `16` | Lower if you hit GPU memory limits. |
| `--device` | `auto` | `cuda` picks GPU; `cpu` forces CPU (slow). |
| `--seed` | `0` | Controls both the sentence sample and the projection. |

## Language coverage

`cv_langs.py` maps 39 CV codes to NLLB-200 codes understood by SONAR. The
two CV codes requested but not supported by NLLB — `ia` (Interlingua) and
`skr` (Saraiki) — are skipped with a logged warning.

To add or override a mapping, edit `CV_TO_NLLB` in `cv_langs.py`.

## Extending to speech

Common Voice also ships `clips/<sentence_id>.mp3` per language. To embed
audio instead of text:

1. Resolve each sampled `sentence_id` to a clip path and write a sonar-probe
   `wav.scp` + segments TSV using CV's `clip_duration` for `end`.
2. Swap `extract_text(...)` for `extract(get_source("plain_text", …), …)`
   with the `kaldi_ark` sink.
3. Keep the projection / scatter step unchanged — speech and text embeddings
   share the same 1024-d SONAR space, so you can even plot them together.
