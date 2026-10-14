# CoVoST 2 cross-modal recipe

Shows SONAR's defining feature: speech and text land in a single
1024-d space. For each sampled utterance from
[`facebook/covost2`](https://huggingface.co/datasets/facebook/covost2),
we embed the source-language audio AND the English translation, then
project both sets together to 2-D.

- `--color-by modality` → two overlapping clouds, confirming speech and
  text share the space.
- `--color-by pair` → each utterance's two points sit near each other,
  confirming cross-modal alignment at the sentence level.
- `--color-by lang` → clusters by source language.

## Prereqs

```bash
pip install -e .[dev,viz]
```

CoVoST 2's audio comes from Common Voice v4. The HuggingFace loader needs
you to download CV v4 yourself; it won't fetch the audio automatically.

For each source language in `--pairs`, download the CV v4 release and
extract it so the layout is:

```
/data/cv4/
  fr/
    clips/           *.mp3
    train.tsv
    test.tsv
    …
  de/…
  es/…
```

Then pass `--cv-root /data/cv4`.

Set `HF_HOME` to a persistent directory to cache the CoVoST TSVs and SONAR
weights across runs.

## Run

```bash
python examples/covost/run.py \
    --cv-root /data/cv4 \
    --per-pair 10 \
    --store-out out/covost_store \
    --viz-out   out/covost_umap.html
```

Defaults produce 200 points (10 preset `xx_en` pairs × 10 utterances × 2
modalities).

| Flag | Default | Notes |
|---|---|---|
| `--cv-root` | *(required)* | Root containing per-language CV v4 subdirs. |
| `--pairs` | 10 preset xx_en pairs | Comma-separated CoVoST pair codes. |
| `--per-pair` | `10` | Utterances per pair (each contributes 2 points). |
| `--split` | `test` | CoVoST split (`train` / `validation` / `test`). |
| `--color-by` | `modality` | `modality` / `lang` / `pair`. |
| `--batch-size` | `8` | Lower if GPU-memory bound. |

## What it exercises

- `datasets.load_dataset("facebook/covost2", <pair>, data_dir=…)`
- Both `extract()` (speech) and `extract_text()` (text) in one run
- Combining two stores before projecting via `project_embeddings`
- The shared-space property that makes `sonar-probe compare` work
  cross-modally.
