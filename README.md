# sonar-probe

Extract and compare [SONAR](https://github.com/facebookresearch/SONAR) speech
and text embeddings over Kaldi datasets, HuggingFace datasets, and plain-text
segment manifests — from the command line or as a library.

- **1024-d shared speech/text embedding space** (SONAR) — compare speech to
  speech, text to text, or speech to text with the same cosine metric.
- **Three input sources** that all yield the same `Segment` type:
  `plain_text` (TSV + `wav.scp`), `kaldi` (standard data dir), `hf`
  (HuggingFace dataset, with or without an audio column).
- **Two durable output stores**: `kaldi_ark` (ark/scp pair + JSONL metadata
  sidecar) and `hf_dataset` (loadable via `datasets.load_from_disk`).
- **Compare module** with a small operand grammar: paths, segment IDs, inline
  segments, literal text, text files. Returns an `(N, M)` cosine matrix with
  optional `top_k` / `threshold` filtering.
- **Pluggable protocols** (sources, sinks, encoders) with injectable factories
  so tests and research scripts can replace SONAR with any backend that
  produces `(B, 1024) float32` vectors.

## Install

Not yet on PyPI — install from a local clone:

```bash
git clone https://github.com/wa3dbk/sonar-probe.git
cd sonar-probe
pip install -e .[dev]            # add ,viz for UMAP / t-SNE / plotly
```

Runtime dependencies include `torch`, `torchaudio`, `kaldiio`, `datasets`,
`sonar-space`, and `fairseq2`. Linux with CUDA is the smoothest path;
`sonar-space` and `fairseq2` wheels are limited on other platforms.

First speech or text extraction downloads the SONAR checkpoint via
HuggingFace Hub. Set `HF_HOME` to a persistent directory to cache across runs.

> **TODO** — publish to PyPI so this reduces to `pip install sonar-probe`.

## Quick start — CLI

Extract embeddings from a plain-text segment manifest + `wav.scp`:

```bash
sonar-probe extract \
  --source plain_text \
  --input segments.tsv \
  --wav-scp wav.scp \
  --sink kaldi_ark \
  --output out/ \
  --speech-lang eng
```

Compare two stored segments:

```bash
sonar-probe compare \
  --a store:out#spk1-5000-7000 \
  --b store:out#spk2-5000-7000
# cosine=0.6412 modality=speech↔speech
```

Compare a stored speech segment to a text query (cross-modal):

```bash
sonar-probe compare \
  --a store:out#spk1-5000-7000 \
  --b "text:a quick brown fox@eng"
```

Inspect any store:

```bash
sonar-probe inspect out/
# count=42 dim=1024 modalities=['speech'] langs=['eng']
```

Project a store to 2-D and render an interactive HTML scatter (requires the
`viz` extra — see **Visualization** below):

```bash
sonar-probe visualize out/ \
  --method umap \
  --color-by lang \
  --output viz.html
```

## Quick start — library

```python
from sonar_probe import ExtractConfig, extract, load_store, compare, CompareConfig
from sonar_probe.sources import get_source
from sonar_probe.sinks import get_sink

src = get_source("plain_text", input="segments.tsv", wav_scp="wav.scp")
snk = get_sink("kaldi_ark", output="out/")
cfg = ExtractConfig(device="auto", batch_size=16, speech_lang="eng")

result = extract(src, snk, cfg)
print(f"wrote {result.written} segments, skipped {result.skipped}")

# Later: load the store and compare segments.
store = load_store("out/")
ids, mat = store.matrix()
print(ids, mat.shape)  # (N, 1024)

res = compare("store:out#id-a", "store:out#id-b", CompareConfig(device="cpu"))
print(float(res.scores[0, 0]))
```

See the [`tutorials/`](tutorials/) directory for runnable notebooks covering
every feature end-to-end, and [`examples/`](examples/) for runnable recipes:

- [`examples/cv/`](examples/cv/) — sample Common Voice sentences across
  ~40 languages, embed them through SONAR text, render a UMAP scatter.
- [`examples/fleurs/`](examples/fleurs/) — sample FLEURS clips across ~25
  languages, embed them through SONAR speech, render a UMAP scatter.

## Input sources

All three sources yield `Segment(file_id, channel, spkid, start, end, lang?,
audio_ref)` records.

### `plain_text` — TSV + `wav.scp`

A tab-separated manifest with columns `file_id`, `channel`, `spkid`, `start`,
`end` (optional `lang`), paired with a Kaldi-style `wav.scp` that maps
`file_id → path-or-pipe`.

```
# segments.tsv
file_id	channel	spkid	start	end	lang
rec01	0	spk1	0.00	2.50	eng
rec01	0	spk1	3.00	5.50	eng
rec02	1	spk2	0.00	4.00	fra
```

```
# wav.scp
rec01 /data/rec01.wav
rec02 sox /data/rec02.flac -t wav - |
```

Channel labels `A`/`B`/`L`/`R` are accepted and coerced; numeric channels work
too. Pipe entries (trailing `|`) are decoded via subprocess.

### `kaldi` — Kaldi data directory

Point at a directory containing standard files: `wav.scp`, `segments`,
`utt2spk`, optional `reco2file_and_channel`, optional `utt2lang`. Each utt_id
becomes the segment's `segment_id`.

### `hf` — HuggingFace dataset

Loads via `datasets.load_from_disk` (if `--input` is a directory) or
`datasets.load_dataset` (otherwise). Required columns: `file_id`, `channel`,
`spkid`, `start`, `end` (optional `lang`). If the dataset has an `audio`
column it is used directly; otherwise pass `--wav-scp` to resolve `file_id`
to a path.

## Output stores

| Sink          | Files written                                          | Reader                            |
| ------------- | ------------------------------------------------------ | --------------------------------- |
| `kaldi_ark`   | `embeddings.ark`, `embeddings.scp`, `metadata.jsonl`   | `kaldiio` + metadata JSONL sidecar|
| `hf_dataset`  | HuggingFace dataset on disk (`save_to_disk` layout)    | `datasets.load_from_disk`         |

`load_store(path)` sniffs the layout and returns the right store object.
Stores expose `__iter__`, `__getitem__`, `__len__`, `ids()`, and
`matrix(ids=None) -> (list[id], (N, 1024) float32)`.

## Compare — operand grammar

The `--a` and `--b` arguments to `sonar-probe compare` (and the first two
positional args to `compare()`) accept five operand kinds:

| Operand                                     | Meaning                                              |
| ------------------------------------------- | ---------------------------------------------------- |
| `path/to/store`                             | all embeddings in a store (yields an (N,) matrix)    |
| `store:path#segment_id`                     | one embedding from a store                           |
| `file_id,channel,spkid,start,end[,lang]`    | inline segment, decoded + encoded on the fly         |
| `text:<text>@<lang>`                        | literal text, encoded on the fly                     |
| `text-file:<path>@<lang>`                   | one text per line, encoded as a batch                |

The result is a `CompareResult(a_ids, b_ids, modality_a, modality_b, scores)`
with `scores.shape == (len(a_ids), len(b_ids))`. The CLI prints a scalar for
1×1 results and writes long (TSV) or wide (CSV) matrix output otherwise.

## Configuration

### `ExtractConfig`

| Field                 | Default | Meaning                                               |
| --------------------- | ------- | ----------------------------------------------------- |
| `device`              | `auto`  | `auto` picks `cuda` if available, else `cpu`          |
| `batch_size`          | `8`     | segments per encoder call                             |
| `speech_lang`         | `None`  | default source language for untagged segments         |
| `max_seg_seconds`     | `60.0`  | segments longer than this are skipped (or raise in strict mode) |
| `encoder_cache_size`  | `4`     | LRU size of per-language speech encoders              |
| `strict`              | `False` | abort on any soft error instead of logging to `failures.jsonl` |
| `append`              | `False` | append to an existing store instead of overwriting    |
| `deterministic`       | `False` | sets cuDNN deterministic flags when `True`            |

### `CompareConfig`

| Field          | Default   | Meaning                                               |
| -------------- | --------- | ----------------------------------------------------- |
| `device`       | `auto`    |                                                       |
| `speech_lang`  | `None`    | default lang for inline segment operands              |
| `wav_scp`      | `None`    | required for inline segment operands                  |
| `top_k`        | `None`    | keep best-k scores per row of the matrix              |
| `threshold`    | `None`    | drop scores below this                                |
| `output_format`| `long`    | `long` emits TSV of (a_id, b_id, score); `wide` emits a CSV matrix |

## Visualization

Install the optional `viz` extra to project embeddings to 2-D (UMAP or t-SNE)
and render interactive plotly scatter plots:

```bash
pip install -e .[viz]
```

Library:

```python
from sonar_probe import load_store, project_store
from sonar_probe.project import write_scatter_html

store = load_store("out/")
ids, coords = project_store(store, method="umap", n_neighbors=15)
labels = [store[i].lang for i in ids]
write_scatter_html(ids, coords, labels=labels, output="viz.html")
```

CLI:

```bash
sonar-probe visualize out/ \
  --method tsne --perplexity 30 \
  --color-by spkid \
  --output viz.html
```

`--color-by` accepts `modality`, `lang`, `spkid`, or `none`. For UMAP, pass
`--n-neighbors`; for t-SNE, `--perplexity`. Both methods take `--random-state`
for reproducibility.

## Failure handling

Soft errors (bad segments, decode failures, encode errors in `extract_text`)
are written line-by-line to `<output>/failures.jsonl` with the segment or
text id and the error, and the run continues. `MissingLangError` is always
fatal. Pass `--strict` to make every soft error fatal.

## Testing

Unit tests use fake encoder factories and run offline:

```bash
pytest
```

Integration tests load the real SONAR checkpoint and are gated:

```bash
pytest -m integration
```

Set `HF_HOME` to a persistent directory to avoid redownloading the model.

## Project layout

```
sonar_probe/
  types.py         Segment, AudioRef, EmbeddingRecord, TextItem
  config.py        ExtractConfig, CompareConfig, resolve_device
  errors.py        exception hierarchy
  audio.py         wav.scp parsing + waveform decode/resample
  text_input.py    TSV reader for text extraction
  sources/         SegmentSource protocol + plain_text, kaldi, hf
  sinks/           EmbeddingSink/Store protocols + kaldi_ark, hf_dataset
  engine/          SpeechEncoderPool (LRU per-lang), TextEncoder
  driver.py        extract() and extract_text()
  compare.py       operand parsing + cosine compare
  project.py       UMAP/t-SNE projection + plotly scatter (viz extra)
  cli.py           Typer app (extract, extract-text, compare, inspect, visualize)
```

## Citing SONAR

If you use embeddings produced by this tool in a publication, please cite the
upstream SONAR paper: see
[facebookresearch/SONAR](https://github.com/facebookresearch/SONAR) for the
current citation.

## License

See `LICENSE` for terms. SONAR model weights are governed by their own
license; consult the upstream repository.
