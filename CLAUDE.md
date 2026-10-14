# CLAUDE.md

Guide for developers (human and AI) working inside this repo. Keep this file
short; link out to deeper docs rather than duplicating them.

## What this is

`sonar-probe` is a library + Typer CLI for extracting and comparing SONAR
speech/text embeddings. The library is the primary surface — the CLI is a
thin Typer wrapper over public API functions. See `README.md` for the user
pitch, install, and feature walkthrough.

## Repo tour

```
sonar_probe/
  types.py         Segment, AudioRef, EmbeddingRecord, TextItem, EMBEDDING_DIM
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
tests/
  conftest.py      shared fixtures (synthetic WAVs, mini Kaldi dir, HF dataset)
  integration/     real-SONAR tests gated by `@pytest.mark.integration`
tutorials/         runnable Jupyter notebooks for library + CLI
docs/              design specs
```

## Design invariants

- **Everything speaks `Segment` and `EmbeddingRecord`.** Sources yield
  `Segment`; sinks consume `EmbeddingRecord`; stores yield `EmbeddingRecord`.
  Shape of embeddings is always `(1024,) float32` — enforced by the
  `EmbeddingRecord` validator (see `EMBEDDING_DIM` in `types.py`).
- **Protocols are pluggable.** `SegmentSource`, `EmbeddingSink`,
  `EmbeddingStore`, `SpeechEncoder`, and `TextBackend` are all `Protocol`s.
  To add a new source/sink: implement the protocol, decorate with
  `@register_source` / `@register_sink`, and import your module in the
  package `__init__.py` so the registry picks it up.
- **Encoders are injectable.** `SpeechEncoderPool(encoder_factory=...)` and
  `TextEncoder(encoder_factory=...)` take a factory so tests can swap in
  fakes. Every unit test does this; the real SONAR backends are only loaded
  in integration tests.
- **Lang handling is strict.** `MissingLangError` is always fatal in
  `extract()` even without `--strict`. Soft errors (bad segments, decode or
  encode failures) go to `<output>/failures.jsonl` unless `--strict`.
- **Append semantics differ by sink.** `hf_dataset` supports `--append`
  cleanly; `kaldi_ark` currently truncates the ark on append (known
  limitation — see the ledger entries noted below).

## Working on the code

### Install (dev + viz)

```bash
pip install -e .[dev,viz]
```

Linux + CUDA is the smoothest path. `sonar-space` and `fairseq2` wheels are
limited on macOS. Unit tests do **not** require those packages to be
importable at runtime — the fake encoder factories bypass them.

### Tests

```bash
pytest                     # unit tests, all offline
pytest -m integration      # pulls real SONAR checkpoints — set HF_HOME to cache
```

The default `addopts` in `pyproject.toml` deselects integration tests. The
CI-equivalent command is just `pytest`.

### Lint & types

```bash
ruff check .
pyright
```

`ruff` is configured with `line-length = 100` and `target-version = "py310"`.

## Adding features

- **New source** (e.g. a new dataset format): subclass nothing — just
  implement `__iter__ -> Iterator[Segment]` and `__len__`, decorate with
  `@register_source`, declare `name: ClassVar[str]`, and import the module
  in `sonar_probe/sources/__init__.py`. Mirror `kaldi.py` for structure.
- **New sink**: implement `open() -> ContextManager`, `write(rec)`, `close()`,
  decorate with `@register_sink`. If the sink is also readable, pair it with
  a store class implementing the `EmbeddingStore` protocol and have
  `load_store()` sniff its layout.
- **New CLI subcommand**: add a `@app.command()` function in `cli.py` that
  calls the library API. Keep argument validation at the CLI layer; the
  library should accept already-validated kwargs.

## Pointers

- Design spec: `docs/superpowers/specs/2026-10-14-sonar-probe-design.md`
- Implementation plan: `docs/superpowers/plans/2026-10-14-sonar-probe.md`
- Known limitations (parked rulings): `.superpowers/sdd/2026-10-14-sonar-probe/progress.md`
  (git-ignored; present only in the original worktree)
