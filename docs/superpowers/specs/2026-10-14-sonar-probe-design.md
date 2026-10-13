# Sonar Probe — Design Spec

**Date:** 2026-10-14
**Status:** Approved for implementation planning
**Owner:** elaloui.ahlem@gmail.com

## 1. Purpose

`sonar-probe` is a command-line and Python-library tool for extracting
and comparing speech and text embeddings from Meta's SONAR model. It
accepts three input conventions (Kaldi data directories, HuggingFace
datasets, and plain-text manifests), writes embeddings to disk in two
on-disk formats (Kaldi `ark+scp` pair, or HuggingFace
`datasets.Dataset`), and offers a `compare` subcommand that returns
cosine similarity for either a single pair of segments or an N×M
matrix across sets of segments. SONAR's speech and text encoders share
a 1024-d embedding space, so compare operands may mix modalities.

The tool is library-first: the CLI (`cli.py`) is a thin Typer wrapper
around a stable public Python API.

## 2. Scope

### 2.1 In scope

- Three input source backends: `plain_text`, `kaldi`, `hf`.
- Two output sink backends: `kaldi_ark`, `hf_dataset`.
- `extract` subcommand: segments → speech embeddings → sink.
- `extract-text` subcommand: text lines → text embeddings → sink.
- `compare` subcommand: single-pair cosine or pairwise matrix, with
  cross-modal (speech ↔ text) support.
- `inspect` subcommand: summarize contents of a saved embedding store.
- Per-language SONAR speech encoder routing with an LRU pool.
- Silent audio normalization to 16 kHz mono (expected by SONAR).
- Typed error hierarchy with configurable `strict` vs. skip-and-log
  policies, and a `failures.jsonl` sidecar.

### 2.2 Explicitly out of scope (for v1)

- Fine-tuning or training of SONAR encoders.
- Audio format conversion beyond what `torchaudio`/`soundfile` handles
  natively.
- Clustering, diarization, or ANN search over stored embeddings
  (users bring their own tools; the Kaldi/HF outputs are standard
  enough to feed any downstream pipeline).
- Multi-GPU inference, distributed extraction, or multi-process
  producer-consumer audio I/O. `--num-workers` is a plausible future
  addition but not required for v1.
- Third-party backend plugins via `importlib.metadata` entry points.
- `push_to_hub` automation (users can call it on the saved HF
  dataset themselves).

## 3. Approach decisions

The following decisions were made during brainstorming and are
load-bearing for the design that follows.

| # | Decision | Chosen |
|---|---|---|
| 1 | SONAR backend | **Speech + text encoders** (shared 1024-d space) |
| 2 | Audio access | **Per-format conventions**: Kaldi uses `wav.scp`; HF uses `audio` column or `wav.scp` fallback; plain text requires `--wav-scp` |
| 3 | Channel semantics | **Integer index** (0-based) into resolved multichannel audio |
| 4 | Embedding storage | **Kaldi `ark,scp`** + metadata sidecar, OR **HF `save_to_disk`** |
| 5 | CLI shape | **Library-first**, Typer-based CLI with subcommands |
| 6 | Compare input | **Both** single-pair and pairwise-matrix, auto-dispatched by operand shape |
| 7 | Cross-modal | **Mixed compare + `extract-text` subcommand** — text and speech symmetric |
| 8 | Speech language | **Run-level `--speech-lang` default + per-segment override column** |
| 9 | Audio preprocessing | **Silent auto-resample** to 16 kHz mono |
| 10 | Device & batching | **Auto-detect CUDA / CPU**, batch size 8, both overridable; no MPS auto |
| 11 | Error policy | **Default skip + log**, `--strict` flag for fail-fast |
| A | Extensibility | **Protocol-based registry** for sources and sinks (not flat dispatch, not entry-point plugins) |

## 4. Architecture

### 4.1 Module layout

```
sonar_probe/
├── __init__.py              # public API: extract, extract_text, compare, load_store
├── types.py                 # Segment, EmbeddingRecord, AudioRef, SegmentID, Lang
├── config.py                # ExtractConfig, CompareConfig, device/batch defaults
├── sources/
│   ├── __init__.py          # registry: get_source(name) -> SegmentSource
│   ├── base.py              # SegmentSource protocol + shared helpers
│   ├── plain_text.py        # TSV/CSV + wav.scp
│   ├── kaldi.py             # segments + wav.scp + utt2spk (+ reco2file_and_channel)
│   └── hf.py                # datasets.Dataset / load_dataset path
├── sinks/
│   ├── __init__.py          # registry: get_sink(name) -> EmbeddingSink
│   ├── base.py
│   ├── kaldi_ark.py         # kaldiio ark,scp writer/reader
│   └── hf_dataset.py        # datasets.Dataset save_to_disk / load_from_disk
├── audio.py                 # file resolution, channel extraction, resample to 16k mono
├── engine/
│   ├── __init__.py
│   ├── speech.py            # SONAR speech encoder pool (keyed by lang, LRU)
│   └── text.py              # SONAR text encoder (single multilingual checkpoint)
├── compare.py               # single-pair + pairwise matrix, cross-modal dispatch
├── errors.py                # typed exception hierarchy
└── cli.py                   # Typer app: extract, extract-text, compare, inspect
tests/
docs/
pyproject.toml
```

### 4.2 Dependencies

Runtime:

- `sonar-space` and `fairseq2` — SONAR model inference pipelines.
- `torch`, `torchaudio` — audio I/O, resampling, GPU inference.
- `kaldiio` — pure-Python `ark/scp` reader+writer (no Kaldi binary required).
- `datasets` — HuggingFace dataset read/write.
- `soundfile` — fallback decoder for formats `torchaudio` doesn't cover.
- `typer` + `rich` — CLI and progress bars.
- `numpy`, `pydantic` — data model and validation.

Dev: `pytest`, `pytest-cov`, `ruff`, `pyright`.

### 4.3 Extension seams

Two protocols are the only extension points for v1:

- `SegmentSource` (in `sources/base.py`) — any iterable of `Segment`.
- `EmbeddingSink` + `EmbeddingStore` (in `sinks/base.py`) — write and
  read vectors keyed by `segment_id`.

Backends register themselves on import via a decorator; `get_source`
/ `get_sink` return instances by string name. The core driver (`extract`)
depends only on the protocols.

## 5. Core data model

Defined in `types.py` using Pydantic for validation.

```python
SegmentID = str   # canonical: f"{file_id}-{channel}-{spkid}-{start_ms}-{end_ms}"
Lang      = str   # SONAR codes; speech uses ISO-3 ("eng"), text uses BCP-47 ("eng_Latn")

class AudioRef(BaseModel):
    path:   Path | None = None
    bytes_: bytes | None = None
    sr:     int | None = None   # original sample rate if known

class Segment(BaseModel):
    file_id:   str
    channel:   int                # 0-based index into resolved audio file
    spkid:     str
    start:     float              # seconds, inclusive
    end:       float              # seconds, exclusive
    lang:      Lang | None = None # overrides run-level --speech-lang
    audio_ref: AudioRef | None = None  # populated by the source backend

    @property
    def segment_id(self) -> SegmentID: ...  # deterministic composite ID
    @property
    def duration(self) -> float: ...

class EmbeddingRecord(BaseModel):
    segment_id: SegmentID
    file_id:    str
    channel:    int
    spkid:      str
    start:      float
    end:        float
    lang:       Lang | None
    modality:   Literal["speech", "text"]
    embedding:  NDArray[np.float32]   # shape (1024,)
```

### 5.1 Invariants

- `end > start`; `channel >= 0`;
  `duration <= max_seg_seconds` (configurable, default 60.0; SONAR
  speech encoder quality degrades on very long inputs).
- `segment_id` is deterministic from the input tuple, used as the
  primary key in both sinks, so repeat runs are idempotent (same
  input ⇒ same key ⇒ overwrites).
- `embedding` is always float32. Vectors are **not** L2-normalized at
  write time; `compare` normalizes on the fly so stored vectors
  remain usable for other downstream math (e.g. computing norms,
  mean-pooling across segments).

### 5.2 Text extraction counterpart

For `extract-text`, input items are `TextItem(text_id, text, lang)`
and flow through the same `EmbeddingRecord` type with
`modality="text"`. Sinks are modality-agnostic.

## 6. Input sources

All implement:

```python
class SegmentSource(Protocol):
    name: ClassVar[str]
    def __iter__(self) -> Iterator[Segment]: ...   # yields with audio_ref populated
    def __len__(self) -> int | None: ...           # best-effort for progress bars
```

### 6.1 `plain_text`

- TSV input (header required): `file_id, channel, spkid, start, end[, lang]`.
- `--wav-scp PATH` (required) maps `file_id → audio spec` in Kaldi
  `wav.scp` syntax, including pipe entries (`sox ... |`). Pipe
  decoding is deferred to `audio.py`.
- Timestamps are seconds (float). Validation enforces invariants.
- Yields `Segment` with `audio_ref = AudioRef(path=...)`.

### 6.2 `kaldi`

- `--input DIR` points at a Kaldi data directory containing:
  - `segments` — `utt_id file_id start end`
  - `wav.scp` — `file_id <audio-spec>`
  - `utt2spk` — `utt_id spkid`
  - optional `reco2file_and_channel` — `file_id file channel_label`;
    labels `A` → 0, `B` → 1; numeric labels kept as integers.
  - optional `utt2lang` — per-segment language override.
- Yields `Segment` where `segment_id` is overridden to the native
  Kaldi `utt_id` so round-tripping preserves user keys.
- When `reco2file_and_channel` is absent and audio is mono, `channel`
  defaults to 0.

### 6.3 `hf`

- `--input NAME_OR_PATH` plus optional `--hf-split`, `--hf-config`.
  Accepts a Hub name, a local `save_to_disk` directory, or a
  `datasets.Dataset` passed through the Python API.
- Required columns: `file_id, channel, spkid, start, end`; optional
  `lang`.
- Audio comes from either:
  1. an `audio` column (HF `Audio` feature) → `AudioRef(bytes_, sr)`
     without touching disk; or
  2. a `--wav-scp` passed alongside, for datasets that carry only
     metadata.
- The source selects the mode based on column presence.

### 6.4 Shared helpers (`sources/base.py`)

- `parse_wav_scp(path) -> dict[str, str]` — defers pipe-entry
  decoding to `audio.py`.
- `coerce_channel(raw) -> int` — label → index.
- `validate_segment(seg) -> None` — enforces invariants; raises
  `BadSegmentError`.

## 7. Audio I/O (`audio.py`)

Single responsibility: given an `AudioRef`, a channel index, and a
(start, end) window, return a 16 kHz mono `np.float32` waveform.

- Supports filesystem paths, in-memory bytes (HF audio column), and
  Kaldi-style pipe entries (`sox ... |`, `ffmpeg ... |`).
- Decoding prefers `torchaudio`; falls back to `soundfile`; pipe
  entries go through `subprocess.Popen` with a bounded stdout buffer.
- Channel extraction: slice the requested channel index from the
  multi-channel array before resampling.
- Resample to 16 kHz is silent (per decision 9). Logged only at DEBUG.
- Out-of-bounds timestamps or missing channels raise
  `BadSegmentError`.

Isolating audio I/O here keeps `sources/*` free of decoding details
and keeps the engine free of filesystem concerns.

## 8. SONAR engine

Only module that imports `sonar.inference_pipelines.*`.

### 8.1 `engine/speech.py` — SpeechEncoderPool

```python
class SpeechEncoderPool:
    def __init__(self, device: str, dtype: torch.dtype = torch.float32,
                 cache_size: int = 4): ...
    def encode(self, waveforms: list[np.ndarray], langs: list[Lang]
               ) -> np.ndarray:  # (B, 1024) float32 on CPU
```

- LRU cache of `{lang -> SpeechToEmbeddingModelPipeline}`.
- Encoders load lazily on first use of a language; eviction frees
  VRAM via `torch.cuda.empty_cache()`.
- Inputs are 16 kHz mono float32 (guaranteed by `audio.py`).
- Internally groups the batch by language, runs each sub-batch
  through its encoder, and scatters results back into original order.

### 8.2 `engine/text.py` — TextEncoder

```python
class TextEncoder:
    def __init__(self, device: str, dtype: torch.dtype = torch.float32):
        self._pipe = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=device, dtype=dtype,
        )
    def encode(self, texts: list[str], lang: Lang) -> np.ndarray:  # (B, 1024)
```

- Single multilingual checkpoint — no pool needed.
- `lang` is required (full BCP-47 code).

### 8.3 Driver loop

Lives in the public `extract()` function:

```python
def extract(source: SegmentSource, sink: EmbeddingSink, cfg: ExtractConfig):
    pool = SpeechEncoderPool(cfg.device, cache_size=cfg.encoder_cache_size)
    with sink.open():
        for batch in _batched(source, cfg.batch_size):
            waves, langs, metas = [], [], []
            for seg in batch:
                try:
                    w = load_waveform(seg.audio_ref, seg.channel, seg.start, seg.end)
                except BadSegmentError as e:
                    _handle_failure(seg, e, cfg.strict); continue
                waves.append(w)
                langs.append(_resolve_lang(seg, cfg))
                metas.append(seg)
            if not waves:
                continue
            embs = pool.encode(waves, langs)
            for seg, emb, lang in zip(metas, embs, langs):
                sink.write(EmbeddingRecord(..., modality="speech", embedding=emb))
```

- Batches form in input order; per-segment failures skip within a
  batch (or abort under `--strict`) without poisoning sibling
  segments.
- Audio I/O is single-threaded for v1 (see §2.2).

### 8.4 Determinism

Both encoders run under `torch.inference_mode()`. When
`cfg.deterministic` is set, `cudnn.deterministic=True` and
`cudnn.benchmark=False` (slower). The composite `segment_id` ensures
repeat runs overwrite the same keys.

## 9. Output sinks

```python
class EmbeddingSink(Protocol):
    name: ClassVar[str]
    def open(self) -> ContextManager[None]: ...
    def write(self, rec: EmbeddingRecord) -> None: ...
    def close(self) -> None: ...

class EmbeddingStore(Protocol):   # read side, used by compare
    def __iter__(self) -> Iterator[EmbeddingRecord]: ...
    def __getitem__(self, segment_id: SegmentID) -> EmbeddingRecord: ...
    def __len__(self) -> int: ...
    def ids(self) -> list[SegmentID]: ...
    def matrix(self, ids: list[SegmentID] | None = None
               ) -> tuple[list[SegmentID], np.ndarray]: ...
```

Top-level helper `load_store(path) -> EmbeddingStore` sniffs the
directory layout (presence of `embeddings.scp` → Kaldi;
`dataset_info.json` → HF) and returns the right reader.

### 9.1 `kaldi_ark`

Output directory:

```
out/
├── embeddings.ark          # binary vectors, Kaldi float-matrix format
├── embeddings.scp          # text index: "<segment_id>  <path>:<offset>"
└── metadata.jsonl          # one row per segment, full EmbeddingRecord minus vector
```

- Writes stream through `kaldiio.WriteHelper("ark,scp:...")`.
- `metadata.jsonl` carries `file_id`, `channel`, `spkid`, `start`,
  `end`, `lang`, `modality` — the fields Kaldi's `.scp` cannot
  express.
- Reader joins ark vectors and JSONL metadata on `segment_id`.
- `matrix()` loads a cached dense `(N, 1024)` float32 array lazily.

### 9.2 `hf_dataset`

Output directory is a standard HF `save_to_disk` tree with columns:

`segment_id (string) · file_id (string) · channel (int32) · spkid (string) · start (float32) · end (float32) · lang (string, nullable) · modality (string) · embedding (list[float32], length=1024)`

- Writes buffer in-memory records in blocks of `batch_size * 32` and
  flush as Arrow shards via `datasets.Dataset.from_dict` +
  `concatenate_datasets`. The buffering is behind the sink
  interface — swap to a `pyarrow.RecordBatch` streamer later if
  memory becomes a concern.
- Reader uses `datasets.load_from_disk(path)` and builds a
  `segment_id → row_index` map on first access.
- `matrix()` uses `ds.with_format("numpy")["embedding"]`.
- Output is `push_to_hub`-ready without further processing.

### 9.3 Shared behavior

- `open()` is a context manager; the driver always wraps it so that
  `SIGINT` flushes partial state. Writers register an `atexit` flush
  as a backstop.
- Default write mode: overwrite the output directory. `--append`
  opts into merge semantics; both backends support it (kaldiio
  append mode; HF load + concatenate + re-save).
- Both sinks accept `modality="text"` records transparently, so
  `extract-text` reuses the same backends unchanged.

## 10. Compare (`compare.py`)

### 10.1 Operand grammar

Each of `--a` / `--b` accepts one of:

1. **Inline segment spec:** `"file_id,channel,spkid,start,end[,lang]"`.
   Resolved via `--wav-scp` and encoded on the fly.
2. **Segment ID in an existing store:** `"store:<path>#<segment_id>"`.
   No model loaded; vector is read straight from disk.
3. **Manifest / whole store:** `"<path>"` pointing at a TSV manifest
   or a saved store directory. Expands to many segments.
4. **Text operand:** `"text:<string>@<lang>"` or
   `"text-file:<path>@<lang>"`. Encoded via `TextEncoder`.

Dispatch: if both operands resolve to a single vector → scalar
output; if either expands to ≥2 → matrix output.

### 10.2 Driver

```python
def compare(a: str, b: str, cfg: CompareConfig) -> CompareResult:
    left  = resolve_operand(a, cfg)       # OperandVectors(ids, matrix, modality)
    right = resolve_operand(b, cfg)
    L = l2_normalize(left.matrix)
    R = l2_normalize(right.matrix)
    sims = L @ R.T                        # (N, M) cosine
    return CompareResult(
        a_ids=left.ids, b_ids=right.ids,
        modality_a=left.modality, modality_b=right.modality,
        scores=sims,
    )
```

- L2-normalize then matmul = cosine, numerically stable.
- Normalization happens here, not at write time.
- No batching on compare for v1; chunked matmul is a follow-up when
  matrices exceed VRAM.
- Cross-modal needs no special path — SONAR speech and text share
  the 1024-d space, so the matmul is identity-free.

### 10.3 Operand resolution

| Operand kind | Behavior |
|---|---|
| Inline spec | Build `Segment`, run `load_waveform`, `SpeechEncoderPool.encode`. |
| `store:<path>#<id>` | `load_store(path)[id].embedding`. No model. |
| Manifest / store | Iterate records, stack into `(N, 1024)`. Order preserved. |
| `text:...@lang` / `text-file:...@lang` | Batched `TextEncoder.encode`. |

Both encoders are lazily instantiated and reused across operands. If
both operands are store-reads, no model loads at all.

### 10.4 CLI output

- **Scalar (1×1):** print `cosine=0.8431` plus modality tags; exit 0.
- **Matrix:** write a long-form TSV (`a_id, b_id, score`) by default;
  `--output-format wide` emits a 2D CSV.
- `--top-k N` filters to top-N per `a_id` (retrieval use case).
- `--threshold T` filters to pairs with `score >= T`.
- Python API always returns the full `CompareResult`.

### 10.5 Error policy

- Operand expansion via on-the-fly extraction obeys the same
  `strict` policy as `extract`.
- Store-reads fail hard on missing IDs — a silent gap in a
  similarity matrix is worse than a crash.

## 11. CLI surface

Typer app in `cli.py`. Subcommands:

```
sonar-probe extract
  --source {plain_text|kaldi|hf}        [required]
  --input PATH                           [required]
  --wav-scp PATH                         # required for plain_text; optional for hf
  --hf-split TEXT  --hf-config TEXT
  --sink   {kaldi_ark|hf_dataset}        [required]
  --output PATH                          [required]
  --speech-lang TEXT                     # default language (ISO-3), overridable per-segment
  --device {auto|cuda|cpu|mps}           [default: auto]
  --batch-size INT                       [default: 8]
  --max-seg-seconds FLOAT                [default: 60]
  --encoder-cache-size INT               [default: 4]
  --strict / --no-strict                 [default: no-strict]
  --append / --overwrite                 [default: overwrite]
  --deterministic                        [flag]

sonar-probe extract-text
  --input PATH                           # TSV: text_id<TAB>text<TAB>lang
  --lang TEXT                            # default BCP-47 code, e.g. "eng_Latn"
  --sink / --output / --device / --batch-size / --strict / --append

sonar-probe compare
  --a OPERAND                            [required]
  --b OPERAND                            [required]
  --wav-scp PATH                         # for inline/manifest operands
  --speech-lang TEXT                     # default for inline speech operands
  --device {auto|cuda|cpu|mps}
  --output PATH                          # omit → stdout for scalar; required for matrix
  --output-format {long|wide}            [default: long]
  --top-k INT  --threshold FLOAT         # matrix mode only

sonar-probe inspect PATH                 # dump store summary: count, langs, modalities, dim
```

Shared `--verbose` / `--quiet` on the root callback.

Python API mirrors subcommands:

```python
from sonar_probe import (
    extract, extract_text, compare, load_store,
    Segment, EmbeddingRecord, ExtractConfig, CompareConfig,
)
```

## 12. Errors (`errors.py`)

Typed hierarchy, all inherit from `SonarProbeError`:

| Exception | Raised by | Default policy |
|---|---|---|
| `BadSegmentError` (missing audio, bad timestamps, out-of-bounds channel, decode failure) | `audio.py`, source validators | skip + log; `--strict` → abort |
| `BadSpecError` (malformed input file, missing required column) | sources at open time | always abort — unrecoverable |
| `MissingLangError` (no `--speech-lang` and no per-segment lang) | engine | always abort — picking the wrong encoder would be silent corruption |
| `ModelLoadError` (encoder download or load failed) | engine | always abort |
| `StoreError` (write failure, missing ID on read) | sinks / `load_store` | always abort |

In non-strict mode a `failures.jsonl` sidecar is written next to
`--output` with one row per skipped segment:
`{segment_id, file_id, start, end, error_type, message}`. This enables
targeted re-runs of only the failed subset.

End-of-run summary (always printed):
`written=N  skipped=K  elapsed=T  throughput=X seg/s`.

## 13. Testing

### 13.1 Fixtures (`tests/fixtures/`)

- A 2-channel 16 kHz WAV (2s) for channel-extraction tests.
- A stereo 44.1 kHz WAV for the resample path.
- A Kaldi mini-data-dir (3 utterances, with `segments`, `wav.scp`,
  `utt2spk`, `reco2file_and_channel`).
- A small HF dataset saved via `save_to_disk`.
- A plain-text TSV + matching `wav.scp`.

### 13.2 Unit tests (default `pytest`)

- `tests/sources/` — each backend yields the expected `Segment`s;
  channel coercion, `wav.scp` pipe entries, missing columns, bad
  timestamps all round-trip or raise the typed error.
- `tests/audio.py` — resample idempotence, channel extraction
  correctness vs. ground truth, out-of-bounds raises
  `BadSegmentError`.
- `tests/sinks/` — write → read round-trip produces identical
  vectors and metadata; `--append` merges correctly; both sinks
  accept speech and text records.
- `tests/engine/` — a **fake encoder** fixture (deterministic hash
  of waveform → 1024-d vector) injected via a factory arg; verifies
  batching, lang routing, LRU eviction, scatter-back ordering.
- `tests/compare.py` — operand grammar parser, dispatch rules,
  cross-modal matmul shape, top-k / threshold filtering.
- `tests/cli.py` — Typer's `CliRunner`, drives full
  `extract → compare` with the fake encoder monkey-patched in;
  checks exit codes, `failures.jsonl`, summary line.

### 13.3 Integration tests (`pytest -m integration`, off by default)

- One test that loads the real `sonar_speech_encoder_eng` and checks
  that two renditions of the same sentence stay above a cosine floor
  (e.g. 0.7).
- One test that encodes a sentence as text and the matching audio as
  speech and checks the cross-modal cosine is reasonable.
- Runs nightly in CI; uses a cached checkpoint dir to avoid repeated
  downloads.

### 13.4 Coverage targets

- ≥90% on all modules outside `engine/*.py`.
- ≥70% on `engine/*.py` (the rest requires the real model).

## 14. Configuration surface (`config.py`)

```python
@dataclass
class ExtractConfig:
    device: str = "auto"                 # resolved at construction
    batch_size: int = 8
    speech_lang: Lang | None = None      # default; per-segment override wins
    max_seg_seconds: float = 60.0
    encoder_cache_size: int = 4
    strict: bool = False
    append: bool = False
    deterministic: bool = False

@dataclass
class CompareConfig:
    device: str = "auto"
    speech_lang: Lang | None = None
    wav_scp: Path | None = None
    top_k: int | None = None
    threshold: float | None = None
    output_format: Literal["long", "wide"] = "long"
    strict: bool = False
```

`device="auto"` resolves to `cuda` if available else `cpu`; MPS is
never selected automatically.

## 15. Open questions

None at spec time. Any new question raised during implementation is
to be flagged in the implementation plan rather than silently
decided.
