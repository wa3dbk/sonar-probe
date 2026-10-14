"""Typer CLI for sonar-probe."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

import typer

from sonar_probe import (
    CompareConfig,
    ExtractConfig,
    compare as api_compare,
    extract as api_extract,
    extract_text as api_extract_text,
    load_store,
    read_text_tsv,
)
from sonar_probe.project import project_embeddings, write_scatter_html
from sonar_probe.sinks import get_sink
from sonar_probe.sources import get_source

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def _root(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
):
    level = logging.WARNING
    if verbose:
        level = logging.DEBUG
    if quiet:
        level = logging.ERROR
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


@app.command()
def extract(
    source: str = typer.Option(..., "--source"),
    input: Path = typer.Option(..., "--input"),
    wav_scp: Optional[Path] = typer.Option(None, "--wav-scp"),
    hf_split: Optional[str] = typer.Option(None, "--hf-split"),
    hf_config: Optional[str] = typer.Option(None, "--hf-config"),
    sink: str = typer.Option(..., "--sink"),
    output: Path = typer.Option(..., "--output"),
    speech_lang: Optional[str] = typer.Option(None, "--speech-lang"),
    device: str = typer.Option("auto", "--device"),
    batch_size: int = typer.Option(8, "--batch-size"),
    max_seg_seconds: float = typer.Option(60.0, "--max-seg-seconds"),
    encoder_cache_size: int = typer.Option(4, "--encoder-cache-size"),
    strict: bool = typer.Option(False, "--strict/--no-strict"),
    append: bool = typer.Option(False, "--append/--overwrite"),
    deterministic: bool = typer.Option(False, "--deterministic"),
):
    kwargs = {"input": input}
    if source == "plain_text":
        if wav_scp is None:
            typer.echo("ERROR: --wav-scp required for plain_text", err=True)
            raise typer.Exit(2)
        kwargs["wav_scp"] = wav_scp
    elif source == "hf":
        if wav_scp is not None:
            kwargs["wav_scp"] = wav_scp
        if hf_split is not None:
            kwargs["split"] = hf_split
        if hf_config is not None:
            kwargs["config"] = hf_config
    src = get_source(source, **kwargs)
    snk = get_sink(sink, output=output, append=append)
    cfg = ExtractConfig(
        device=device,
        batch_size=batch_size,
        speech_lang=speech_lang,
        max_seg_seconds=max_seg_seconds,
        encoder_cache_size=encoder_cache_size,
        strict=strict,
        append=append,
        deterministic=deterministic,
    )
    res = api_extract(src, snk, cfg)
    typer.echo(f"written={res.written} skipped={res.skipped}")


@app.command("extract-text")
def extract_text_cmd(
    input: Path = typer.Option(..., "--input"),
    lang: Optional[str] = typer.Option(None, "--lang"),
    sink: str = typer.Option(..., "--sink"),
    output: Path = typer.Option(..., "--output"),
    device: str = typer.Option("auto", "--device"),
    batch_size: int = typer.Option(8, "--batch-size"),
    strict: bool = typer.Option(False, "--strict/--no-strict"),
    append: bool = typer.Option(False, "--append/--overwrite"),
):
    snk = get_sink(sink, output=output, append=append)
    cfg = ExtractConfig(
        device=device, batch_size=batch_size, strict=strict, append=append
    )
    res = api_extract_text(read_text_tsv(input, default_lang=lang), snk, cfg)
    typer.echo(f"written={res.written} skipped={res.skipped}")


@app.command()
def compare(
    a: str = typer.Option(..., "--a"),
    b: str = typer.Option(..., "--b"),
    wav_scp: Optional[Path] = typer.Option(None, "--wav-scp"),
    speech_lang: Optional[str] = typer.Option(None, "--speech-lang"),
    device: str = typer.Option("auto", "--device"),
    output: Optional[Path] = typer.Option(None, "--output"),
    output_format: str = typer.Option("long", "--output-format"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    threshold: Optional[float] = typer.Option(None, "--threshold"),
):
    cfg = CompareConfig(
        device=device,
        speech_lang=speech_lang,
        wav_scp=wav_scp,
        top_k=top_k,
        threshold=threshold,
        output_format=output_format,  # type: ignore
    )
    res = api_compare(a, b, cfg)
    if res.scores.shape == (1, 1):
        typer.echo(
            f"cosine={float(res.scores[0,0]):.4f} "
            f"modality={res.modality_a}↔{res.modality_b}"
        )
        return
    if output is None:
        typer.echo("ERROR: matrix output requires --output", err=True)
        raise typer.Exit(2)
    if output_format == "long":
        with Path(output).open("w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["a_id", "b_id", "score"])
            for a_id, b_id, s in res.iter_pairs(top_k=top_k, threshold=threshold):
                w.writerow([a_id, b_id, f"{s:.6f}"])
    elif output_format == "wide":
        with Path(output).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([""] + res.b_ids)
            for a_id, row in zip(res.a_ids, res.scores):
                w.writerow([a_id] + [f"{float(v):.6f}" for v in row])
    else:
        typer.echo(f"ERROR: unknown --output-format {output_format!r}", err=True)
        raise typer.Exit(2)
    typer.echo(f"wrote {output}")


@app.command()
def visualize(
    path: Path = typer.Argument(..., help="Store path"),
    method: str = typer.Option("umap", "--method", help="umap | tsne"),
    output: Path = typer.Option(Path("projection.html"), "--output", "-o"),
    color_by: str = typer.Option(
        "modality", "--color-by",
        help="One of: modality, lang, spkid, none",
    ),
    n_neighbors: Optional[int] = typer.Option(None, "--n-neighbors"),
    perplexity: Optional[float] = typer.Option(None, "--perplexity"),
    random_state: int = typer.Option(42, "--random-state"),
):
    if color_by not in ("modality", "lang", "spkid", "none"):
        typer.echo(
            "ERROR: --color-by must be one of modality/lang/spkid/none",
            err=True,
        )
        raise typer.Exit(2)
    store = load_store(path)
    ids, mat = store.matrix()
    kwargs: dict = {}
    if method == "umap" and n_neighbors is not None:
        kwargs["n_neighbors"] = n_neighbors
    if method == "tsne" and perplexity is not None:
        kwargs["perplexity"] = perplexity
    coords = project_embeddings(
        mat, method=method, random_state=random_state, **kwargs  # type: ignore[arg-type]
    )
    labels: Optional[list[str]] = None
    if color_by != "none":
        meta = {rec.segment_id: getattr(rec, color_by) for rec in store}
        labels = [str(meta.get(i) or "(none)") for i in ids]
    write_scatter_html(
        ids, coords, labels=labels,
        title=f"{path.name} ({method})", output=output,
    )
    typer.echo(f"wrote {output} ({len(ids)} points)")


@app.command()
def inspect(path: Path):
    store = load_store(path)
    langs: set[str] = set()
    modalities: set[str] = set()
    for rec in store:
        if rec.lang:
            langs.add(rec.lang)
        modalities.add(rec.modality)
    typer.echo(
        f"count={len(store)} dim=1024 "
        f"modalities={sorted(modalities)} langs={sorted(langs)}"
    )


if __name__ == "__main__":
    app()
