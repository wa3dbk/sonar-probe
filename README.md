# sonar-probe

Extract and compare [SONAR](https://github.com/facebookresearch/SONAR) speech and text embeddings.

## Install

    pip install -e .[dev]

## Quick usage

See `docs/superpowers/specs/2026-10-14-sonar-probe-design.md` for the full design.

## Running integration tests

Unit tests use a fake encoder and run offline. Integration tests load
real SONAR checkpoints (several hundred MB on first run) and are off by
default:

    pytest -m integration

Set `HF_HOME` to a persistent dir to cache checkpoints across runs.
