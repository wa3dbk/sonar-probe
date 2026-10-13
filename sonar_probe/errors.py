"""Typed exception hierarchy."""


class SonarProbeError(Exception):
    """Base for all sonar-probe errors."""


class BadSegmentError(SonarProbeError):
    """Recoverable per-segment failure (missing audio, decode, bounds)."""


class BadSpecError(SonarProbeError):
    """Malformed input manifest, missing required column — unrecoverable."""


class MissingLangError(SonarProbeError):
    """No speech-lang supplied and no per-segment override."""


class ModelLoadError(SonarProbeError):
    """Encoder checkpoint failed to load."""


class StoreError(SonarProbeError):
    """Sink write failed or store read missing an ID."""
