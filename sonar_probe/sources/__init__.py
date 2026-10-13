"""Source backends and registry."""
from sonar_probe.sources.base import (
    SegmentSource,
    get_source,
    register_source,
)
# Import backends so they register on package load.
from sonar_probe.sources import kaldi, plain_text  # noqa: F401

__all__ = ["SegmentSource", "get_source", "register_source"]
