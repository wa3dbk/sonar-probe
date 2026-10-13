from sonar_probe.config import CompareConfig, ExtractConfig, resolve_device


def test_extract_config_defaults():
    c = ExtractConfig()
    assert c.batch_size == 8
    assert c.max_seg_seconds == 60.0
    assert c.encoder_cache_size == 4
    assert c.strict is False
    assert c.append is False
    assert c.deterministic is False
    assert c.device == "auto"


def test_compare_config_defaults():
    c = CompareConfig()
    assert c.device == "auto"
    assert c.output_format == "long"
    assert c.top_k is None
    assert c.threshold is None


def test_resolve_device_cpu_fallback(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_cuda_when_available(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"
