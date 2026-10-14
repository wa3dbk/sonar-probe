"""Tests for sonar_probe.project (UMAP / t-SNE 2-D projection)."""
from __future__ import annotations

import sys

import numpy as np
import pytest

from sonar_probe import project_embeddings, project_store
from sonar_probe.project import write_scatter_html


def _random_mat(n: int = 20, d: int = 1024, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


class _StubStore:
    """Minimal EmbeddingStore stand-in exposing just ids() and matrix()."""

    def __init__(self, ids: list[str], mat: np.ndarray):
        self._ids = list(ids)
        self._mat = mat

    def ids(self) -> list[str]:
        return list(self._ids)

    def matrix(self, ids=None):
        if ids is None:
            return list(self._ids), self._mat
        index = {sid: i for i, sid in enumerate(self._ids)}
        picks = [index[s] for s in ids]
        return list(ids), self._mat[picks]

    def __iter__(self):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError

    def __len__(self):
        return len(self._ids)


def test_project_embeddings_tsne_shape_and_dtype():
    pytest.importorskip("sklearn")
    mat = _random_mat(n=20)
    out = project_embeddings(mat, method="tsne", perplexity=5)
    assert out.shape == (20, 2)
    assert out.dtype == np.float32


def test_project_embeddings_umap_shape_and_dtype():
    pytest.importorskip("umap")
    mat = _random_mat(n=20)
    out = project_embeddings(mat, method="umap", n_neighbors=5)
    assert out.shape == (20, 2)
    assert out.dtype == np.float32


def test_project_embeddings_n_components_passthrough():
    pytest.importorskip("sklearn")
    mat = _random_mat(n=20)
    out = project_embeddings(mat, method="tsne", n_components=3, perplexity=5)
    assert out.shape == (20, 3)


def test_project_embeddings_unknown_method_raises():
    mat = _random_mat(n=5)
    with pytest.raises(ValueError, match="unknown method"):
        project_embeddings(mat, method="pca")  # type: ignore[arg-type]


def test_project_embeddings_1d_input_raises():
    with pytest.raises(ValueError, match="2-D"):
        project_embeddings(np.zeros(10, dtype=np.float32), method="tsne")


def test_project_embeddings_umap_import_error_hints_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "umap", None)
    mat = _random_mat(n=5)
    with pytest.raises(ImportError, match=r"pip install sonar-probe\[viz\]"):
        project_embeddings(mat, method="umap")


def test_project_embeddings_tsne_import_error_hints_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.manifold", None)
    mat = _random_mat(n=5)
    with pytest.raises(ImportError, match=r"pip install sonar-probe\[viz\]"):
        project_embeddings(mat, method="tsne")


def test_project_store_round_trip():
    pytest.importorskip("sklearn")
    mat = _random_mat(n=20)
    store = _StubStore(ids=[f"seg{i}" for i in range(20)], mat=mat)
    ids, coords = project_store(store, method="tsne", perplexity=5)
    assert ids == [f"seg{i}" for i in range(20)]
    assert coords.shape == (20, 2)
    assert coords.dtype == np.float32


def test_project_store_subset_preserves_order():
    pytest.importorskip("sklearn")
    mat = _random_mat(n=20)
    store = _StubStore(ids=[f"seg{i}" for i in range(20)], mat=mat)
    subset = [f"seg{i}" for i in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]]
    ids, coords = project_store(store, method="tsne", ids=subset, perplexity=3)
    assert ids == subset
    assert coords.shape == (10, 2)


def test_write_scatter_html_single_trace(tmp_path):
    pytest.importorskip("plotly")
    coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    out = tmp_path / "viz.html"
    returned = write_scatter_html(["a", "b", "c"], coords, output=out)
    assert returned == out
    assert out.is_file()
    content = out.read_text()
    assert "plotly" in content.lower()
    for sid in ["a", "b", "c"]:
        assert sid in content


def test_write_scatter_html_grouped_by_label(tmp_path):
    pytest.importorskip("plotly")
    coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    out = tmp_path / "viz.html"
    write_scatter_html(
        ["a", "b", "c"], coords, labels=["eng", "fra", "eng"], output=out
    )
    content = out.read_text()
    assert "eng" in content
    assert "fra" in content


def test_write_scatter_html_shape_mismatch_raises(tmp_path):
    pytest.importorskip("plotly")
    coords = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="does not match"):
        write_scatter_html(["a", "b", "c"], coords, output=tmp_path / "x.html")


def test_write_scatter_html_wrong_dim_raises(tmp_path):
    pytest.importorskip("plotly")
    coords = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        write_scatter_html(["a", "b", "c"], coords, output=tmp_path / "x.html")


def test_write_scatter_html_import_error_hints_extra(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "plotly", None)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    coords = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ImportError, match=r"pip install sonar-probe\[viz\]"):
        write_scatter_html(["a", "b"], coords, output=tmp_path / "x.html")
