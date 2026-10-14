"""2-D (or N-D) projection of embeddings via UMAP or t-SNE, plus a plotly
HTML scatter writer.

Requires the `viz` extra: ``pip install sonar-probe[viz]``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from sonar_probe.sinks.base import EmbeddingStore
from sonar_probe.types import SegmentID

ProjectionMethod = Literal["umap", "tsne"]

_VIZ_INSTALL_HINT = "pip install sonar-probe[viz]"


def _build_umap(n_components: int, random_state: int | None, **kwargs):
    try:
        import umap
    except ImportError as e:
        raise ImportError(
            f"umap-learn not available; {_VIZ_INSTALL_HINT}"
        ) from e
    return umap.UMAP(
        n_components=n_components, random_state=random_state, **kwargs
    )


def _build_tsne(n_components: int, random_state: int | None, **kwargs):
    try:
        from sklearn.manifold import TSNE
    except ImportError as e:
        raise ImportError(
            f"scikit-learn not available; {_VIZ_INSTALL_HINT}"
        ) from e
    return TSNE(
        n_components=n_components, random_state=random_state, **kwargs
    )


def project_embeddings(
    mat: np.ndarray,
    method: ProjectionMethod = "umap",
    n_components: int = 2,
    random_state: int | None = 42,
    **kwargs,
) -> np.ndarray:
    """Project an ``(N, D)`` embedding matrix to ``(N, n_components)`` float32.

    ``kwargs`` are forwarded to the underlying ``umap.UMAP`` or
    ``sklearn.manifold.TSNE`` constructor — e.g. ``n_neighbors`` for UMAP,
    ``perplexity`` for t-SNE. Set these when ``N`` is small; the library
    defaults assume >15 samples.
    """
    if mat.ndim != 2:
        raise ValueError(f"expected 2-D matrix, got shape {mat.shape}")
    if method == "umap":
        reducer = _build_umap(n_components, random_state, **kwargs)
    elif method == "tsne":
        reducer = _build_tsne(n_components, random_state, **kwargs)
    else:
        raise ValueError(
            f"unknown method {method!r}; expected 'umap' or 'tsne'"
        )
    coords = reducer.fit_transform(mat)
    return np.asarray(coords, dtype=np.float32)


def project_store(
    store: EmbeddingStore,
    method: ProjectionMethod = "umap",
    ids: list[SegmentID] | None = None,
    n_components: int = 2,
    random_state: int | None = 42,
    **kwargs,
) -> tuple[list[SegmentID], np.ndarray]:
    """Project every (or a selected subset of) embeddings in a store.

    Returns ``(ids, coords)`` where ``coords.shape == (len(ids), n_components)``.
    """
    picked_ids, mat = store.matrix(ids=ids)
    coords = project_embeddings(
        mat,
        method=method,
        n_components=n_components,
        random_state=random_state,
        **kwargs,
    )
    return picked_ids, coords


def write_scatter_html(
    ids: list[SegmentID],
    coords: np.ndarray,
    labels: list[str] | None = None,
    title: str = "sonar-probe projection",
    output: Path | str = "projection.html",
) -> Path:
    """Write an interactive 2-D plotly scatter plot to ``output``.

    ``labels`` groups points by category for the legend; if ``None`` the plot
    is a single trace. ``ids`` are shown on hover. Returns the output path.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError(
            f"plotly not available; {_VIZ_INSTALL_HINT}"
        ) from e
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            f"coords must be (N, 2); got shape {coords.shape}"
        )
    if len(ids) != coords.shape[0]:
        raise ValueError(
            f"ids length {len(ids)} does not match coords rows {coords.shape[0]}"
        )
    if labels is not None and len(labels) != len(ids):
        raise ValueError(
            f"labels length {len(labels)} does not match ids length {len(ids)}"
        )

    fig = go.Figure()
    if labels is None:
        fig.add_trace(
            go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                text=list(ids),
                hovertemplate="%{text}<extra></extra>",
            )
        )
    else:
        for group in sorted(set(labels)):
            picks = [i for i, lab in enumerate(labels) if lab == group]
            fig.add_trace(
                go.Scatter(
                    x=coords[picks, 0],
                    y=coords[picks, 1],
                    mode="markers",
                    name=str(group),
                    text=[ids[i] for i in picks],
                    hovertemplate="%{text}<br>" + str(group) + "<extra></extra>",
                )
            )
    fig.update_layout(
        title=title, xaxis_title="dim 1", yaxis_title="dim 2"
    )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out))
    return out
