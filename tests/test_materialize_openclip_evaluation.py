"""Pruebas de materialización de evaluación OpenCLIP."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.openclip_baseline.materialize_evaluation import (
    OpenCLIPEvaluationConfig,
    build_openclip_evaluation_artifacts,
    materialize_openclip_evaluation,
)
from src.openclip_baseline.retrieval import (
    OpenCLIPEmbeddingStore,
)


def _build_store() -> OpenCLIPEmbeddingStore:
    """Construye tres pares sintéticos."""

    return OpenCLIPEmbeddingStore(
        item_ids=np.asarray(
            [
                "item-a",
                "item-b",
                "item-c",
            ],
            dtype=np.str_,
        ),
        museums=np.asarray(
            [
                "CMA",
                "MET",
                "MET",
            ],
            dtype=np.str_,
        ),
        image_embeddings=np.eye(
            3,
            dtype=np.float32,
        ),
        text_visual_embeddings=np.eye(
            3,
            dtype=np.float32,
        ),
        text_metadata_embeddings=np.asarray(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def _config() -> OpenCLIPEvaluationConfig:
    """Devuelve una configuración compatible con tres registros."""

    return OpenCLIPEvaluationConfig(
        cutoffs=(1, 2, 3),
        alphas=(0.0, 0.5, 1.0),
        selected_alpha=0.5,
    )


def test_config_rejects_invalid_alpha() -> None:
    """El peso seleccionado debe pertenecer al intervalo unitario."""

    with pytest.raises(
        ValueError,
        match="selected_alpha",
    ):
        OpenCLIPEvaluationConfig(
            selected_alpha=1.1
        )


def test_build_evaluation_artifacts() -> None:
    """La construcción debe incluir resúmenes y consultas."""

    artifacts = build_openclip_evaluation_artifacts(
        store=_build_store(),
        config=_config(),
    )

    assert len(artifacts.per_query) == 9
    assert len(artifacts.alpha_grid) == 9
    assert len(
        artifacts.comparison_vs_random
    ) == 2

    assert set(
        artifacts.per_query[
            "configuration"
        ]
    ) == {
        "text_visual",
        "text_metadata",
        "text_fused_alpha_0.50",
    }

    assert "random" in set(
        artifacts.summary[
            "configuration"
        ]
    )


def test_comparison_contains_random_lifts() -> None:
    """La comparación debe calcular mejoras respecto del azar."""

    artifacts = build_openclip_evaluation_artifacts(
        store=_build_store(),
        config=_config(),
    )

    comparison = (
        artifacts.comparison_vs_random
    )

    assert "mrr_lift_vs_random" in comparison
    assert (
        comparison[
            "mrr_lift_vs_random"
        ] > 0.0
    ).all()

    assert (
        comparison[
            "median_rank_reduction_vs_random"
        ].notna()
    ).all()


def test_materialize_openclip_evaluation(
    tmp_path: Path,
) -> None:
    """Los artefactos y la procedencia deben escribirse."""

    store = _build_store()
    embeddings_path = (
        tmp_path / "embeddings.npz"
    )

    np.savez(
        embeddings_path,
        item_ids=store.item_ids,
        museums=store.museums,
        image_embeddings=(
            store.image_embeddings
        ),
        text_visual_embeddings=(
            store.text_visual_embeddings
        ),
        text_metadata_embeddings=(
            store.text_metadata_embeddings
        ),
    )

    summary_path = tmp_path / "summary.csv"
    per_query_path = tmp_path / "per_query.csv"
    alpha_grid_path = tmp_path / "alpha_grid.csv"
    comparison_path = tmp_path / "comparison.csv"
    provenance_path = tmp_path / "provenance.json"

    materialize_openclip_evaluation(
        embeddings_path=embeddings_path,
        summary_path=summary_path,
        per_query_path=per_query_path,
        alpha_grid_path=alpha_grid_path,
        comparison_path=comparison_path,
        provenance_path=provenance_path,
        repository_root=tmp_path,
        config=_config(),
    )

    for path in (
        summary_path,
        per_query_path,
        alpha_grid_path,
        comparison_path,
        provenance_path,
    ):
        assert path.is_file()

    summary = pd.read_csv(
        summary_path
    )
    per_query = pd.read_csv(
        per_query_path
    )

    assert not summary.empty
    assert len(per_query) == 9

    provenance = json.loads(
        provenance_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        provenance["coverage"][
            "total_records"
        ]
        == 3
    )
    assert (
        provenance["configuration"][
            "selected_alpha"
        ]
        == pytest.approx(0.5)
    )
    assert (
        provenance["input"][
            "embeddings"
        ]["sha256"]
    )