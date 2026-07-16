"""Pruebas de evaluación de recuperación OpenCLIP."""

from __future__ import annotations

import numpy as np
import pytest

from src.openclip_baseline.evaluation import (
    build_fused_text_matrix,
    build_random_paired_baseline,
    evaluate_alpha_grid,
    evaluate_fused_text_retrieval,
    evaluate_paired_retrieval,
)

from src.openclip_baseline.retrieval import (
    OpenCLIPEmbeddingStore,
)


def _build_store() -> OpenCLIPEmbeddingStore:
    """Construye un corpus sintético con tres pares."""

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

def test_random_paired_baseline() -> None:
    """El ranking aleatorio debe tener expectativas analíticas."""

    baseline = build_random_paired_baseline(
        candidate_count=4,
        cutoffs=(1, 2, 4),
    )

    row = baseline.iloc[0]

    assert row["scope"] == "RANDOM"
    assert row["total_queries"] == 4

    assert row["recall_at_1"] == pytest.approx(
        0.25
    )
    assert row["recall_at_2"] == pytest.approx(
        0.50
    )
    assert row["recall_at_4"] == pytest.approx(
        1.00
    )

    expected_mrr = (
        1.0
        + 1.0 / 2.0
        + 1.0 / 3.0
        + 1.0 / 4.0
    ) / 4.0

    assert row["mrr"] == pytest.approx(
        expected_mrr
    )
    assert row["mean_rank"] == pytest.approx(
        2.5
    )
    assert row["median_rank"] == pytest.approx(
        2.5
    )


def test_random_baseline_rejects_empty_candidates() -> None:
    """La línea base requiere al menos un candidato."""

    with pytest.raises(
        ValueError,
        match="candidate_count",
    ):
        build_random_paired_baseline(
            candidate_count=0,
            cutoffs=(1,),
        )

def test_perfect_paired_retrieval() -> None:
    """Modalidades idénticas deben recuperar todos los pares."""

    evaluation = evaluate_paired_retrieval(
        store=_build_store(),
        query_modality="text_visual",
        candidate_modality="image",
        cutoffs=(1, 2, 3),
    )

    assert (
        evaluation.per_query[
            "matched_rank"
        ].tolist()
        == [1, 1, 1]
    )

    all_summary = evaluation.summary.loc[
        evaluation.summary["scope"] == "ALL"
    ].iloc[0]

    assert all_summary["recall_at_1"] == 1.0
    assert all_summary["recall_at_2"] == 1.0
    assert all_summary["mrr"] == 1.0
    assert all_summary["mean_rank"] == 1.0


def test_summary_contains_museum_scopes() -> None:
    """El resumen debe separar consultas CMA y MET."""

    evaluation = evaluate_paired_retrieval(
        store=_build_store(),
        query_modality="text_visual",
        cutoffs=(1,),
    )

    assert set(
        evaluation.summary["scope"]
    ) == {
        "ALL",
        "CMA",
        "MET",
    }

    counts = dict(
        zip(
            evaluation.summary["scope"],
            evaluation.summary[
                "total_queries"
            ],
        )
    )

    assert counts == {
        "ALL": 3,
        "CMA": 1,
        "MET": 2,
    }


def test_misaligned_metadata_has_lower_performance() -> None:
    """Metadatos permutados no deben recuperar todos en rango uno."""

    evaluation = evaluate_paired_retrieval(
        store=_build_store(),
        query_modality="text_metadata",
        candidate_modality="image",
        cutoffs=(1, 3),
    )

    all_summary = evaluation.summary.loc[
        evaluation.summary["scope"] == "ALL"
    ].iloc[0]

    assert all_summary["recall_at_1"] == 0.0
    assert all_summary["recall_at_3"] == 1.0
    assert all_summary["mrr"] < 1.0


def test_build_fused_text_matrix_at_visual_endpoint() -> None:
    """Alpha uno debe reproducir el texto visual."""

    store = _build_store()

    fused = build_fused_text_matrix(
        store=store,
        alpha=1.0,
    )

    np.testing.assert_allclose(
        fused,
        store.text_visual_embeddings,
        atol=1e-6,
    )


def test_fused_retrieval_records_alpha() -> None:
    """La evaluación fusionada debe registrar su peso."""

    evaluation = evaluate_fused_text_retrieval(
        store=_build_store(),
        alpha=0.75,
        cutoffs=(1, 2, 3),
    )

    np.testing.assert_allclose(
        evaluation.summary["alpha"].to_numpy(
            dtype=np.float64,
        ),
        0.75,
        atol=1e-12,
    )

    assert (
        evaluation.summary[
            "query_modality"
        ]
        == "text_fused"
    ).all()


def test_alpha_grid_returns_each_scope_and_alpha() -> None:
    """La cuadrícula debe conservar todas las combinaciones."""

    result = evaluate_alpha_grid(
        store=_build_store(),
        alphas=(0.0, 0.5, 1.0),
        cutoffs=(1, 3),
    )

    assert len(result) == 9
    assert set(result["alpha"]) == {
        0.0,
        0.5,
        1.0,
    }
    assert set(result["scope"]) == {
        "ALL",
        "CMA",
        "MET",
    }


def test_invalid_cutoffs_are_rejected() -> None:
    """Los valores de K deben ser válidos."""

    with pytest.raises(
        ValueError,
        match="positivos",
    ):
        evaluate_paired_retrieval(
            store=_build_store(),
            query_modality="text_visual",
            cutoffs=(0, 1),
        )

    with pytest.raises(
        ValueError,
        match="superar",
    ):
        evaluate_paired_retrieval(
            store=_build_store(),
            query_modality="text_visual",
            cutoffs=(4,),
        )


def test_invalid_alpha_is_rejected() -> None:
    """El peso de fusión debe permanecer en el intervalo unitario."""

    with pytest.raises(
        ValueError,
        match="alpha",
    ):
        build_fused_text_matrix(
            store=_build_store(),
            alpha=-0.1,
        )