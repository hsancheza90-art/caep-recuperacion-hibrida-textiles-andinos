"""Pruebas de las métricas de recuperación."""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.retrieval_metrics import (
    aggregate_metrics,
    aggregate_metrics_by_culture,
    evaluate_rankings,
)


def build_toy_ground_truth() -> pd.DataFrame:
    """Ground truth con dos culturas y dos objetos por cultura."""

    return pd.DataFrame(
        [
            {
                "item_id": "A1",
                "culture_canonical": "Culture_A",
                "relevant_item_count": 1,
                "query_eligible": True,
            },
            {
                "item_id": "A2",
                "culture_canonical": "Culture_A",
                "relevant_item_count": 1,
                "query_eligible": True,
            },
            {
                "item_id": "B1",
                "culture_canonical": "Culture_B",
                "relevant_item_count": 1,
                "query_eligible": True,
            },
            {
                "item_id": "B2",
                "culture_canonical": "Culture_B",
                "relevant_item_count": 1,
                "query_eligible": True,
            },
        ]
    )


def relevant_for(
    query_item_id: str,
) -> str:
    mapping = {
        "A1": "A2",
        "A2": "A1",
        "B1": "B2",
        "B2": "B1",
    }

    return mapping[query_item_id]


def build_perfect_ranking() -> pd.DataFrame:
    """Coloca el relevante en la primera posición."""

    item_ids = [
        "A1",
        "A2",
        "B1",
        "B2",
    ]

    rows = []

    for query_item_id in item_ids:
        relevant_item_id = relevant_for(
            query_item_id
        )

        non_relevant = sorted(
            set(item_ids)
            - {
                query_item_id,
                relevant_item_id,
            }
        )

        candidates = [
            relevant_item_id,
            *non_relevant,
        ]

        for rank, candidate_item_id in enumerate(
            candidates,
            start=1,
        ):
            rows.append(
                {
                    "query_item_id": query_item_id,
                    "candidate_item_id": (
                        candidate_item_id
                    ),
                    "rank": rank,
                }
            )

    return pd.DataFrame(rows)


def build_worst_ranking() -> pd.DataFrame:
    """Coloca el único relevante en la última posición."""

    item_ids = [
        "A1",
        "A2",
        "B1",
        "B2",
    ]

    rows = []

    for query_item_id in item_ids:
        relevant_item_id = relevant_for(
            query_item_id
        )

        non_relevant = sorted(
            set(item_ids)
            - {
                query_item_id,
                relevant_item_id,
            }
        )

        candidates = [
            *non_relevant,
            relevant_item_id,
        ]

        for rank, candidate_item_id in enumerate(
            candidates,
            start=1,
        ):
            rows.append(
                {
                    "query_item_id": query_item_id,
                    "candidate_item_id": (
                        candidate_item_id
                    ),
                    "rank": rank,
                }
            )

    return pd.DataFrame(rows)


def test_perfect_ranking_scores_are_one() -> None:
    ground_truth = build_toy_ground_truth()
    rankings = build_perfect_ranking()

    metrics = evaluate_rankings(
        rankings,
        ground_truth,
        k_values=(1, 3),
    )

    assert metrics["first_relevant_rank"].eq(1).all()
    assert metrics["reciprocal_rank"].eq(1.0).all()

    assert metrics["precision_at_1"].eq(1.0).all()
    assert metrics["recall_at_1"].eq(1.0).all()
    assert metrics["ndcg_at_1"].eq(1.0).all()

    assert metrics["recall_at_3"].eq(1.0).all()
    assert metrics["ndcg_at_3"].eq(1.0).all()

    assert metrics["precision_at_3"].eq(
        1.0 / 3.0
    ).all()


def test_worst_ranking_scores_are_correct() -> None:
    ground_truth = build_toy_ground_truth()
    rankings = build_worst_ranking()

    metrics = evaluate_rankings(
        rankings,
        ground_truth,
        k_values=(1, 3),
    )

    assert metrics["first_relevant_rank"].eq(3).all()

    assert metrics["reciprocal_rank"].eq(
        1.0 / 3.0
    ).all()

    assert metrics["precision_at_1"].eq(0.0).all()
    assert metrics["recall_at_1"].eq(0.0).all()
    assert metrics["ndcg_at_1"].eq(0.0).all()

    assert metrics["precision_at_3"].eq(
        1.0 / 3.0
    ).all()

    assert metrics["recall_at_3"].eq(1.0).all()

    assert metrics["ndcg_at_3"].eq(0.5).all()


def test_aggregate_metrics_matches_query_means() -> None:
    ground_truth = build_toy_ground_truth()

    per_query = evaluate_rankings(
        build_perfect_ranking(),
        ground_truth,
        k_values=(1, 3),
    )

    aggregate = aggregate_metrics(
        per_query,
        k_values=(1, 3),
    )

    assert len(aggregate) == 1
    assert aggregate.loc[0, "query_count"] == 4
    assert aggregate.loc[0, "mrr"] == 1.0
    assert aggregate.loc[0, "precision_at_1"] == 1.0
    assert aggregate.loc[0, "recall_at_1"] == 1.0
    assert aggregate.loc[0, "ndcg_at_1"] == 1.0


def test_culture_summary_has_one_row_per_culture() -> None:
    ground_truth = build_toy_ground_truth()

    per_query = evaluate_rankings(
        build_perfect_ranking(),
        ground_truth,
        k_values=(1, 3),
    )

    summary = aggregate_metrics_by_culture(
        per_query,
        k_values=(1, 3),
    )

    assert set(
        summary["culture_canonical"]
    ) == {
        "Culture_A",
        "Culture_B",
    }

    assert summary["query_count"].eq(2).all()
    assert summary["mrr"].eq(1.0).all()


def test_validation_rejects_self_pairs() -> None:
    ground_truth = build_toy_ground_truth()
    rankings = build_perfect_ranking()

    rankings.loc[0, "candidate_item_id"] = (
        rankings.loc[0, "query_item_id"]
    )

    with pytest.raises(
        ValueError,
        match="autorreferentes",
    ):
        evaluate_rankings(
            rankings,
            ground_truth,
            k_values=(1, 3),
        )


def test_validation_rejects_duplicate_pairs() -> None:
    ground_truth = build_toy_ground_truth()
    rankings = build_perfect_ranking()

    duplicate_row = rankings.iloc[[0]].copy()

    rankings = pd.concat(
        [
            rankings,
            duplicate_row,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicados",
    ):
        evaluate_rankings(
            rankings,
            ground_truth,
            k_values=(1, 3),
        )


def test_validation_rejects_incomplete_query_ranking() -> None:
    ground_truth = build_toy_ground_truth()
    rankings = build_perfect_ranking()

    rankings = rankings[
        ~(
            rankings["query_item_id"].eq("A1")
            & rankings["rank"].eq(3)
        )
    ].copy()

    with pytest.raises(
        ValueError,
        match="universo completo",
    ):
        evaluate_rankings(
            rankings,
            ground_truth,
            k_values=(1, 3),
        )


def test_validation_rejects_unknown_candidate() -> None:
    ground_truth = build_toy_ground_truth()
    rankings = build_perfect_ranking()

    rankings.loc[
        rankings.index[0],
        "candidate_item_id",
    ] = "UNKNOWN"

    with pytest.raises(
        ValueError,
        match="fuera del universo",
    ):
        evaluate_rankings(
            rankings,
            ground_truth,
            k_values=(1, 3),
        )


def test_validation_rejects_nonconsecutive_ranks() -> None:
    ground_truth = build_toy_ground_truth()
    rankings = build_perfect_ranking()

    mask = (
        rankings["query_item_id"].eq("A1")
        & rankings["rank"].eq(3)
    )

    rankings.loc[mask, "rank"] = 4

    with pytest.raises(
        ValueError,
        match="consecutivas",
    ):
        evaluate_rankings(
            rankings,
            ground_truth,
            k_values=(1, 3),
        )


def test_ground_truth_rejects_inconsistent_counts() -> None:
    ground_truth = build_toy_ground_truth()

    ground_truth.loc[
        ground_truth["item_id"].eq("A1"),
        "relevant_item_count",
    ] = 2

    with pytest.raises(
        ValueError,
        match="conteos de relevancia",
    ):
        evaluate_rankings(
            build_perfect_ranking(),
            ground_truth,
            k_values=(1, 3),
        )


def test_k_cannot_exceed_candidate_universe() -> None:
    ground_truth = build_toy_ground_truth()

    with pytest.raises(
        ValueError,
        match="superar",
    ):
        evaluate_rankings(
            build_perfect_ranking(),
            ground_truth,
            k_values=(1, 4),
        )


def test_evaluation_is_deterministic() -> None:
    ground_truth = build_toy_ground_truth()
    rankings = build_perfect_ranking()

    first = evaluate_rankings(
        rankings,
        ground_truth,
        k_values=(1, 3),
    )

    second = evaluate_rankings(
        rankings,
        ground_truth,
        k_values=(1, 3),
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )