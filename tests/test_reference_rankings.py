"""Pruebas de los rankings oracle y aleatorio."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.build_reference_rankings import (
    COMBINED_SUMMARY_PATH,
    DEFAULT_RANDOM_SEED,
    ORACLE_BY_CULTURE_PATH,
    ORACLE_OVERALL_PATH,
    ORACLE_PER_QUERY_PATH,
    ORACLE_RANKING_PATH,
    RANDOM_BY_CULTURE_PATH,
    RANDOM_OVERALL_PATH,
    RANDOM_PER_QUERY_PATH,
    RANDOM_RANKING_PATH,
    build_oracle_ranking,
    build_random_ranking,
    build_reference_artifacts,
    stable_query_seed,
)
from src.evaluation.retrieval_metrics import (
    aggregate_metrics,
    evaluate_rankings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "culture_ground_truth_strict_v1.csv"
)


def load_ground_truth() -> pd.DataFrame:
    return pd.read_csv(
        GROUND_TRUTH_PATH
    )


def load_oracle_ranking() -> pd.DataFrame:
    return pd.read_csv(
        ORACLE_RANKING_PATH
    )


def load_random_ranking() -> pd.DataFrame:
    return pd.read_csv(
        RANDOM_RANKING_PATH
    )


def test_reference_files_exist() -> None:
    expected_paths = [
        ORACLE_RANKING_PATH,
        RANDOM_RANKING_PATH,
        ORACLE_PER_QUERY_PATH,
        ORACLE_OVERALL_PATH,
        ORACLE_BY_CULTURE_PATH,
        RANDOM_PER_QUERY_PATH,
        RANDOM_OVERALL_PATH,
        RANDOM_BY_CULTURE_PATH,
        COMBINED_SUMMARY_PATH,
    ]

    for path in expected_paths:
        assert path.exists(), path


def test_rankings_have_expected_dimensions() -> None:
    oracle = load_oracle_ranking()
    random_ranking = load_random_ranking()

    expected_rows = 157 * 158

    assert len(oracle) == expected_rows
    assert len(random_ranking) == expected_rows

    assert (
        oracle["query_item_id"].nunique()
        == 157
    )

    assert (
        random_ranking[
            "query_item_id"
        ].nunique()
        == 157
    )

    assert oracle.groupby(
        "query_item_id"
    ).size().eq(158).all()

    assert random_ranking.groupby(
        "query_item_id"
    ).size().eq(158).all()


def test_rankings_have_no_self_pairs() -> None:
    oracle = load_oracle_ranking()
    random_ranking = load_random_ranking()

    assert not oracle[
        "query_item_id"
    ].eq(
        oracle["candidate_item_id"]
    ).any()

    assert not random_ranking[
        "query_item_id"
    ].eq(
        random_ranking[
            "candidate_item_id"
        ]
    ).any()


def test_oracle_places_relevant_items_first() -> None:
    ground_truth = load_ground_truth()
    oracle = load_oracle_ranking()

    culture_by_item = (
        ground_truth.set_index("item_id")[
            "culture_canonical"
        ]
        .to_dict()
    )

    relevant_count_by_item = (
        ground_truth.set_index("item_id")[
            "relevant_item_count"
        ]
        .astype(int)
        .to_dict()
    )

    for query_item_id, group in (
        oracle.groupby(
            "query_item_id",
            sort=False,
        )
    ):
        ordered = group.sort_values(
            "rank"
        )

        query_culture = culture_by_item[
            query_item_id
        ]

        relevance = (
            ordered["candidate_item_id"]
            .map(culture_by_item)
            .eq(query_culture)
            .astype(int)
            .tolist()
        )

        relevant_count = (
            relevant_count_by_item[
                query_item_id
            ]
        )

        assert relevance[
            :relevant_count
        ] == [1] * relevant_count

        assert relevance[
            relevant_count:
        ] == [0] * (
            len(relevance)
            - relevant_count
        )


def test_oracle_metrics_match_expected_values() -> None:
    ground_truth = load_ground_truth()

    oracle = build_oracle_ranking(
        ground_truth
    )

    per_query = evaluate_rankings(
        oracle,
        ground_truth,
        k_values=(1, 5, 10),
    )

    overall = aggregate_metrics(
        per_query,
        k_values=(1, 5, 10),
    ).iloc[0]

    assert overall["query_count"] == 157

    assert overall["mrr"] == pytest.approx(
        1.0
    )

    assert overall[
        "precision_at_1"
    ] == pytest.approx(1.0)

    assert overall[
        "recall_at_1"
    ] == pytest.approx(
        0.07083480199598871
    )

    assert overall[
        "ndcg_at_1"
    ] == pytest.approx(1.0)

    assert overall[
        "precision_at_5"
    ] == pytest.approx(
        0.978343949044586
    )

    assert overall[
        "recall_at_5"
    ] == pytest.approx(
        0.27455617558503903
    )

    assert overall[
        "ndcg_at_5"
    ] == pytest.approx(1.0)

    assert overall[
        "precision_at_10"
    ] == pytest.approx(
        0.9350318471337581
    )

    assert overall[
        "recall_at_10"
    ] == pytest.approx(
        0.440832096393008
    )

    assert overall[
        "ndcg_at_10"
    ] == pytest.approx(1.0)


def test_random_ranking_is_deterministic() -> None:
    ground_truth = load_ground_truth()

    first = build_random_ranking(
        ground_truth,
        seed=DEFAULT_RANDOM_SEED,
    )

    second = build_random_ranking(
        ground_truth,
        seed=DEFAULT_RANDOM_SEED,
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )


def test_different_seeds_change_random_ranking() -> None:
    ground_truth = load_ground_truth()

    first = build_random_ranking(
        ground_truth,
        seed=100,
    )

    second = build_random_ranking(
        ground_truth,
        seed=200,
    )

    assert not first[
        "candidate_item_id"
    ].equals(
        second["candidate_item_id"]
    )


def test_query_seed_is_stable() -> None:
    first = stable_query_seed(
        DEFAULT_RANDOM_SEED,
        "MET_123",
    )

    second = stable_query_seed(
        DEFAULT_RANDOM_SEED,
        "MET_123",
    )

    different_query = stable_query_seed(
        DEFAULT_RANDOM_SEED,
        "CMA_123",
    )

    assert first == second
    assert first != different_query


def test_random_ranking_differs_from_oracle() -> None:
    ground_truth = load_ground_truth()

    oracle = build_oracle_ranking(
        ground_truth
    )

    random_ranking = (
        build_random_ranking(
            ground_truth,
            seed=DEFAULT_RANDOM_SEED,
        )
    )

    assert not oracle[
        "candidate_item_id"
    ].equals(
        random_ranking[
            "candidate_item_id"
        ]
    )


def test_random_metrics_are_valid_probabilities() -> None:
    ground_truth = load_ground_truth()

    random_ranking = (
        build_random_ranking(
            ground_truth,
            seed=DEFAULT_RANDOM_SEED,
        )
    )

    per_query = evaluate_rankings(
        random_ranking,
        ground_truth,
        k_values=(1, 5, 10),
    )

    metric_columns = [
        "reciprocal_rank",
        "precision_at_1",
        "recall_at_1",
        "ndcg_at_1",
        "precision_at_5",
        "recall_at_5",
        "ndcg_at_5",
        "precision_at_10",
        "recall_at_10",
        "ndcg_at_10",
    ]

    for column in metric_columns:
        assert per_query[
            column
        ].between(
            0.0,
            1.0,
            inclusive="both",
        ).all()


def test_reference_artifacts_are_deterministic() -> None:
    ground_truth = load_ground_truth()

    first = build_reference_artifacts(
        ground_truth,
        random_seed=DEFAULT_RANDOM_SEED,
        k_values=(1, 5, 10),
    )

    second = build_reference_artifacts(
        ground_truth,
        random_seed=DEFAULT_RANDOM_SEED,
        k_values=(1, 5, 10),
    )

    assert set(first) == set(second)

    for artifact_name in first:
        pd.testing.assert_frame_equal(
            first[artifact_name],
            second[artifact_name],
        )


def test_combined_summary_contains_both_methods() -> None:
    summary = pd.read_csv(
        COMBINED_SUMMARY_PATH
    )

    assert set(
        summary["ranking_method"]
    ) == {
        "oracle_culture",
        "random",
    }

    assert summary["query_count"].eq(
        157
    ).all()