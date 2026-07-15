"""Pruebas del baseline aleatorio multisemilla."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.build_random_baseline_distribution import (
    BY_CULTURE_RUNS_PATH,
    BY_CULTURE_SUMMARY_PATH,
    RUNS_PATH,
    SUMMARY_PATH,
    THEORETICAL_PATH,
    build_multiseed_baseline,
    build_theoretical_expectations,
    generate_seeds,
    metric_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "culture_ground_truth_strict_v1.csv"
)


EXPECTED_QUERY_CULTURES = {
    "Paracas",
    "Nasca",
    "Chancay",
    "Inca",
    "Wari",
    "Chimu",
    "Moche",
    "Recuay",
    "Ychsma",
}


def load_ground_truth() -> pd.DataFrame:
    return pd.read_csv(GROUND_TRUTH_PATH)


def test_multiseed_output_files_exist() -> None:
    paths = [
        RUNS_PATH,
        SUMMARY_PATH,
        BY_CULTURE_RUNS_PATH,
        BY_CULTURE_SUMMARY_PATH,
        THEORETICAL_PATH,
    ]

    for path in paths:
        assert path.exists(), path


def test_seed_generation_is_deterministic() -> None:
    first = generate_seeds(
        base_seed=100,
        run_count=4,
    )

    second = generate_seeds(
        base_seed=100,
        run_count=4,
    )

    assert first == second
    assert first == [100, 101, 102, 103]


def test_generated_seeds_are_unique() -> None:
    seeds = generate_seeds(
        base_seed=20260715,
        run_count=100,
    )

    assert len(seeds) == 100
    assert len(set(seeds)) == 100


def test_seed_generation_requires_two_runs() -> None:
    with pytest.raises(
        ValueError,
        match="al menos",
    ):
        generate_seeds(
            base_seed=100,
            run_count=1,
        )


def test_multiseed_artifacts_have_expected_runs() -> None:
    ground_truth = load_ground_truth()

    artifacts = build_multiseed_baseline(
        ground_truth,
        seeds=[10, 20, 30],
        k_values=(1, 5, 10),
    )

    assert len(artifacts["runs"]) == 3

    assert set(
        artifacts["runs"]["generator_seed"]
    ) == {
        10,
        20,
        30,
    }

    assert artifacts["runs"][
        "run_index"
    ].tolist() == [
        1,
        2,
        3,
    ]


def test_each_run_uses_157_queries() -> None:
    ground_truth = load_ground_truth()

    artifacts = build_multiseed_baseline(
        ground_truth,
        seeds=[10, 20],
        k_values=(1, 5, 10),
    )

    assert artifacts["runs"][
        "query_count"
    ].eq(157).all()


def test_run_metrics_are_valid_probabilities() -> None:
    ground_truth = load_ground_truth()

    artifacts = build_multiseed_baseline(
        ground_truth,
        seeds=[10, 20],
        k_values=(1, 5, 10),
    )

    metrics = metric_columns(
        (1, 5, 10)
    )

    for metric in metrics:
        assert artifacts["runs"][
            metric
        ].between(
            0.0,
            1.0,
            inclusive="both",
        ).all()


def test_summary_contains_all_metrics() -> None:
    ground_truth = load_ground_truth()

    artifacts = build_multiseed_baseline(
        ground_truth,
        seeds=[10, 20, 30],
        k_values=(1, 5, 10),
    )

    expected_metrics = set(
        metric_columns((1, 5, 10))
    )

    assert set(
        artifacts["summary"]["metric"]
    ) == expected_metrics

    assert artifacts["summary"][
        "run_count"
    ].eq(3).all()


def test_by_culture_results_cover_eligible_cultures() -> None:
    ground_truth = load_ground_truth()

    artifacts = build_multiseed_baseline(
        ground_truth,
        seeds=[10, 20],
        k_values=(1, 5, 10),
    )

    assert set(
        artifacts[
            "by_culture_runs"
        ]["culture_canonical"]
    ) == EXPECTED_QUERY_CULTURES


def test_theoretical_precision_is_correct() -> None:
    ground_truth = load_ground_truth()

    theoretical = build_theoretical_expectations(
        ground_truth,
        k_values=(1, 5, 10),
    )

    precision_rows = theoretical[
        theoretical["metric"].str.startswith(
            "precision"
        )
    ]

    assert precision_rows[
        "expected_value"
    ].tolist() == pytest.approx(
        [
            0.17705393856325083,
            0.17705393856325083,
            0.17705393856325083,
        ]
    )


def test_theoretical_recall_is_correct() -> None:
    ground_truth = load_ground_truth()

    theoretical = build_theoretical_expectations(
        ground_truth,
        k_values=(1, 5, 10),
    )

    expected = {
        "recall_at_1": 1 / 158,
        "recall_at_5": 5 / 158,
        "recall_at_10": 10 / 158,
    }

    observed = theoretical.set_index(
        "metric"
    )["expected_value"].to_dict()

    for metric, expected_value in expected.items():
        assert observed[metric] == pytest.approx(
            expected_value
        )


def test_multiseed_build_is_deterministic() -> None:
    ground_truth = load_ground_truth()

    first = build_multiseed_baseline(
        ground_truth,
        seeds=[10, 20],
        k_values=(1, 5, 10),
    )

    second = build_multiseed_baseline(
        ground_truth,
        seeds=[10, 20],
        k_values=(1, 5, 10),
    )

    assert set(first) == set(second)

    for artifact_name in first:
        pd.testing.assert_frame_equal(
            first[artifact_name],
            second[artifact_name],
        )