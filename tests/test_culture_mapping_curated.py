"""Pruebas del mapeo cultural consolidado."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.metadata.finalize_culture_mapping_review import (
    build_curated_mapping,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REVIEW_PATH = (
    PROJECT_ROOT
    / "config"
    / "mappings"
    / "culture_mapping_review_v1.csv"
)

CURATED_PATH = (
    PROJECT_ROOT
    / "config"
    / "mappings"
    / "culture_mapping_curated_v1.csv"
)


def load_review() -> pd.DataFrame:
    return pd.read_csv(REVIEW_PATH)


def load_curated() -> pd.DataFrame:
    return pd.read_csv(CURATED_PATH)


def text_values(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
    )


def test_curated_mapping_file_exists() -> None:
    assert CURATED_PATH.exists()


def test_curated_mapping_preserves_all_source_categories() -> None:
    review = load_review()
    curated = load_curated()

    assert len(curated) == len(review)
    assert len(curated) == 57


def test_curated_mapping_has_unique_source_pairs() -> None:
    curated = load_curated()

    assert not curated.duplicated(
        subset=["museum", "source_label"]
    ).any()


def test_curated_mapping_has_no_pending_decisions() -> None:
    curated = load_curated()

    assert "pending" not in set(
        text_values(curated["review_decision"])
    )

    assert text_values(
        curated["review_status"]
    ).eq("resolved").all()


def test_strict_rows_are_direct_attributions() -> None:
    curated = load_curated()

    strict_rows = curated[
        curated["review_decision"].eq("include_strict")
    ]

    assert strict_rows["attribution_type"].eq("direct").all()
    assert strict_rows[
        "strict_ground_truth_eligible"
    ].astype(bool).all()


def test_non_strict_rows_preserve_non_direct_attributions() -> None:
    curated = load_curated()

    non_strict = curated[
        curated["review_decision"].eq("include_non_strict")
    ]

    assert set(non_strict["attribution_type"]) == {
        "style",
        "composite",
        "uncertain",
    }

    assert not non_strict[
        "strict_ground_truth_eligible"
    ].astype(bool).any()

    assert text_values(
        non_strict["final_canonical_components"]
    ).ne("").all()


def test_composite_rows_preserve_multiple_components() -> None:
    curated = load_curated()

    composite = curated[
        curated["attribution_type"].eq("composite")
    ]

    assert text_values(
        composite["final_canonical_components"]
    ).str.contains(
        r"\|",
        regex=True,
    ).all()


def test_excluded_rows_have_no_final_components() -> None:
    curated = load_curated()

    excluded = curated[
        curated["review_decision"].eq("exclude")
    ]

    assert excluded["attribution_type"].eq(
        "unattributed"
    ).all()

    assert text_values(
        excluded["final_canonical_components"]
    ).eq("").all()

    assert not excluded[
        "strict_ground_truth_eligible"
    ].astype(bool).any()


def test_curated_mapping_has_expected_record_distribution() -> None:
    curated = load_curated()

    distribution = (
        curated.groupby("review_decision")["record_count"]
        .sum()
        .to_dict()
    )

    assert distribution == {
        "exclude": 13,
        "include_non_strict": 43,
        "include_strict": 159,
    }

    assert int(curated["record_count"].sum()) == 215


def test_curated_mapping_build_is_deterministic() -> None:
    review = load_review()

    first = build_curated_mapping(review)
    second = build_curated_mapping(review)

    pd.testing.assert_frame_equal(first, second)