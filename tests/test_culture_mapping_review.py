"""Pruebas de la tabla de revisión del mapeo cultural."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.metadata.prepare_culture_mapping_review import (
    build_review_table,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROPOSALS_PATH = (
    PROJECT_ROOT
    / "config"
    / "mappings"
    / "culture_mapping_proposals_v1.csv"
)

REVIEW_PATH = (
    PROJECT_ROOT
    / "config"
    / "mappings"
    / "culture_mapping_review_v1.csv"
)


def load_proposals() -> pd.DataFrame:
    return pd.read_csv(PROPOSALS_PATH)


def load_review() -> pd.DataFrame:
    return pd.read_csv(REVIEW_PATH)


def test_review_file_exists() -> None:
    assert REVIEW_PATH.exists()


def test_review_preserves_all_proposal_rows() -> None:
    proposals = load_proposals()
    review = load_review()

    assert len(review) == len(proposals)


def test_review_pairs_are_unique() -> None:
    review = load_review()

    assert not review.duplicated(
        subset=["museum", "source_label"]
    ).any()


def test_review_decisions_are_valid() -> None:
    review = load_review()

    allowed_decisions = {
        "include_strict",
        "include_non_strict",
        "exclude",
        "pending",
    }

    assert set(review["review_decision"]).issubset(
        allowed_decisions
    )


def test_only_direct_attributions_are_strictly_eligible() -> None:
    review = load_review()

    strict_rows = review[
        review["strict_ground_truth_eligible"].astype(bool)
    ]

    assert strict_rows["review_decision"].eq(
        "include_strict"
    ).all()

    assert strict_rows["attribution_type"].eq(
        "direct"
    ).all()


def test_non_direct_attributions_are_not_strict_ground_truth() -> None:
    review = load_review()

    non_direct = review[
        review["attribution_type"].isin(
            {
                "style",
                "uncertain",
                "composite",
                "unattributed",
            }
        )
    ]

    assert not non_direct[
        "strict_ground_truth_eligible"
    ].astype(bool).any()


def test_unattributed_rows_have_no_final_components() -> None:
    review = load_review()

    unattributed = review[
        review["attribution_type"].eq("unattributed")
    ]

    values = (
        unattributed["final_canonical_components"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    assert values.eq("").all()


def test_review_build_is_deterministic() -> None:
    proposals = load_proposals()

    first = build_review_table(proposals)
    second = build_review_table(proposals)

    pd.testing.assert_frame_equal(first, second)