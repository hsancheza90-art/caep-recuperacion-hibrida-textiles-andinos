"""Pruebas del ground truth cultural estricto."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.build_culture_ground_truth import (
    build_culture_ground_truth,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paper_corpus_culture_enriched_v1.csv"
)

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "culture_ground_truth_strict_v1.csv"
)

PAIRS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "culture_relevance_pairs_strict_v1.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "culture_ground_truth_summary_v1.csv"
)


EXPECTED_DISTRIBUTION = {
    "Paracas": 42,
    "Nasca": 35,
    "Chancay": 31,
    "Inca": 17,
    "Wari": 15,
    "Chimu": 6,
    "Moche": 6,
    "Recuay": 3,
    "Ychsma": 2,
    "Chuquibamba": 1,
    "Lambayeque_Sican": 1,
}


def load_corpus() -> pd.DataFrame:
    return pd.read_csv(CORPUS_PATH)


def load_ground_truth() -> pd.DataFrame:
    return pd.read_csv(GROUND_TRUTH_PATH)


def load_pairs() -> pd.DataFrame:
    return pd.read_csv(PAIRS_PATH)


def clean_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
    )


def test_ground_truth_files_exist() -> None:
    assert GROUND_TRUTH_PATH.exists()
    assert PAIRS_PATH.exists()
    assert SUMMARY_PATH.exists()


def test_ground_truth_has_159_strict_records() -> None:
    ground_truth = load_ground_truth()

    assert len(ground_truth) == 159
    assert ground_truth["item_id"].is_unique


def test_ground_truth_distribution_is_correct() -> None:
    ground_truth = load_ground_truth()

    distribution = (
        ground_truth["culture_canonical"]
        .value_counts()
        .to_dict()
    )

    assert distribution == EXPECTED_DISTRIBUTION


def test_only_strict_corpus_records_are_in_ground_truth() -> None:
    corpus = load_corpus()
    ground_truth = load_ground_truth()

    strict_ids = set(
        corpus.loc[
            corpus[
                "culture_mapping_decision"
            ].eq("include_strict"),
            "item_id",
        ]
    )

    ground_truth_ids = set(
        ground_truth["item_id"]
    )

    assert ground_truth_ids == strict_ids


def test_group_sizes_match_distribution() -> None:
    ground_truth = load_ground_truth()

    for culture, expected_size in (
        EXPECTED_DISTRIBUTION.items()
    ):
        rows = ground_truth[
            ground_truth[
                "culture_canonical"
            ].eq(culture)
        ]

        assert rows[
            "culture_group_size"
        ].eq(expected_size).all()

        assert rows[
            "relevant_item_count"
        ].eq(expected_size - 1).all()


def test_157_records_are_query_eligible() -> None:
    ground_truth = load_ground_truth()

    query_eligible = (
        ground_truth["query_eligible"]
        .astype(bool)
    )

    assert int(query_eligible.sum()) == 157
    assert int((~query_eligible).sum()) == 2


def test_singleton_cultures_are_not_query_eligible() -> None:
    ground_truth = load_ground_truth()

    singleton = ground_truth[
        ground_truth[
            "culture_group_size"
        ].eq(1)
    ]

    assert set(
        singleton["culture_canonical"]
    ) == {
        "Chuquibamba",
        "Lambayeque_Sican",
    }

    assert not singleton[
        "query_eligible"
    ].astype(bool).any()

    assert singleton[
        "relevant_item_count"
    ].eq(0).all()


def test_relevance_pairs_have_expected_size() -> None:
    pairs = load_pairs()

    assert len(pairs) == 4392


def test_relevance_pairs_are_unique_and_not_self_pairs() -> None:
    pairs = load_pairs()

    assert not pairs.duplicated(
        subset=[
            "query_item_id",
            "relevant_item_id",
        ]
    ).any()

    assert not pairs[
        "query_item_id"
    ].eq(
        pairs["relevant_item_id"]
    ).any()


def test_pair_items_share_the_same_culture() -> None:
    ground_truth = load_ground_truth()
    pairs = load_pairs()

    item_culture = (
        ground_truth.set_index("item_id")[
            "culture_canonical"
        ]
        .to_dict()
    )

    query_culture = pairs[
        "query_item_id"
    ].map(item_culture)

    relevant_culture = pairs[
        "relevant_item_id"
    ].map(item_culture)

    assert query_culture.notna().all()
    assert relevant_culture.notna().all()

    assert query_culture.equals(
        relevant_culture
    )

    assert clean_text(
        pairs["culture_canonical"]
    ).equals(
        clean_text(query_culture)
    )


def test_pair_count_per_query_matches_ground_truth() -> None:
    ground_truth = load_ground_truth()
    pairs = load_pairs()

    actual_counts = (
        pairs.groupby("query_item_id")
        .size()
        .to_dict()
    )

    for row in ground_truth.itertuples():
        actual = actual_counts.get(
            row.item_id,
            0,
        )

        assert actual == row.relevant_item_count


def test_ground_truth_build_is_deterministic() -> None:
    corpus = load_corpus()

    first_ground_truth, first_pairs, first_summary = (
        build_culture_ground_truth(corpus)
    )

    second_ground_truth, second_pairs, second_summary = (
        build_culture_ground_truth(corpus)
    )

    pd.testing.assert_frame_equal(
        first_ground_truth,
        second_ground_truth,
    )

    pd.testing.assert_frame_equal(
        first_pairs,
        second_pairs,
    )

    pd.testing.assert_frame_equal(
        first_summary,
        second_summary,
    )