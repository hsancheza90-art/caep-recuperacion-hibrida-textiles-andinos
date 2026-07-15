"""Pruebas del corpus enriquecido culturalmente."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.metadata.apply_culture_mapping import (
    build_culture_enriched_corpus,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paper_corpus_enriched_v1.csv"
)

MAPPING_PATH = (
    PROJECT_ROOT
    / "config"
    / "mappings"
    / "culture_mapping_curated_v1.csv"
)

OUTPUT_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paper_corpus_culture_enriched_v1.csv"
)

OUTPUT_PARQUET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paper_corpus_culture_enriched_v1.parquet"
)


CULTURE_DERIVED_COLUMNS = {
    "culture_source",
    "culture_canonical",
    "culture_components",
    "culture_component_count",
    "culture_attribution_type",
    "culture_mapping_decision",
    "culture_mapping_basis",
    "culture_strict_ground_truth_eligible",
    "culture_mapping_status",
    "culture_mapping_note",
    "culture_mapping_version",
}


def load_base() -> pd.DataFrame:
    return pd.read_csv(BASE_CORPUS_PATH)


def load_mapping() -> pd.DataFrame:
    return pd.read_csv(MAPPING_PATH)


def load_enriched() -> pd.DataFrame:
    return pd.read_csv(OUTPUT_CSV_PATH)


def clean_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
    )


def parse_bool(series: pd.Series) -> pd.Series:
    mapping = {
        "true": True,
        "1": True,
        "false": False,
        "0": False,
    }

    parsed = (
        clean_text(series)
        .str.lower()
        .map(mapping)
    )

    assert not parsed.isna().any()

    return parsed.astype(bool)


def test_culture_enriched_files_exist() -> None:
    assert OUTPUT_CSV_PATH.exists()
    assert OUTPUT_PARQUET_PATH.exists()


def test_culture_enriched_corpus_preserves_size() -> None:
    base = load_base()
    enriched = load_enriched()

    assert len(base) == 215
    assert len(enriched) == 215

    assert len(enriched.columns) == (
        len(base.columns)
        + len(CULTURE_DERIVED_COLUMNS)
    )

    assert len(enriched.columns) == 53


def test_culture_enriched_corpus_preserves_museum_distribution() -> None:
    enriched = load_enriched()

    distribution = (
        enriched["museum"]
        .value_counts()
        .to_dict()
    )

    assert distribution == {
        "MET": 127,
        "CMA": 88,
    }


def test_item_ids_remain_unique_and_ordered() -> None:
    base = load_base()
    enriched = load_enriched()

    assert enriched["item_id"].is_unique

    assert enriched["item_id"].tolist() == (
        base["item_id"].tolist()
    )


def test_original_columns_are_preserved() -> None:
    base = load_base()
    enriched = load_enriched()

    for column in base.columns:
        pd.testing.assert_series_equal(
            enriched[column],
            base[column],
            check_dtype=False,
            check_names=True,
        )


def test_culture_source_preserves_original_label() -> None:
    enriched = load_enriched()

    assert clean_text(
        enriched["culture_source"]
    ).equals(
        clean_text(enriched["culture"])
    )


def test_all_records_receive_a_resolved_mapping() -> None:
    enriched = load_enriched()

    for column in [
        "culture_source",
        "culture_attribution_type",
        "culture_mapping_decision",
        "culture_mapping_basis",
        "culture_mapping_status",
        "culture_mapping_note",
        "culture_mapping_version",
    ]:
        assert clean_text(
            enriched[column]
        ).ne("").all()

    assert clean_text(
        enriched["culture_mapping_status"]
    ).eq("resolved").all()


def test_mapping_decision_distribution_is_correct() -> None:
    enriched = load_enriched()

    distribution = (
        enriched[
            "culture_mapping_decision"
        ]
        .value_counts()
        .to_dict()
    )

    assert distribution == {
        "include_strict": 159,
        "include_non_strict": 43,
        "exclude": 13,
    }


def test_strict_rows_are_direct_and_single_component() -> None:
    enriched = load_enriched()

    strict = enriched[
        enriched[
            "culture_mapping_decision"
        ].eq("include_strict")
    ]

    eligible = parse_bool(
        strict[
            "culture_strict_ground_truth_eligible"
        ]
    )

    assert eligible.all()

    assert strict[
        "culture_attribution_type"
    ].eq("direct").all()

    assert strict[
        "culture_component_count"
    ].eq(1).all()

    assert clean_text(
        strict["culture_canonical"]
    ).equals(
        clean_text(
            strict["culture_components"]
        )
    )


def test_non_strict_rows_are_not_strict_ground_truth() -> None:
    enriched = load_enriched()

    non_strict = enriched[
        enriched[
            "culture_mapping_decision"
        ].eq("include_non_strict")
    ]

    eligible = parse_bool(
        non_strict[
            "culture_strict_ground_truth_eligible"
        ]
    )

    assert not eligible.any()

    assert set(
        non_strict[
            "culture_attribution_type"
        ]
    ) == {
        "style",
        "composite",
        "uncertain",
    }


def test_composite_rows_preserve_all_components() -> None:
    enriched = load_enriched()

    composite = enriched[
        enriched[
            "culture_attribution_type"
        ].eq("composite")
    ]

    assert composite[
        "culture_component_count"
    ].ge(2).all()

    assert clean_text(
        composite["culture_components"]
    ).str.contains(
        "|",
        regex=False,
    ).all()

    assert clean_text(
        composite["culture_canonical"]
    ).eq("").all()


def test_excluded_rows_have_no_canonical_culture() -> None:
    enriched = load_enriched()

    excluded = enriched[
        enriched[
            "culture_mapping_decision"
        ].eq("exclude")
    ]

    assert len(excluded) == 13

    assert clean_text(
        excluded["culture_components"]
    ).eq("").all()

    assert clean_text(
        excluded["culture_canonical"]
    ).eq("").all()

    assert excluded[
        "culture_component_count"
    ].eq(0).all()

    assert not parse_bool(
        excluded[
            "culture_strict_ground_truth_eligible"
        ]
    ).any()


def test_culture_enrichment_build_is_deterministic() -> None:
    base = load_base()
    mapping = load_mapping()

    first = build_culture_enriched_corpus(
        base,
        mapping,
    )

    second = build_culture_enriched_corpus(
        base,
        mapping,
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )