"""Pruebas de la auditoría de entradas OpenCLIP."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from src.evaluation.audit_openclip_inputs import (
    AUDIT_PATH,
    SUMMARY_PATH,
    build_openclip_input_audit,
    build_summary,
    build_text_input,
    detect_image_url_columns,
    detect_local_image_columns,
    detect_text_columns,
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


def load_corpus() -> pd.DataFrame:
    return pd.read_csv(CORPUS_PATH)


def load_ground_truth() -> pd.DataFrame:
    return pd.read_csv(GROUND_TRUTH_PATH)


def load_audit() -> pd.DataFrame:
    return pd.read_csv(AUDIT_PATH)


def test_column_detection() -> None:
    dataframe = pd.DataFrame(
        {
            "item_id": ["A"],
            "local_image_path": ["images/a.jpg"],
            "image_url": ["https://example.org/a.jpg"],
            "title": ["Objeto A"],
            "culture": ["Paracas"],
        }
    )

    assert detect_local_image_columns(
        dataframe
    ) == ["local_image_path"]

    assert detect_image_url_columns(
        dataframe
    ) == ["image_url"]

    assert detect_text_columns(
        dataframe
    ) == [
        "title",
        "culture",
    ]


def test_text_input_removes_duplicate_values() -> None:
    row = pd.Series(
        {
            "title": "Mantle",
            "object_name": "Mantle",
            "culture": "Paracas",
        }
    )

    text, used_fields = build_text_input(
        row,
        [
            "title",
            "object_name",
            "culture",
        ],
    )

    assert text == (
        "title: Mantle. culture: Paracas"
    )

    assert used_fields == "title | culture"


def test_readable_image_is_detected(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "valid.png"

    Image.new(
        "RGB",
        (20, 10),
    ).save(image_path)

    corpus = pd.DataFrame(
        [
            {
                "item_id": "A1",
                "museum": "MET",
                "culture_mapping_decision": (
                    "include_strict"
                ),
                "culture_strict_ground_truth_eligible": (
                    True
                ),
                "local_image_path": str(
                    image_path
                ),
                "title": "Textile",
            }
        ]
    )

    ground_truth = pd.DataFrame(
        [
            {
                "item_id": "A1",
                "query_eligible": True,
            }
        ]
    )

    audit, _ = build_openclip_input_audit(
        corpus,
        ground_truth,
    )

    row = audit.iloc[0]

    assert bool(row["local_image_exists"])
    assert bool(row["local_image_readable"])
    assert row["image_width"] == 20
    assert row["image_height"] == 10
    assert bool(row["openclip_image_ready"])
    assert bool(
        row[
            "openclip_multimodal_evaluation_ready"
        ]
    )


def test_corrupt_image_is_not_readable(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "corrupt.jpg"

    image_path.write_text(
        "not an image",
        encoding="utf-8",
    )

    corpus = pd.DataFrame(
        [
            {
                "item_id": "A1",
                "museum": "MET",
                "culture_mapping_decision": (
                    "include_strict"
                ),
                "culture_strict_ground_truth_eligible": (
                    True
                ),
                "image_path": str(image_path),
                "title": "Textile",
            }
        ]
    )

    ground_truth = pd.DataFrame(
        [
            {
                "item_id": "A1",
                "query_eligible": True,
            }
        ]
    )

    audit, _ = build_openclip_input_audit(
        corpus,
        ground_truth,
    )

    row = audit.iloc[0]

    assert bool(
        row["local_image_exists"]
    )

    assert not bool(
        row["local_image_readable"]
    )

    assert not bool(
        row["openclip_image_ready"]
    )

    assert row["image_error"]

def test_audit_files_exist() -> None:
    assert AUDIT_PATH.exists()
    assert SUMMARY_PATH.exists()


def test_audit_preserves_all_records() -> None:
    corpus = load_corpus()
    audit = load_audit()

    assert len(corpus) == 215
    assert len(audit) == 215
    assert audit["item_id"].is_unique


def test_audit_preserves_museum_distribution() -> None:
    audit = load_audit()

    assert (
        audit["museum"]
        .value_counts()
        .to_dict()
    ) == {
        "MET": 127,
        "CMA": 88,
    }


def test_audit_has_157_query_eligible_records() -> None:
    audit = load_audit()

    query_eligible = (
        audit["culture_query_eligible"]
        .astype(bool)
    )

    assert int(query_eligible.sum()) == 157


def test_ready_statuses_are_consistent() -> None:
    audit = load_audit()

    expected_multimodal = (
        audit["openclip_image_ready"].astype(bool)
        & audit["openclip_text_ready"].astype(bool)
    )

    assert (
        audit[
            "openclip_multimodal_ready"
        ].astype(bool)
        == expected_multimodal
    ).all()

    expected_evaluation = (
        expected_multimodal
        & audit[
            "culture_query_eligible"
        ].astype(bool)
    )

    assert (
        audit[
            "openclip_multimodal_evaluation_ready"
        ].astype(bool)
        == expected_evaluation
    ).all()


def test_summary_all_row_matches_audit() -> None:
    audit = load_audit()
    summary = pd.read_csv(SUMMARY_PATH)

    all_row = summary[
        summary["museum"].eq("ALL")
    ].iloc[0]

    assert all_row["records"] == len(audit)

    assert (
        all_row["query_eligible_records"]
        == int(
            audit[
                "culture_query_eligible"
            ].astype(bool).sum()
        )
    )

    assert (
        all_row["readable_local_images"]
        == int(
            audit[
                "local_image_readable"
            ].astype(bool).sum()
        )
    )


def test_audit_build_is_deterministic() -> None:
    corpus = load_corpus()
    ground_truth = load_ground_truth()

    first_audit, first_metadata = (
        build_openclip_input_audit(
            corpus,
            ground_truth,
        )
    )

    second_audit, second_metadata = (
        build_openclip_input_audit(
            corpus,
            ground_truth,
        )
    )

    pd.testing.assert_frame_equal(
        first_audit,
        second_audit,
    )

    assert first_metadata == second_metadata

    pd.testing.assert_frame_equal(
        build_summary(first_audit),
        build_summary(second_audit),
    )