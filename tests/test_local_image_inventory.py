"""Pruebas del inventario y las propuestas de imágenes locales."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from src.metadata.inventory_local_images import (
    INVENTORY_PATH,
    PROPOSALS_PATH,
    SUMMARY_PATH,
    build_image_inventory,
    build_image_match_proposals,
    build_proposal_summary,
    discover_image_paths,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paper_corpus_culture_enriched_v1.csv"
)


def create_image(
    path: Path,
    size: tuple[int, int] = (20, 10),
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.new(
        "RGB",
        size,
    ).save(path)


def test_discovery_excludes_virtual_environment(
    tmp_path: Path,
) -> None:
    valid_path = (
        tmp_path
        / "data"
        / "images"
        / "valid.png"
    )

    excluded_path = (
        tmp_path
        / ".venv"
        / "ignored.png"
    )

    create_image(valid_path)
    create_image(excluded_path)

    discovered = discover_image_paths(
        tmp_path
    )

    assert valid_path.resolve() in discovered
    assert excluded_path.resolve() not in discovered


def test_inventory_detects_readable_image(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "sample.png"

    create_image(
        image_path,
        size=(32, 16),
    )

    inventory = build_image_inventory(
        [image_path],
        tmp_path,
    )

    row = inventory.iloc[0]

    assert bool(row["image_readable"])
    assert row["image_width"] == 32
    assert row["image_height"] == 16
    assert row["sha256"]


def test_inventory_detects_corrupt_image(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "corrupt.jpg"

    image_path.write_text(
        "not an image",
        encoding="utf-8",
    )

    inventory = build_image_inventory(
        [image_path],
        tmp_path,
    )

    row = inventory.iloc[0]

    assert not bool(row["image_readable"])
    assert row["image_error"]


def test_exact_item_id_match_is_unique(
    tmp_path: Path,
) -> None:
    image_path = (
        tmp_path
        / "images"
        / "MET_001.jpg"
    )

    create_image(image_path)

    inventory = build_image_inventory(
        [image_path],
        tmp_path,
    )

    corpus = pd.DataFrame(
        [
            {
                "item_id": "MET_001",
                "museum": "MET",
                "image_url": "",
            }
        ]
    )

    proposals = build_image_match_proposals(
        corpus,
        inventory,
    )

    row = proposals.iloc[0]

    assert (
        row["proposal_status"]
        == "matched_unique"
    )

    assert row["match_score"] == 100

    assert (
        row["match_rule"]
        == "exact_item_id_stem"
    )

    assert (
        row["proposed_image_local_path"]
        == "images/MET_001.jpg"
    )


def test_url_filename_match_is_detected(
    tmp_path: Path,
) -> None:
    image_path = (
        tmp_path
        / "downloads"
        / "object-image.jpg"
    )

    create_image(image_path)

    inventory = build_image_inventory(
        [image_path],
        tmp_path,
    )

    corpus = pd.DataFrame(
        [
            {
                "item_id": "UNRELATED_ID",
                "museum": "CMA",
                "image_url": (
                    "https://example.org/images/"
                    "object-image.jpg?size=large"
                ),
            }
        ]
    )

    proposals = build_image_match_proposals(
        corpus,
        inventory,
    )

    row = proposals.iloc[0]

    assert (
        row["proposal_status"]
        == "matched_unique"
    )

    assert row["match_score"] == 90

    assert (
        row["match_rule"]
        == "exact_url_filename"
    )


def test_duplicate_stems_are_ambiguous(
    tmp_path: Path,
) -> None:
    first_path = (
        tmp_path
        / "folder_a"
        / "MET_001.jpg"
    )

    second_path = (
        tmp_path
        / "folder_b"
        / "MET_001.png"
    )

    create_image(first_path)
    create_image(second_path)

    inventory = build_image_inventory(
        [
            first_path,
            second_path,
        ],
        tmp_path,
    )

    corpus = pd.DataFrame(
        [
            {
                "item_id": "MET_001",
                "museum": "MET",
                "image_url": "",
            }
        ]
    )

    proposals = build_image_match_proposals(
        corpus,
        inventory,
    )

    row = proposals.iloc[0]

    assert (
        row["proposal_status"]
        == "ambiguous"
    )

    assert row["top_candidate_count"] == 2

    assert (
        row["proposed_image_local_path"]
        == ""
    )


def test_unmatched_record_remains_empty(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "OTHER.jpg"

    create_image(image_path)

    inventory = build_image_inventory(
        [image_path],
        tmp_path,
    )

    corpus = pd.DataFrame(
        [
            {
                "item_id": "MET_999",
                "museum": "MET",
                "image_url": "",
            }
        ]
    )

    proposals = build_image_match_proposals(
        corpus,
        inventory,
    )

    row = proposals.iloc[0]

    assert (
        row["proposal_status"]
        == "unmatched"
    )

    assert row["candidate_count"] == 0

    assert (
        row["proposed_image_local_path"]
        == ""
    )


def test_summary_counts_statuses() -> None:
    proposals = pd.DataFrame(
        [
            {
                "museum": "MET",
                "proposal_status": (
                    "matched_unique"
                ),
            },
            {
                "museum": "MET",
                "proposal_status": (
                    "unmatched"
                ),
            },
            {
                "museum": "CMA",
                "proposal_status": (
                    "unmatched"
                ),
            },
        ]
    )

    summary = build_proposal_summary(
        proposals
    )

    all_rows = summary[
        summary["museum"].eq("ALL")
    ]

    observed = all_rows.set_index(
        "proposal_status"
    )["records"].to_dict()

    assert observed == {
        "matched_unique": 1,
        "unmatched": 2,
    }


def test_output_files_exist() -> None:
    assert INVENTORY_PATH.exists()
    assert PROPOSALS_PATH.exists()
    assert SUMMARY_PATH.exists()


def test_current_proposals_preserve_corpus() -> None:
    corpus = pd.read_csv(CORPUS_PATH)
    proposals = pd.read_csv(PROPOSALS_PATH)

    assert len(corpus) == 215
    assert len(proposals) == 215

    assert proposals["item_id"].is_unique

    assert proposals["item_id"].tolist() == (
        corpus["item_id"].tolist()
    )


def test_current_proposal_statuses_are_safe() -> None:
    proposals = pd.read_csv(PROPOSALS_PATH)

    allowed_statuses = {
        "matched_unique",
        "ambiguous",
        "unmatched",
    }

    assert set(
        proposals["proposal_status"]
    ).issubset(allowed_statuses)

    non_unique = proposals[
        ~proposals[
            "proposal_status"
        ].eq("matched_unique")
    ]

    proposed_paths = (
        non_unique[
            "proposed_image_local_path"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    assert proposed_paths.eq("").all()


def test_matching_is_deterministic() -> None:
    corpus = pd.read_csv(CORPUS_PATH)
    inventory = pd.read_csv(INVENTORY_PATH)

    first = build_image_match_proposals(
        corpus,
        inventory,
    )

    second = build_image_match_proposals(
        corpus,
        inventory,
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )