"""Pruebas del constructor del corpus multimodal derivado."""

from __future__ import annotations

import pandas as pd
import pytest

from src.corpus.build_multimodal_corpus import (
    build_multimodal_corpus,
)


def make_cultural_corpus() -> pd.DataFrame:
    """Crea un corpus cultural mínimo para las pruebas."""

    return pd.DataFrame(
        [
            {
                "item_id": "MET:2",
                "museum": "MET",
                "title": "Textile fragment",
                "culture_normalized": "Wari",
            },
            {
                "item_id": "CMA:1",
                "museum": "CMA",
                "title": "Mantle",
                "culture_normalized": "Paracas",
            },
        ]
    )


def make_image_manifest() -> pd.DataFrame:
    """Crea un manifiesto técnico mínimo para las pruebas."""

    return pd.DataFrame(
        [
            {
                "item_id": "CMA:1",
                "museum": "CMA",
                "image_local_path": (
                    "data/images/cma/CMA_1.jpg"
                ),
                "download_status": "available",
                "acquisition_action": "reused",
                "final_url": "https://example.org/cma.jpg",
                "content_type": "image/jpeg",
                "image_bytes": "200",
                "image_width": "20",
                "image_height": "10",
                "image_format": "JPEG",
                "sha256": "b" * 64,
                "download_version": "image_download_v1",
            },
            {
                "item_id": "MET:2",
                "museum": "MET",
                "image_local_path": (
                    "data/images/met/MET_2.jpg"
                ),
                "download_status": "available",
                "acquisition_action": "downloaded",
                "final_url": "https://example.org/met.jpg",
                "content_type": "image/jpeg",
                "image_bytes": "100",
                "image_width": "10",
                "image_height": "20",
                "image_format": "JPEG",
                "sha256": "a" * 64,
                "download_version": "image_download_v1",
            },
        ]
    )


def test_build_multimodal_corpus_preserves_rows_and_order() -> None:
    cultural = make_cultural_corpus()
    manifest = make_image_manifest()

    result = build_multimodal_corpus(
        cultural,
        manifest,
    )

    assert len(result) == len(cultural)
    assert result["item_id"].tolist() == [
        "MET:2",
        "CMA:1",
    ]
    assert result["title"].tolist() == [
        "Textile fragment",
        "Mantle",
    ]


def test_build_multimodal_corpus_adds_explicit_image_columns() -> None:
    result = build_multimodal_corpus(
        make_cultural_corpus(),
        make_image_manifest(),
    )

    expected_columns = {
        "image_local_path",
        "image_download_status",
        "image_acquisition_action",
        "image_final_url",
        "image_content_type",
        "image_bytes",
        "image_width",
        "image_height",
        "image_format",
        "image_sha256",
        "image_download_version",
    }

    assert expected_columns.issubset(result.columns)
    assert not any(
        column.endswith(("_x", "_y"))
        for column in result.columns
    )

    met_row = result.loc[
        result["item_id"].eq("MET:2")
    ].iloc[0]

    assert (
        met_row["image_local_path"]
        == "data/images/met/MET_2.jpg"
    )
    assert (
        met_row["image_download_status"]
        == "available"
    )
    assert met_row["image_sha256"] == "a" * 64


def test_build_multimodal_corpus_does_not_modify_inputs() -> None:
    cultural = make_cultural_corpus()
    manifest = make_image_manifest()

    cultural_before = cultural.copy(deep=True)
    manifest_before = manifest.copy(deep=True)

    build_multimodal_corpus(
        cultural,
        manifest,
    )

    pd.testing.assert_frame_equal(
        cultural,
        cultural_before,
    )
    pd.testing.assert_frame_equal(
        manifest,
        manifest_before,
    )


def test_build_multimodal_corpus_rejects_duplicate_corpus_ids() -> None:
    cultural = make_cultural_corpus()
    cultural.loc[1, "item_id"] = "MET:2"

    with pytest.raises(
        ValueError,
        match="corpus cultural.*duplicad",
    ):
        build_multimodal_corpus(
            cultural,
            make_image_manifest(),
        )


def test_build_multimodal_corpus_rejects_duplicate_manifest_ids() -> None:
    manifest = make_image_manifest()
    manifest.loc[1, "item_id"] = "CMA:1"

    with pytest.raises(
        ValueError,
        match="manifiesto.*duplicad",
    ):
        build_multimodal_corpus(
            make_cultural_corpus(),
            manifest,
        )


def test_build_multimodal_corpus_requires_complete_coverage() -> None:
    manifest = make_image_manifest().iloc[[0]].copy()

    with pytest.raises(
        ValueError,
        match="cobertura incompleta",
    ):
        build_multimodal_corpus(
            make_cultural_corpus(),
            manifest,
        )


@pytest.mark.parametrize(
    "invalid_path",
    [
        r"D:\datos\images\MET_2.jpg",
        "../images/MET_2.jpg",
    ],
)
def test_build_multimodal_corpus_rejects_nonportable_paths(
    invalid_path: str,
) -> None:
    manifest = make_image_manifest()
    manifest.loc[
        manifest["item_id"].eq("MET:2"),
        "image_local_path",
    ] = invalid_path

    with pytest.raises(
        ValueError,
        match="ruta.*portable",
    ):
        build_multimodal_corpus(
            make_cultural_corpus(),
            manifest,
        )

def test_build_multimodal_corpus_fills_existing_empty_columns() -> None:
    cultural = make_cultural_corpus()

    cultural["image_local_path"] = ""
    cultural["image_sha256"] = ""

    result = build_multimodal_corpus(
        cultural,
        make_image_manifest(),
    )

    assert result.columns.tolist().count(
        "image_local_path"
    ) == 1

    assert result.columns.tolist().count(
        "image_sha256"
    ) == 1

    assert result["image_local_path"].tolist() == [
        "data/images/met/MET_2.jpg",
        "data/images/cma/CMA_1.jpg",
    ]

    assert result["image_sha256"].tolist() == [
        "a" * 64,
        "b" * 64,
    ]


def test_build_multimodal_corpus_accepts_compatible_existing_values(
) -> None:
    cultural = make_cultural_corpus()

    cultural["image_local_path"] = [
        "data/images/met/MET_2.jpg",
        "data/images/cma/CMA_1.jpg",
    ]

    cultural["image_sha256"] = [
        "A" * 64,
        "B" * 64,
    ]

    result = build_multimodal_corpus(
        cultural,
        make_image_manifest(),
    )

    assert result["image_local_path"].tolist() == [
        "data/images/met/MET_2.jpg",
        "data/images/cma/CMA_1.jpg",
    ]

    # El manifiesto se utiliza como representación canónica.
    assert result["image_sha256"].tolist() == [
        "a" * 64,
        "b" * 64,
    ]


@pytest.mark.parametrize(
    (
        "column",
        "conflicting_value",
    ),
    [
        (
            "image_local_path",
            "data/images/met/OTHER_IMAGE.jpg",
        ),
        (
            "image_sha256",
            "c" * 64,
        ),
    ],
)
def test_build_multimodal_corpus_rejects_existing_conflicts(
    column: str,
    conflicting_value: str,
) -> None:
    cultural = make_cultural_corpus()

    cultural["image_local_path"] = ""
    cultural["image_sha256"] = ""

    cultural.loc[
        cultural["item_id"].eq("MET:2"),
        column,
    ] = conflicting_value

    with pytest.raises(
        ValueError,
        match=(
            "conflicto.*"
            f"{column}"
        ),
    ):
        build_multimodal_corpus(
            cultural,
            make_image_manifest(),
        )