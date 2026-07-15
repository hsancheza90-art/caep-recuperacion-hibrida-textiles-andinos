"""Pruebas de construcción de entradas textuales para OpenCLIP."""

from __future__ import annotations

import re

import pandas as pd
import pytest

from src.openclip_baseline.text_inputs import (
    METADATA_TEXT_VERSION,
    VISUAL_TEXT_VERSION,
    build_openclip_text_inputs,
)


def make_corpus() -> pd.DataFrame:
    """Crea un corpus mínimo con metadatos completos e incompletos."""

    return pd.DataFrame(
        [
            {
                "item_id": "MET:2",
                "museum": "MET",
                "title": "  Textile   fragment  ",
                "object_type": "Textile fragment",
                "material": "Cotton, camelid fiber",
                "technique": "Tapestry weave",
                "description": (
                    "Fragment with geometric motifs.\n"
                    "  Red and yellow bands."
                ),
                "culture": "Wari",
                "culture_canonical": "Wari",
                "period": "Middle Horizon",
                "classification": "Textiles",
                "country": "Peru",
                "region": "Central Andes",
            },
            {
                "item_id": "CMA:1",
                "museum": "CMA",
                "title": "Mantle",
                "object_type": "Mantle",
                "material": "Cotton",
                "technique": "",
                "description": "",
                "culture": "Paracas",
                "culture_canonical": "Paracas",
                "period": "Early Intermediate Period",
                "classification": "Textiles",
                "country": "Peru",
                "region": "",
            },
        ]
    )


def test_text_inputs_preserve_rows_and_order() -> None:
    corpus = make_corpus()

    result = build_openclip_text_inputs(corpus)

    assert len(result) == len(corpus)
    assert result["item_id"].tolist() == [
        "MET:2",
        "CMA:1",
    ]
    assert result["museum"].tolist() == [
        "MET",
        "CMA",
    ]


def test_visual_text_excludes_ground_truth_metadata() -> None:
    result = build_openclip_text_inputs(
        make_corpus()
    )

    text = result.loc[
        result["item_id"].eq("MET:2"),
        "text_visual",
    ].iloc[0]

    assert "Title: Textile fragment" in text
    assert "Object type: Textile fragment" in text
    assert "Material: Cotton, camelid fiber" in text
    assert "Technique: Tapestry weave" in text
    assert (
        "Description: Fragment with geometric motifs. "
        "Red and yellow bands."
    ) in text

    forbidden_values = [
        "Wari",
        "Middle Horizon",
        "Textiles",
        "Central Andes",
    ]

    for value in forbidden_values:
        assert value not in text

    forbidden_labels = [
        "Culture:",
        "Period:",
        "Classification:",
        "Country:",
        "Region:",
    ]

    for label in forbidden_labels:
        assert label not in text


def test_metadata_text_includes_curatorial_metadata() -> None:
    result = build_openclip_text_inputs(
        make_corpus()
    )

    text = result.loc[
        result["item_id"].eq("MET:2"),
        "text_metadata",
    ].iloc[0]

    assert "Culture: Wari" in text
    assert "Period: Middle Horizon" in text
    assert "Classification: Textiles" in text
    assert "Country: Peru" in text
    assert "Region: Central Andes" in text


def test_text_inputs_skip_empty_fields() -> None:
    result = build_openclip_text_inputs(
        make_corpus()
    )

    visual_text = result.loc[
        result["item_id"].eq("CMA:1"),
        "text_visual",
    ].iloc[0]

    metadata_text = result.loc[
        result["item_id"].eq("CMA:1"),
        "text_metadata",
    ].iloc[0]

    assert "Technique:" not in visual_text
    assert "Description:" not in visual_text
    assert "Region:" not in metadata_text
    assert ";;" not in visual_text
    assert not visual_text.endswith(";")


def test_text_inputs_normalize_whitespace() -> None:
    result = build_openclip_text_inputs(
        make_corpus()
    )

    text = result.loc[
        result["item_id"].eq("MET:2"),
        "text_visual",
    ].iloc[0]

    assert "\n" not in text
    assert "\t" not in text
    assert "  " not in text
    assert not re.search(r"\s+;", text)
    assert not re.search(r";\s*;", text)


def test_text_inputs_include_stable_versions_and_hashes() -> None:
    first = build_openclip_text_inputs(
        make_corpus()
    )

    second = build_openclip_text_inputs(
        make_corpus()
    )

    assert first["text_visual_version"].eq(
        VISUAL_TEXT_VERSION
    ).all()

    assert first["text_metadata_version"].eq(
        METADATA_TEXT_VERSION
    ).all()

    assert (
        first["text_visual_sha256"]
        .str.fullmatch(r"[0-9a-f]{64}")
        .all()
    )

    assert (
        first["text_metadata_sha256"]
        .str.fullmatch(r"[0-9a-f]{64}")
        .all()
    )

    assert (
        first["text_visual_sha256"].tolist()
        == second["text_visual_sha256"].tolist()
    )

    assert (
        first["text_metadata_sha256"].tolist()
        == second["text_metadata_sha256"].tolist()
    )


def test_text_inputs_do_not_modify_source() -> None:
    corpus = make_corpus()
    corpus_before = corpus.copy(deep=True)

    build_openclip_text_inputs(corpus)

    pd.testing.assert_frame_equal(
        corpus,
        corpus_before,
    )


def test_text_inputs_require_item_id() -> None:
    corpus = make_corpus().drop(
        columns=["item_id"]
    )

    with pytest.raises(
        ValueError,
        match="item_id",
    ):
        build_openclip_text_inputs(corpus)