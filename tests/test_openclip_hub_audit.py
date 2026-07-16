"""Pruebas de la auditoría enriquecida de hubs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.openclip_baseline.hub_audit import (
    build_openclip_hub_audit,
)


def _build_corpus(
    tmp_path: Path,
) -> pd.DataFrame:
    """Construye un corpus sintético con imágenes locales."""

    image_dir = tmp_path / "images"
    image_dir.mkdir()

    records = []

    for item_id, museum, culture in (
        ("item-a", "CMA", "Paracas"),
        ("item-b", "CMA", "Chancay"),
        ("item-c", "MET", "Paracas"),
        ("item-d", "MET", "Wari"),
    ):
        image_path = (
            image_dir / f"{item_id}.jpg"
        )
        image_path.write_bytes(b"image")

        records.append(
            {
                "item_id": item_id,
                "museum": museum,
                "title": f"Title {item_id}",
                "title_original": f"Original {item_id}",
                "description": "Description",
                "culture": culture,
                "culture_canonical": culture,
                "period": "1000",
                "object_type": "Textile",
                "material": "Cotton",
                "technique": "Weave",
                "classification": "Textile",
                "image_local_path": str(
                    image_path.relative_to(
                        tmp_path
                    )
                ),
                "image_url": "https://example.org/image",
                "object_url": "https://example.org/object",
                "image_width": "100",
                "image_height": "100",
                "image_bytes": "5",
                "image_sha256": "hash",
            }
        )

    return pd.DataFrame.from_records(
        records
    )


def _build_text_inputs() -> pd.DataFrame:
    """Construye textos sintéticos."""

    return pd.DataFrame(
        {
            "item_id": [
                "item-a",
                "item-b",
                "item-c",
                "item-d",
            ],
            "museum": [
                "CMA",
                "CMA",
                "MET",
                "MET",
            ],
            "text_visual": [
                "visual a",
                "visual b",
                "visual c",
                "visual d",
            ],
            "text_visual_version": [
                "v1",
                "v1",
                "v1",
                "v1",
            ],
            "text_metadata": [
                "metadata a",
                "metadata b",
                "metadata c",
                "metadata d",
            ],
            "text_metadata_version": [
                "v1",
                "v1",
                "v1",
                "v1",
            ],
        }
    )


def _build_per_query() -> pd.DataFrame:
    """Construye eventos de recuperación para dos configuraciones."""

    records = []

    items = [
        ("item-a", "CMA"),
        ("item-b", "CMA"),
        ("item-c", "MET"),
        ("item-d", "MET"),
    ]

    top1_by_configuration = {
        "config_a": [
            "item-a",
            "item-a",
            "item-a",
            "item-d",
        ],
        "config_b": [
            "item-a",
            "item-a",
            "item-c",
            "item-d",
        ],
    }

    museum_by_item = dict(items)

    for configuration, top1_items in (
        top1_by_configuration.items()
    ):
        for query_index, (
            item_id,
            museum,
        ) in enumerate(items):
            top1_item_id = top1_items[
                query_index
            ]

            records.append(
                {
                    "configuration": configuration,
                    "query_index": query_index,
                    "item_id": item_id,
                    "museum": museum,
                    "matched_rank": query_index + 1,
                    "matched_score": 0.3,
                    "top1_item_id": top1_item_id,
                    "top1_museum": museum_by_item[
                        top1_item_id
                    ],
                    "top1_score": 0.4,
                }
            )

    return pd.DataFrame.from_records(
        records
    )


def _build_persistent_hubs() -> pd.DataFrame:
    """Construye un hub persistente."""

    return pd.DataFrame(
        {
            "candidate_item_id": [
                "item-a",
            ],
            "candidate_museum": [
                "CMA",
            ],
            "configurations_as_hub": [
                2,
            ],
            "configurations": [
                "config_a|config_b",
            ],
            "total_top1_count": [
                5,
            ],
            "total_false_top1_count": [
                3,
            ],
            "max_top1_count": [
                3,
            ],
        }
    )


def _build_candidate_counts() -> pd.DataFrame:
    """Construye conteos por configuración."""

    records = []

    for configuration, count in (
        ("config_a", 3),
        ("config_b", 2),
    ):
        records.append(
            {
                "configuration": configuration,
                "candidate_rank": 1,
                "candidate_item_id": "item-a",
                "candidate_museum": "CMA",
                "top1_count": count,
                "self_match_count": 1,
                "false_top1_count": count - 1,
                "top1_share": count / 4,
                "is_top1_hub": True,
            }
        )

    return pd.DataFrame.from_records(
        records
    )


def _build_audit(
    tmp_path: Path,
):
    """Construye la auditoría sintética."""

    return build_openclip_hub_audit(
        corpus=_build_corpus(tmp_path),
        text_inputs=_build_text_inputs(),
        per_query=_build_per_query(),
        persistent_hubs=(
            _build_persistent_hubs()
        ),
        candidate_counts=(
            _build_candidate_counts()
        ),
        repository_root=tmp_path,
    )


def test_audit_builds_enriched_profile(
    tmp_path: Path,
) -> None:
    """El perfil debe incluir texto e imagen disponible."""

    audit = _build_audit(tmp_path)

    assert len(audit.hub_profiles) == 1

    profile = audit.hub_profiles.iloc[0]

    assert profile["candidate_item_id"] == "item-a"
    assert profile["text_visual"] == "visual a"
    assert profile["text_metadata"] == "metadata a"
    assert bool(profile["image_exists"]) is True


def test_audit_counts_attraction_events(
    tmp_path: Path,
) -> None:
    """Los eventos deben coincidir con los conteos Top-1."""

    audit = _build_audit(tmp_path)

    assert len(audit.attraction_events) == 5

    counts = (
        audit.hub_configuration_counts
        .set_index("configuration")
    )

    assert counts.loc[
        "config_a",
        "attracted_query_count",
    ] == 3

    assert counts.loc[
        "config_b",
        "attracted_query_count",
    ] == 2


def test_audit_detects_cross_museum_events(
    tmp_path: Path,
) -> None:
    """La auditoría debe identificar atracción entre museos."""

    audit = _build_audit(tmp_path)

    events = audit.attraction_events

    assert events["cross_museum"].sum() == 1

    summary = audit.summary.set_index(
        "configuration"
    )

    assert summary.loc[
        "config_a",
        "top1_events_to_persistent_hubs",
    ] == 3

    assert summary.loc[
        "config_a",
        "cross_museum_events",
    ] == 1

    assert summary.loc[
        "config_b",
        "top1_events_to_persistent_hubs",
    ] == 2

    assert summary.loc[
        "config_b",
        "cross_museum_events",
    ] == 0
    

def test_audit_rejects_missing_hub_in_corpus(
    tmp_path: Path,
) -> None:
    """Todo hub persistente debe existir en el corpus."""

    persistent = _build_persistent_hubs()

    persistent.loc[
        0,
        "candidate_item_id",
    ] = "missing-item"

    with pytest.raises(
        ValueError,
        match="ausentes en el corpus",
    ):
        build_openclip_hub_audit(
            corpus=_build_corpus(tmp_path),
            text_inputs=_build_text_inputs(),
            per_query=_build_per_query(),
            persistent_hubs=persistent,
            candidate_counts=(
                _build_candidate_counts()
            ),
            repository_root=tmp_path,
        )