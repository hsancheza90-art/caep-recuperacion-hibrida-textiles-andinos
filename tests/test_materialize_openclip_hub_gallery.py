"""Pruebas de la galería HTML de hubs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.openclip_baseline.materialize_hub_gallery import (
    OpenCLIPHubGalleryConfig,
    build_openclip_hub_gallery_html,
    materialize_openclip_hub_gallery,
)


def _build_profiles(
    tmp_path: Path,
) -> pd.DataFrame:
    """Construye perfiles sintéticos de hubs."""

    image = tmp_path / "hub.jpg"
    image.write_bytes(b"image")

    return pd.DataFrame(
        [
            {
                "candidate_item_id": "item-hub",
                "candidate_museum": "MET",
                "title": "Hub textile",
                "culture_canonical": "Paracas",
                "object_type": "Textile",
                "material": "Cotton",
                "image_local_path": "hub.jpg",
                "configurations_as_hub": 2,
                "total_top1_count": 8,
                "total_false_top1_count": 7,
                "text_visual": "Visual text",
                "text_metadata": "Metadata text",
            }
        ]
    )


def _build_events(
    tmp_path: Path,
) -> pd.DataFrame:
    """Construye consultas atraídas sintéticas."""

    records = []

    for index in range(3):
        image = tmp_path / f"query-{index}.jpg"
        image.write_bytes(b"image")

        records.append(
            {
                "configuration": "config_a",
                "query_index": index,
                "query_item_id": f"item-{index}",
                "query_museum": "CMA",
                "matched_rank": 10 + index,
                "matched_score": 0.25,
                "top1_score": 0.40 + index / 100,
                "candidate_item_id": "item-hub",
                "candidate_museum": "MET",
                "self_match": False,
                "cross_museum": True,
                "same_culture_canonical": (
                    index == 0
                ),
                "same_object_type": True,
                "query_title": f"Query {index}",
                "query_culture_canonical": (
                    "Paracas"
                ),
                "query_object_type": "Textile",
                "query_image_local_path": (
                    f"query-{index}.jpg"
                ),
            }
        )

    return pd.DataFrame.from_records(
        records
    )


def test_gallery_contains_hub_and_queries(
    tmp_path: Path,
) -> None:
    """El HTML debe incluir el hub y sus consultas."""

    output = tmp_path / "gallery.html"

    html = build_openclip_hub_gallery_html(
        hub_profiles=_build_profiles(
            tmp_path
        ),
        attraction_events=_build_events(
            tmp_path
        ),
        repository_root=tmp_path,
        output_html_path=output,
        config=OpenCLIPHubGalleryConfig(
            configurations=("config_a",),
            max_queries_per_configuration=2,
        ),
    )

    assert "item-hub" in html
    assert "Hub textile" in html
    assert "Query 2" in html
    assert "Query 1" in html
    assert "Query 0" not in html
    assert "hub.jpg" in html


def test_gallery_rejects_missing_columns(
    tmp_path: Path,
) -> None:
    """Las tablas deben contener el esquema requerido."""

    with pytest.raises(
        ValueError,
        match="columnas requeridas",
    ):
        build_openclip_hub_gallery_html(
            hub_profiles=pd.DataFrame(
                {"candidate_item_id": ["x"]}
            ),
            attraction_events=_build_events(
                tmp_path
            ),
            repository_root=tmp_path,
            output_html_path=(
                tmp_path / "gallery.html"
            ),
            config=OpenCLIPHubGalleryConfig(
                configurations=("config_a",)
            ),
        )


def test_materializer_writes_html_and_provenance(
    tmp_path: Path,
) -> None:
    """La galería y la procedencia deben materializarse."""

    profiles_path = tmp_path / "profiles.csv"
    events_path = tmp_path / "events.csv"
    output_path = tmp_path / "gallery.html"
    provenance_path = (
        tmp_path / "provenance.json"
    )

    _build_profiles(tmp_path).to_csv(
        profiles_path,
        index=False,
    )
    _build_events(tmp_path).to_csv(
        events_path,
        index=False,
    )

    materialize_openclip_hub_gallery(
        hub_profiles_path=profiles_path,
        attraction_events_path=events_path,
        output_html_path=output_path,
        provenance_path=provenance_path,
        repository_root=tmp_path,
        config=OpenCLIPHubGalleryConfig(
            configurations=("config_a",),
            max_queries_per_configuration=2,
        ),
    )

    assert output_path.is_file()
    assert provenance_path.is_file()

    provenance = json.loads(
        provenance_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        provenance["coverage"][
            "selected_hubs"
        ]
        == 1
    )
    assert (
        provenance["coverage"][
            "attraction_event_rows"
        ]
        == 3
    )


def test_config_rejects_invalid_limits() -> None:
    """Los límites deben ser estrictamente positivos."""

    with pytest.raises(
        ValueError,
        match="max_hubs",
    ):
        OpenCLIPHubGalleryConfig(
            max_hubs=0
        )