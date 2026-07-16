"""Pruebas de la plantilla de revisión manual de hubs."""

from __future__ import annotations

import pandas as pd
import pytest

from src.openclip_baseline.hub_review import (
    OpenCLIPHubReviewConfig,
    build_openclip_hub_review_codebook,
    build_openclip_hub_review_template,
    summarize_hub_review_template,
)


def _build_profiles() -> pd.DataFrame:
    """Construye dos perfiles de hubs."""

    return pd.DataFrame(
        [
            {
                "candidate_item_id": "hub-a",
                "candidate_museum": "MET",
                "title": "Hub A",
                "culture_canonical": "Paracas",
                "object_type": "Textile",
                "material": "Cotton",
                "text_visual": "Visual A",
                "text_metadata": "Metadata A",
                "total_top1_count": 10,
                "total_false_top1_count": 9,
                "max_top1_count": 7,
            },
            {
                "candidate_item_id": "hub-b",
                "candidate_museum": "CMA",
                "title": "Hub B",
                "culture_canonical": "Chancay",
                "object_type": "Panel",
                "material": "Camelid",
                "text_visual": "Visual B",
                "text_metadata": "Metadata B",
                "total_top1_count": 6,
                "total_false_top1_count": 6,
                "max_top1_count": 4,
            },
        ]
    )


def _build_events() -> pd.DataFrame:
    """Construye eventos atraídos por los hubs."""

    records = []

    for candidate_item_id, candidate_museum in (
        ("hub-a", "MET"),
        ("hub-b", "CMA"),
    ):
        for configuration in (
            "config-a",
            "config-b",
        ):
            for query_index in range(4):
                records.append(
                    {
                        "configuration": configuration,
                        "query_index": query_index,
                        "query_item_id": (
                            f"{configuration}-q{query_index}"
                        ),
                        "query_museum": (
                            "CMA"
                            if query_index % 2 == 0
                            else "MET"
                        ),
                        "matched_rank": (
                            20 + query_index
                        ),
                        "matched_score": 0.20,
                        "top1_score": (
                            0.30
                            + query_index / 100
                        ),
                        "candidate_item_id": (
                            candidate_item_id
                        ),
                        "candidate_museum": (
                            candidate_museum
                        ),
                        "self_match": False,
                        "cross_museum": (
                            candidate_museum
                            != (
                                "CMA"
                                if query_index % 2 == 0
                                else "MET"
                            )
                        ),
                        "same_culture_canonical": (
                            query_index == 0
                        ),
                        "same_object_type": (
                            query_index < 2
                        ),
                        "query_title": (
                            f"Query {query_index}"
                        ),
                        "query_culture_canonical": (
                            "Paracas"
                        ),
                        "query_object_type": (
                            "Textile"
                        ),
                        "query_image_local_path": (
                            f"query-{query_index}.jpg"
                        ),
                        "candidate_title": (
                            f"Candidate {candidate_item_id}"
                        ),
                        "candidate_culture_canonical": (
                            "Paracas"
                        ),
                        "candidate_object_type": (
                            "Textile"
                        ),
                        "candidate_image_local_path": (
                            f"{candidate_item_id}.jpg"
                        ),
                    }
                )

    return pd.DataFrame.from_records(
        records
    )


def test_review_template_selects_expected_rows() -> None:
    """Debe respetarse el límite por hub y configuración."""

    review = build_openclip_hub_review_template(
        hub_profiles=_build_profiles(),
        attraction_events=_build_events(),
        config=OpenCLIPHubReviewConfig(
            configurations=(
                "config-a",
                "config-b",
            ),
            max_hubs=2,
            max_events_per_hub_configuration=2,
        ),
    )

    assert len(review) == 8

    counts = (
        review.groupby(
            [
                "candidate_item_id",
                "configuration",
            ]
        )
        .size()
    )

    assert (
        counts == 2
    ).all()


def test_review_template_prioritizes_larger_margin() -> None:
    """Los eventos con mayor margen deben aparecer primero."""

    review = build_openclip_hub_review_template(
        hub_profiles=_build_profiles(),
        attraction_events=_build_events(),
        config=OpenCLIPHubReviewConfig(
            configurations=("config-a",),
            max_hubs=1,
            max_events_per_hub_configuration=2,
        ),
    )

    assert review[
        "query_index"
    ].tolist() == [
        3,
        2,
    ]

    assert review[
        "score_margin"
    ].is_monotonic_decreasing


def test_review_template_contains_manual_fields() -> None:
    """La plantilla debe incluir campos editables."""

    review = build_openclip_hub_review_template(
        hub_profiles=_build_profiles(),
        attraction_events=_build_events(),
        config=OpenCLIPHubReviewConfig(
            configurations=("config-a",),
            max_hubs=1,
            max_events_per_hub_configuration=1,
        ),
    )

    assert review.iloc[0][
        "review_status"
    ] == "pendiente"

    assert review.iloc[0][
        "dominant_cause"
    ] == ""

    assert review.iloc[0][
        "review_notes"
    ] == ""


def test_codebook_contains_controlled_fields() -> None:
    """El vocabulario debe incluir las categorías principales."""

    codebook = build_openclip_hub_review_codebook()

    assert {
        "review_status",
        "visual_similarity",
        "dominant_cause",
        "hubness_severity",
        "recommended_action",
    }.issubset(
        set(codebook["field"])
    )

    causes = set(
        codebook.loc[
            codebook["field"]
            == "dominant_cause",
            "allowed_value",
        ]
    )

    assert "similitud_visual" in causes
    assert "hubness_embedding" in causes


def test_review_summary_reports_configuration_coverage() -> None:
    """El resumen debe separar las configuraciones."""

    review = build_openclip_hub_review_template(
        hub_profiles=_build_profiles(),
        attraction_events=_build_events(),
        config=OpenCLIPHubReviewConfig(
            configurations=(
                "config-a",
                "config-b",
            ),
            max_hubs=2,
            max_events_per_hub_configuration=2,
        ),
    )

    summary = summarize_hub_review_template(
        review
    )

    assert set(
        summary["configuration"]
    ) == {
        "config-a",
        "config-b",
    }

    assert (
        summary["review_rows"] == 4
    ).all()


def test_config_rejects_invalid_limits() -> None:
    """Los límites deben ser positivos."""

    with pytest.raises(
        ValueError,
        match="max_hubs",
    ):
        OpenCLIPHubReviewConfig(
            max_hubs=0
        )