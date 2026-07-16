"""Pruebas del análisis de hubness OpenCLIP."""

from __future__ import annotations

import pandas as pd
import pytest

from src.openclip_baseline.hubness import (
    analyze_top1_hubness,
    build_persistent_hubs,
    build_top1_candidate_counts,
)


def _build_per_query() -> pd.DataFrame:
    """Construye dos configuraciones con un hub común."""

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
    records = []

    for configuration, top1_items in (
        top1_by_configuration.items()
    ):
        for query_index, (
            item,
            museum,
        ) in enumerate(items):
            top1_item = top1_items[
                query_index
            ]

            records.append(
                {
                    "configuration": (
                        configuration
                    ),
                    "query_index": query_index,
                    "item_id": item,
                    "museum": museum,
                    "top1_item_id": top1_item,
                    "top1_museum": (
                        museum_by_item[
                            top1_item
                        ]
                    ),
                }
            )

    return pd.DataFrame.from_records(
        records
    )


def test_candidate_counts_include_zero_candidates() -> None:
    """El universo debe incluir candidatos nunca recuperados."""

    counts = build_top1_candidate_counts(
        per_query=_build_per_query(),
        configuration="config_a",
        hub_min_count=2,
    )

    assert len(counts) == 4

    indexed = counts.set_index(
        "candidate_item_id"
    )

    assert indexed.loc[
        "item-a",
        "top1_count",
    ] == 3

    assert indexed.loc[
        "item-a",
        "self_match_count",
    ] == 1

    assert indexed.loc[
        "item-a",
        "false_top1_count",
    ] == 2

    assert indexed.loc[
        "item-b",
        "top1_count",
    ] == 0


def test_analysis_builds_summary_and_flow() -> None:
    """El análisis debe resumir ambas configuraciones."""

    analysis = analyze_top1_hubness(
        per_query=_build_per_query(),
        configurations=(
            "config_a",
            "config_b",
        ),
        hub_min_count=2,
    )

    assert len(
        analysis.candidate_counts
    ) == 8

    assert len(analysis.summary) == 2

    assert set(
        analysis.summary[
            "configuration"
        ]
    ) == {
        "config_a",
        "config_b",
    }

    summary_a = analysis.summary.loc[
        analysis.summary[
            "configuration"
        ]
        == "config_a"
    ].iloc[0]

    assert (
        summary_a[
            "unique_top1_candidates"
        ]
        == 2
    )

    assert (
        summary_a[
            "max_top1_count"
        ]
        == 3
    )

    assert (
        summary_a[
            "top1_candidate_coverage"
        ]
        == pytest.approx(0.5)
    )

def test_common_hub_is_persistent() -> None:
    """Un candidato hub en dos configuraciones debe persistir."""

    analysis = analyze_top1_hubness(
        per_query=_build_per_query(),
        configurations=(
            "config_a",
            "config_b",
        ),
        hub_min_count=2,
        persistent_min_configurations=2,
    )

    assert len(
        analysis.persistent_hubs
    ) == 1

    hub = analysis.persistent_hubs.iloc[0]

    assert (
        hub["candidate_item_id"]
        == "item-a"
    )
    assert (
        hub["configurations_as_hub"]
        == 2
    )
    assert (
        hub["total_top1_count"]
        == 5
    )


def test_persistent_hubs_can_be_empty() -> None:
    """Debe devolverse una tabla vacía con columnas estables."""

    counts = build_top1_candidate_counts(
        per_query=_build_per_query(),
        configuration="config_a",
        hub_min_count=4,
    )

    persistent = build_persistent_hubs(
        counts,
        min_configurations=2,
    )

    assert persistent.empty
    assert "candidate_item_id" in persistent
    assert "configurations_as_hub" in persistent


def test_unknown_top1_candidate_is_rejected() -> None:
    """Todo candidato Top-1 debe pertenecer al corpus."""

    frame = _build_per_query()

    frame.loc[
        frame.index[0],
        "top1_item_id",
    ] = "unknown-item"

    with pytest.raises(
        ValueError,
        match="no pertenecen",
    ):
        build_top1_candidate_counts(
            per_query=frame,
            configuration="config_a",
        )
