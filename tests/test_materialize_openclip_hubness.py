"""Pruebas de materialización del hubness OpenCLIP."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.openclip_baseline.materialize_hubness import (
    OpenCLIPHubnessConfig,
    build_openclip_hubness_artifacts,
    materialize_openclip_hubness,
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
                    "top1_item_id": top1_item_id,
                    "top1_museum": museum_by_item[
                        top1_item_id
                    ],
                }
            )

    return pd.DataFrame.from_records(
        records
    )


def _config() -> OpenCLIPHubnessConfig:
    """Devuelve una configuración de prueba."""

    return OpenCLIPHubnessConfig(
        configurations=(
            "config_a",
            "config_b",
        ),
        hub_min_count=2,
        persistent_min_configurations=2,
    )


def test_config_rejects_duplicate_configurations() -> None:
    """Las configuraciones deben ser únicas."""

    with pytest.raises(
        ValueError,
        match="únicas",
    ):
        OpenCLIPHubnessConfig(
            configurations=(
                "same",
                "same",
            )
        )


def test_build_hubness_artifacts() -> None:
    """La construcción debe producir las cuatro tablas."""

    analysis = build_openclip_hubness_artifacts(
        per_query=_build_per_query(),
        config=_config(),
    )

    assert len(
        analysis.candidate_counts
    ) == 8

    assert len(
        analysis.summary
    ) == 2

    assert not analysis.museum_flow.empty
    assert len(
        analysis.persistent_hubs
    ) == 1


def test_materialize_hubness(
    tmp_path: Path,
) -> None:
    """Los CSV y el JSON deben quedar materializados."""

    per_query_path = (
        tmp_path / "per_query.csv"
    )
    candidate_counts_path = (
        tmp_path / "candidate_counts.csv"
    )
    summary_path = (
        tmp_path / "summary.csv"
    )
    museum_flow_path = (
        tmp_path / "museum_flow.csv"
    )
    persistent_hubs_path = (
        tmp_path / "persistent_hubs.csv"
    )
    provenance_path = (
        tmp_path / "provenance.json"
    )

    _build_per_query().to_csv(
        per_query_path,
        index=False,
    )

    materialize_openclip_hubness(
        per_query_path=per_query_path,
        candidate_counts_path=(
            candidate_counts_path
        ),
        summary_path=summary_path,
        museum_flow_path=museum_flow_path,
        persistent_hubs_path=(
            persistent_hubs_path
        ),
        provenance_path=provenance_path,
        repository_root=tmp_path,
        config=_config(),
    )

    for path in (
        candidate_counts_path,
        summary_path,
        museum_flow_path,
        persistent_hubs_path,
        provenance_path,
    ):
        assert path.is_file()

    candidate_counts = pd.read_csv(
        candidate_counts_path
    )
    summary = pd.read_csv(
        summary_path
    )

    assert len(candidate_counts) == 8
    assert len(summary) == 2

    provenance = json.loads(
        provenance_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        provenance["configuration"][
            "ranking_position"
        ]
        == 1
    )
    assert (
        provenance["coverage"][
            "candidate_count_rows"
        ]
        == 8
    )
    assert (
        provenance["coverage"][
            "persistent_hub_rows"
        ]
        == 1
    )


def test_materializer_rejects_missing_input(
    tmp_path: Path,
) -> None:
    """El archivo por consulta debe existir."""

    with pytest.raises(
        FileNotFoundError,
        match="No existe",
    ):
        materialize_openclip_hubness(
            per_query_path=(
                tmp_path / "missing.csv"
            ),
            candidate_counts_path=(
                tmp_path / "candidate_counts.csv"
            ),
            summary_path=(
                tmp_path / "summary.csv"
            ),
            museum_flow_path=(
                tmp_path / "museum_flow.csv"
            ),
            persistent_hubs_path=(
                tmp_path / "persistent_hubs.csv"
            ),
            provenance_path=(
                tmp_path / "provenance.json"
            ),
            repository_root=tmp_path,
            config=_config(),
        )