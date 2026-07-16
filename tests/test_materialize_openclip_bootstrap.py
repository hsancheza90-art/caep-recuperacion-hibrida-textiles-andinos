"""Pruebas de materialización del bootstrap OpenCLIP."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.openclip_baseline.materialize_bootstrap import (
    OpenCLIPBootstrapConfig,
    build_openclip_bootstrap_artifacts,
    materialize_openclip_bootstrap,
)


def _build_per_query() -> pd.DataFrame:
    """Construye dos configuraciones pareadas."""

    records = []

    museums = [
        "CMA",
        "CMA",
        "CMA",
        "MET",
        "MET",
        "MET",
    ]

    configurations = {
        "config_a": [
            3,
            4,
            5,
            3,
            4,
            5,
        ],
        "config_b": [
            1,
            2,
            3,
            1,
            2,
            3,
        ],
    }

    for configuration, ranks in configurations.items():
        for query_index, rank in enumerate(ranks):
            records.append(
                {
                    "configuration": configuration,
                    "query_index": query_index,
                    "item_id": f"item-{query_index}",
                    "museum": museums[query_index],
                    "matched_rank": rank,
                    "reciprocal_rank": 1.0 / rank,
                }
            )

    return pd.DataFrame.from_records(
        records
    )


def _config() -> OpenCLIPBootstrapConfig:
    """Devuelve una configuración pequeña de prueba."""

    return OpenCLIPBootstrapConfig(
        configuration_a="config_a",
        configuration_b="config_b",
        cutoffs=(1, 3, 5),
        n_resamples=25,
        confidence_level=0.95,
        random_seed=42,
    )


def test_config_rejects_equal_configurations() -> None:
    """La comparación requiere dos configuraciones distintas."""

    with pytest.raises(
        ValueError,
        match="diferentes",
    ):
        OpenCLIPBootstrapConfig(
            configuration_a="same",
            configuration_b="same",
        )


def test_build_bootstrap_artifacts() -> None:
    """La construcción debe generar réplicas y resumen."""

    result = build_openclip_bootstrap_artifacts(
        per_query=_build_per_query(),
        config=_config(),
    )

    expected_metrics = 6

    assert len(result.replicates) == (
        25 * expected_metrics
    )
    assert len(result.summary) == (
        3 * expected_metrics
    )

    assert set(
        result.summary["estimate_type"]
    ) == {
        "config_a",
        "config_b",
        "difference_b_minus_a",
    }


def test_materialize_bootstrap(
    tmp_path: Path,
) -> None:
    """La materialización debe producir CSV y JSON válidos."""

    per_query_path = (
        tmp_path / "per_query.csv"
    )
    replicates_path = (
        tmp_path / "replicates.csv"
    )
    summary_path = (
        tmp_path / "summary.csv"
    )
    provenance_path = (
        tmp_path / "provenance.json"
    )

    _build_per_query().to_csv(
        per_query_path,
        index=False,
    )

    materialize_openclip_bootstrap(
        per_query_path=per_query_path,
        replicates_path=replicates_path,
        summary_path=summary_path,
        provenance_path=provenance_path,
        repository_root=tmp_path,
        config=_config(),
    )

    assert replicates_path.is_file()
    assert summary_path.is_file()
    assert provenance_path.is_file()

    replicates = pd.read_csv(
        replicates_path
    )
    summary = pd.read_csv(
        summary_path
    )

    assert len(replicates) == 150
    assert len(summary) == 18

    provenance = json.loads(
        provenance_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        provenance["configuration"][
            "paired"
        ]
        is True
    )
    assert (
        provenance["configuration"][
            "stratified_by"
        ]
        == "museum"
    )
    assert (
        provenance["coverage"][
            "replicate_rows"
        ]
        == 150
    )


def test_materializer_rejects_missing_input(
    tmp_path: Path,
) -> None:
    """El archivo por consulta debe existir."""

    with pytest.raises(
        FileNotFoundError,
        match="No existe",
    ):
        materialize_openclip_bootstrap(
            per_query_path=(
                tmp_path / "missing.csv"
            ),
            replicates_path=(
                tmp_path / "replicates.csv"
            ),
            summary_path=(
                tmp_path / "summary.csv"
            ),
            provenance_path=(
                tmp_path / "provenance.json"
            ),
            repository_root=tmp_path,
            config=_config(),
        )