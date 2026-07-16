"""Pruebas del bootstrap pareado OpenCLIP."""

from __future__ import annotations

import pandas as pd
import pytest

from src.openclip_baseline.bootstrap import (
    paired_stratified_bootstrap,
)


def _build_per_query() -> pd.DataFrame:
    """Construye resultados pareados de dos configuraciones."""

    records = []

    museums = [
        "CMA",
        "CMA",
        "CMA",
        "MET",
        "MET",
        "MET",
    ]

    ranks_a = [
        3,
        4,
        5,
        3,
        4,
        5,
    ]
    ranks_b = [
        1,
        2,
        3,
        1,
        2,
        3,
    ]

    for configuration, ranks in (
        (
            "config_a",
            ranks_a,
        ),
        (
            "config_b",
            ranks_b,
        ),
    ):
        for query_index, (
            museum,
            rank,
        ) in enumerate(
            zip(
                museums,
                ranks,
                strict=True,
            )
        ):
            records.append(
                {
                    "configuration": (
                        configuration
                    ),
                    "query_index": query_index,
                    "item_id": (
                        f"item-{query_index}"
                    ),
                    "museum": museum,
                    "matched_rank": rank,
                    "reciprocal_rank": (
                        1.0 / rank
                    ),
                }
            )

    return pd.DataFrame.from_records(
        records
    )


def test_bootstrap_is_reproducible() -> None:
    """La misma semilla debe producir réplicas idénticas."""

    first = paired_stratified_bootstrap(
        per_query=_build_per_query(),
        configuration_a="config_a",
        configuration_b="config_b",
        cutoffs=(1, 3, 5),
        n_resamples=100,
        random_seed=42,
    )

    second = paired_stratified_bootstrap(
        per_query=_build_per_query(),
        configuration_a="config_a",
        configuration_b="config_b",
        cutoffs=(1, 3, 5),
        n_resamples=100,
        random_seed=42,
    )

    pd.testing.assert_frame_equal(
        first.replicates,
        second.replicates,
    )
    pd.testing.assert_frame_equal(
        first.summary,
        second.summary,
    )


def test_better_configuration_has_expected_differences() -> None:
    """La configuración B debe mejorar las métricas de ranking."""

    result = paired_stratified_bootstrap(
        per_query=_build_per_query(),
        configuration_a="config_a",
        configuration_b="config_b",
        cutoffs=(1, 3, 5),
        n_resamples=200,
        random_seed=7,
    )

    differences = result.summary.loc[
        result.summary["estimate_type"]
        == "difference_b_minus_a"
    ].set_index("metric")

    assert (
        differences.loc[
            "mrr",
            "point_estimate",
        ]
        > 0.0
    )
    assert (
        differences.loc[
            "recall_at_3",
            "point_estimate",
        ]
        > 0.0
    )
    assert (
        differences.loc[
            "mean_rank",
            "point_estimate",
        ]
        < 0.0
    )
    assert (
        differences.loc[
            "median_rank",
            "point_estimate",
        ]
        < 0.0
    )


def test_bootstrap_has_expected_number_of_rows() -> None:
    """Debe producir una fila por métrica y réplica."""

    result = paired_stratified_bootstrap(
        per_query=_build_per_query(),
        configuration_a="config_a",
        configuration_b="config_b",
        cutoffs=(1, 3, 5),
        n_resamples=25,
    )

    expected_metrics = {
        "mrr",
        "mean_rank",
        "median_rank",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
    }

    assert set(
        result.replicates["metric"]
    ) == expected_metrics

    assert len(
        result.replicates
    ) == 25 * len(expected_metrics)

    assert len(
        result.summary
    ) == 3 * len(expected_metrics)


def test_misaligned_configurations_are_rejected() -> None:
    """Ambas configuraciones deben contener las mismas consultas."""

    frame = _build_per_query()

    frame = frame.loc[
        ~(
            (
                frame["configuration"]
                == "config_b"
            )
            & (
                frame["query_index"]
                == 5
            )
        )
    ]

    with pytest.raises(
        ValueError,
        match="mismas consultas",
    ):
        paired_stratified_bootstrap(
            per_query=frame,
            configuration_a="config_a",
            configuration_b="config_b",
        )


def test_invalid_bootstrap_arguments_are_rejected() -> None:
    """El número de réplicas y nivel de confianza deben ser válidos."""

    frame = _build_per_query()

    with pytest.raises(
        ValueError,
        match="n_resamples",
    ):
        paired_stratified_bootstrap(
            per_query=frame,
            configuration_a="config_a",
            configuration_b="config_b",
            n_resamples=0,
        )

    with pytest.raises(
        ValueError,
        match="confidence_level",
    ):
        paired_stratified_bootstrap(
            per_query=frame,
            configuration_a="config_a",
            configuration_b="config_b",
            confidence_level=1.0,
        )