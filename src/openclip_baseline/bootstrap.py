"""Bootstrap pareado y estratificado para recuperación OpenCLIP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Réplicas y resumen de una comparación pareada."""

    replicates: pd.DataFrame
    summary: pd.DataFrame


def _validate_cutoffs(
    cutoffs: Iterable[int],
) -> tuple[int, ...]:
    """Valida y normaliza los valores de K."""

    normalized = tuple(
        sorted(
            {
                int(cutoff)
                for cutoff in cutoffs
            }
        )
    )

    if not normalized:
        raise ValueError(
            "cutoffs debe contener al menos un valor."
        )

    if any(cutoff <= 0 for cutoff in normalized):
        raise ValueError(
            "Todos los cutoffs deben ser positivos."
        )

    return normalized


def _configuration_frame(
    per_query: pd.DataFrame,
    *,
    configuration: str,
) -> pd.DataFrame:
    """Extrae y valida una configuración por consulta."""

    required_columns = {
        "configuration",
        "query_index",
        "item_id",
        "museum",
        "matched_rank",
        "reciprocal_rank",
    }

    missing_columns = (
        required_columns
        - set(per_query.columns)
    )

    if missing_columns:
        missing = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            "Faltan columnas requeridas: "
            f"{missing}"
        )

    selected = per_query.loc[
        per_query["configuration"]
        == configuration,
        [
            "query_index",
            "item_id",
            "museum",
            "matched_rank",
            "reciprocal_rank",
        ],
    ].copy()

    if selected.empty:
        raise ValueError(
            "No existen filas para la configuración: "
            f"{configuration}"
        )

    if selected["query_index"].duplicated().any():
        raise ValueError(
            "Cada configuración debe contener una sola "
            "fila por query_index."
        )

    if not selected["matched_rank"].ge(1).all():
        raise ValueError(
            "matched_rank debe ser mayor o igual que uno."
        )

    if not np.isfinite(
        selected[
            [
                "matched_rank",
                "reciprocal_rank",
            ]
        ].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(
            "Las métricas por consulta deben ser finitas."
        )

    return selected.sort_values(
        "query_index",
        kind="stable",
    ).reset_index(drop=True)


def _align_configurations(
    per_query: pd.DataFrame,
    *,
    configuration_a: str,
    configuration_b: str,
) -> pd.DataFrame:
    """Alinea dos configuraciones sobre las mismas consultas."""

    left = _configuration_frame(
        per_query,
        configuration=configuration_a,
    )
    right = _configuration_frame(
        per_query,
        configuration=configuration_b,
    )

    keys = [
        "query_index",
        "item_id",
        "museum",
    ]

    aligned = left.merge(
        right,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_a", "_b"),
    )

    if len(aligned) != len(left) or len(aligned) != len(right):
        raise ValueError(
            "Las configuraciones no contienen exactamente "
            "las mismas consultas."
        )

    return aligned.sort_values(
        "query_index",
        kind="stable",
    ).reset_index(drop=True)


def _metrics_from_ranks(
    ranks: np.ndarray,
    *,
    cutoffs: tuple[int, ...],
) -> dict[str, float]:
    """Calcula métricas agregadas a partir de rangos."""

    rank_values = np.asarray(
        ranks,
        dtype=np.float64,
    )

    metrics = {
        "mrr": float(
            np.mean(
                1.0 / rank_values
            )
        ),
        "mean_rank": float(
            np.mean(rank_values)
        ),
        "median_rank": float(
            np.median(rank_values)
        ),
    }

    for cutoff in cutoffs:
        metrics[
            f"recall_at_{cutoff}"
        ] = float(
            np.mean(
                rank_values <= cutoff
            )
        )

    return metrics


def _point_estimates(
    aligned: pd.DataFrame,
    *,
    configuration_a: str,
    configuration_b: str,
    cutoffs: tuple[int, ...],
) -> dict[
    tuple[str, str],
    float
]:
    """Calcula estimaciones sobre la muestra completa."""

    metrics_a = _metrics_from_ranks(
        aligned["matched_rank_a"].to_numpy(),
        cutoffs=cutoffs,
    )
    metrics_b = _metrics_from_ranks(
        aligned["matched_rank_b"].to_numpy(),
        cutoffs=cutoffs,
    )

    estimates: dict[
        tuple[str, str],
        float
    ] = {}

    for metric, value in metrics_a.items():
        estimates[
            (
                configuration_a,
                metric,
            )
        ] = value

    for metric, value in metrics_b.items():
        estimates[
            (
                configuration_b,
                metric,
            )
        ] = value

        estimates[
            (
                "difference_b_minus_a",
                metric,
            )
        ] = (
            value
            - metrics_a[metric]
        )

    return estimates


def paired_stratified_bootstrap(
    *,
    per_query: pd.DataFrame,
    configuration_a: str,
    configuration_b: str,
    cutoffs: Iterable[int] = (1, 5, 10),
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    random_seed: int = 20260715,
) -> PairedBootstrapResult:
    """Compara dos configuraciones mediante bootstrap pareado."""

    normalized_cutoffs = _validate_cutoffs(
        cutoffs
    )

    if n_resamples <= 0:
        raise ValueError(
            "n_resamples debe ser mayor que cero."
        )

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level debe estar entre 0 y 1."
        )

    if configuration_a == configuration_b:
        raise ValueError(
            "Las configuraciones deben ser diferentes."
        )

    aligned = _align_configurations(
        per_query,
        configuration_a=configuration_a,
        configuration_b=configuration_b,
    )

    grouped_indices = [
        group.index.to_numpy(
            dtype=np.int64
        )
        for _, group in aligned.groupby(
            "museum",
            sort=True,
        )
    ]

    generator = np.random.default_rng(
        random_seed
    )

    records: list[
        dict[str, object]
    ] = []

    for replicate in range(n_resamples):
        sampled_parts = [
            generator.choice(
                indices,
                size=len(indices),
                replace=True,
            )
            for indices in grouped_indices
        ]

        sampled_indices = np.concatenate(
            sampled_parts
        )

        ranks_a = aligned.loc[
            sampled_indices,
            "matched_rank_a",
        ].to_numpy(dtype=np.float64)

        ranks_b = aligned.loc[
            sampled_indices,
            "matched_rank_b",
        ].to_numpy(dtype=np.float64)

        metrics_a = _metrics_from_ranks(
            ranks_a,
            cutoffs=normalized_cutoffs,
        )
        metrics_b = _metrics_from_ranks(
            ranks_b,
            cutoffs=normalized_cutoffs,
        )

        for metric in metrics_a:
            records.append(
                {
                    "replicate": replicate,
                    "metric": metric,
                    "configuration_a": (
                        configuration_a
                    ),
                    "value_a": metrics_a[metric],
                    "configuration_b": (
                        configuration_b
                    ),
                    "value_b": metrics_b[metric],
                    "difference_b_minus_a": (
                        metrics_b[metric]
                        - metrics_a[metric]
                    ),
                }
            )

    replicates = pd.DataFrame.from_records(
        records
    )

    estimates = _point_estimates(
        aligned,
        configuration_a=configuration_a,
        configuration_b=configuration_b,
        cutoffs=normalized_cutoffs,
    )

    tail_probability = (
        1.0 - confidence_level
    ) / 2.0

    lower_quantile = tail_probability
    upper_quantile = 1.0 - tail_probability

    summary_records: list[
        dict[str, object]
    ] = []

    for metric, group in replicates.groupby(
        "metric",
        sort=False,
    ):
        for label, column in (
            (
                configuration_a,
                "value_a",
            ),
            (
                configuration_b,
                "value_b",
            ),
            (
                "difference_b_minus_a",
                "difference_b_minus_a",
            ),
        ):
            values = group[column].to_numpy(
                dtype=np.float64
            )

            summary_records.append(
                {
                    "metric": metric,
                    "estimate_type": label,
                    "point_estimate": estimates[
                        (
                            label,
                            metric,
                        )
                    ],
                    "bootstrap_mean": float(
                        np.mean(values)
                    ),
                    "ci_lower": float(
                        np.quantile(
                            values,
                            lower_quantile,
                        )
                    ),
                    "ci_upper": float(
                        np.quantile(
                            values,
                            upper_quantile,
                        )
                    ),
                    "confidence_level": (
                        confidence_level
                    ),
                    "n_resamples": n_resamples,
                    "random_seed": random_seed,
                }
            )

    summary = pd.DataFrame.from_records(
        summary_records
    )

    return PairedBootstrapResult(
        replicates=replicates,
        summary=summary,
    )