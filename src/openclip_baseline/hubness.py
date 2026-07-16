"""Análisis de hubness Top-1 en recuperación OpenCLIP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Top1HubnessAnalysis:
    """Tablas resultantes del análisis de concentración Top-1."""

    candidate_counts: pd.DataFrame
    summary: pd.DataFrame
    museum_flow: pd.DataFrame
    persistent_hubs: pd.DataFrame


_REQUIRED_COLUMNS = {
    "configuration",
    "query_index",
    "item_id",
    "museum",
    "top1_item_id",
    "top1_museum",
}


def _require_columns(
    per_query: pd.DataFrame,
) -> None:
    """Valida las columnas necesarias."""

    missing = _REQUIRED_COLUMNS - set(
        per_query.columns
    )

    if missing:
        missing_text = ", ".join(
            sorted(missing)
        )

        raise ValueError(
            "Faltan columnas requeridas: "
            f"{missing_text}"
        )


def _select_configuration(
    per_query: pd.DataFrame,
    *,
    configuration: str,
) -> pd.DataFrame:
    """Extrae una configuración y valida sus consultas."""

    selected = per_query.loc[
        per_query["configuration"]
        == configuration
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

    if selected["item_id"].duplicated().any():
        raise ValueError(
            "Cada configuración debe contener una sola "
            "fila por item_id."
        )

    return selected.sort_values(
        "query_index",
        kind="stable",
    ).reset_index(drop=True)


def _validate_candidate_references(
    selected: pd.DataFrame,
) -> None:
    """Comprueba que cada Top-1 pertenezca al corpus."""

    candidate_museums = dict(
        zip(
            selected["item_id"].astype(str),
            selected["museum"].astype(str),
            strict=True,
        )
    )

    unknown_candidates = (
        set(
            selected["top1_item_id"].astype(str)
        )
        - set(candidate_museums)
    )

    if unknown_candidates:
        unknown = ", ".join(
            sorted(unknown_candidates)
        )

        raise ValueError(
            "Se encontraron candidatos Top-1 que no "
            f"pertenecen al corpus: {unknown}"
        )

    expected_museums = selected[
        "top1_item_id"
    ].astype(str).map(
        candidate_museums
    )

    observed_museums = selected[
        "top1_museum"
    ].astype(str)

    if not expected_museums.equals(
        observed_museums
    ):
        raise ValueError(
            "top1_museum no coincide con el museo "
            "registrado para top1_item_id."
        )


def build_top1_candidate_counts(
    *,
    per_query: pd.DataFrame,
    configuration: str,
    hub_min_count: int = 3,
) -> pd.DataFrame:
    """Cuenta cuántas consultas recupera cada candidato en Top-1."""

    _require_columns(per_query)

    if hub_min_count <= 0:
        raise ValueError(
            "hub_min_count debe ser mayor que cero."
        )

    selected = _select_configuration(
        per_query,
        configuration=configuration,
    )

    _validate_candidate_references(
        selected
    )

    candidate_universe = selected[
        [
            "item_id",
            "museum",
        ]
    ].rename(
        columns={
            "item_id": "candidate_item_id",
            "museum": "candidate_museum",
        }
    )

    events = selected[
        [
            "item_id",
            "top1_item_id",
        ]
    ].copy()

    events["self_match"] = (
        events["item_id"].astype(str)
        == events["top1_item_id"].astype(str)
    )

    counts = (
        events.groupby(
            "top1_item_id",
            sort=False,
        )
        .agg(
            top1_count=(
                "top1_item_id",
                "size",
            ),
            self_match_count=(
                "self_match",
                "sum",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "top1_item_id": (
                    "candidate_item_id"
                )
            }
        )
    )

    result = candidate_universe.merge(
        counts,
        on="candidate_item_id",
        how="left",
        validate="one_to_one",
    )

    result[
        [
            "top1_count",
            "self_match_count",
        ]
    ] = (
        result[
            [
                "top1_count",
                "self_match_count",
            ]
        ]
        .fillna(0)
        .astype(np.int64)
    )

    result["false_top1_count"] = (
        result["top1_count"]
        - result["self_match_count"]
    )

    result["top1_share"] = (
        result["top1_count"]
        / len(selected)
    )

    result["is_top1_hub"] = (
        result["top1_count"]
        >= hub_min_count
    )

    result.insert(
        0,
        "configuration",
        configuration,
    )

    result = result.sort_values(
        [
            "top1_count",
            "false_top1_count",
            "candidate_item_id",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)

    result.insert(
        1,
        "candidate_rank",
        np.arange(
            1,
            len(result) + 1,
            dtype=np.int64,
        ),
    )

    return result


def _gini_coefficient(
    values: np.ndarray,
) -> float:
    """Calcula Gini para valores no negativos."""

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.ndim != 1:
        raise ValueError(
            "values debe ser un vector."
        )

    if len(array) == 0:
        raise ValueError(
            "values no puede estar vacío."
        )

    if np.any(array < 0):
        raise ValueError(
            "values no puede contener negativos."
        )

    total = float(array.sum())

    if total == 0.0:
        return 0.0

    ordered = np.sort(array)
    positions = np.arange(
        1,
        len(ordered) + 1,
        dtype=np.float64,
    )

    return float(
        (
            2.0
            * np.sum(
                positions * ordered
            )
            / (
                len(ordered)
                * total
            )
        )
        - (
            len(ordered) + 1
        )
        / len(ordered)
    )


def _normalized_entropy(
    counts: np.ndarray,
) -> float:
    """Calcula entropía normalizada de las apariciones Top-1."""

    values = np.asarray(
        counts,
        dtype=np.float64,
    )

    total = float(values.sum())

    if total <= 0.0:
        return 0.0

    if len(values) == 1:
        return 1.0

    probabilities = values[
        values > 0
    ] / total

    entropy = -float(
        np.sum(
            probabilities
            * np.log(probabilities)
        )
    )

    return entropy / float(
        np.log(len(values))
    )


def build_top1_hubness_summary(
    candidate_counts: pd.DataFrame,
) -> pd.DataFrame:
    """Resume la concentración Top-1 de una configuración."""

    required = {
        "configuration",
        "top1_count",
        "top1_share",
        "false_top1_count",
        "is_top1_hub",
    }

    missing = required - set(
        candidate_counts.columns
    )

    if missing:
        raise ValueError(
            "candidate_counts no contiene todas "
            "las columnas requeridas."
        )

    configurations = candidate_counts[
        "configuration"
    ].unique()

    if len(configurations) != 1:
        raise ValueError(
            "candidate_counts debe corresponder a "
            "una sola configuración."
        )

    counts = candidate_counts[
        "top1_count"
    ].to_numpy(
        dtype=np.float64
    )

    total_queries = int(
        counts.sum()
    )
    total_candidates = len(
        candidate_counts
    )

    hub_mask = candidate_counts[
        "is_top1_hub"
    ].to_numpy(dtype=bool)

    hub_query_count = int(
        candidate_counts.loc[
            hub_mask,
            "top1_count",
        ].sum()
    )

    shares = candidate_counts[
        "top1_share"
    ].to_numpy(
        dtype=np.float64
    )

    record = {
        "configuration": str(
            configurations[0]
        ),
        "total_queries": total_queries,
        "total_candidates": total_candidates,
        "unique_top1_candidates": int(
            np.count_nonzero(counts)
        ),
        "top1_candidate_coverage": float(
            np.count_nonzero(counts)
            / total_candidates
        ),
        "mean_top1_count": float(
            np.mean(counts)
        ),
        "max_top1_count": int(
            np.max(counts)
        ),
        "max_top1_share": float(
            np.max(shares)
        ),
        "gini_top1_counts": (
            _gini_coefficient(counts)
        ),
        "normalized_top1_entropy": (
            _normalized_entropy(counts)
        ),
        "top1_hhi": float(
            np.sum(
                shares**2
            )
        ),
        "total_hubs": int(
            np.count_nonzero(hub_mask)
        ),
        "queries_captured_by_hubs": (
            hub_query_count
        ),
        "hub_query_share": float(
            hub_query_count
            / total_queries
        ),
        "false_top1_events": int(
            candidate_counts[
                "false_top1_count"
            ].sum()
        ),
    }

    return pd.DataFrame.from_records(
        [record]
    )


def build_top1_museum_flow(
    *,
    per_query: pd.DataFrame,
    configuration: str,
) -> pd.DataFrame:
    """Resume el flujo museo de consulta → museo Top-1."""

    _require_columns(per_query)

    selected = _select_configuration(
        per_query,
        configuration=configuration,
    )

    _validate_candidate_references(
        selected
    )

    flow = (
        selected.groupby(
            [
                "museum",
                "top1_museum",
            ],
            sort=True,
        )
        .size()
        .rename("count")
        .reset_index()
        .rename(
            columns={
                "museum": "query_museum",
            }
        )
    )

    flow["share_within_query_museum"] = (
        flow["count"]
        / flow.groupby(
            "query_museum"
        )["count"].transform("sum")
    )

    flow.insert(
        0,
        "configuration",
        configuration,
    )

    return flow


def build_persistent_hubs(
    candidate_counts: pd.DataFrame,
    *,
    min_configurations: int = 2,
) -> pd.DataFrame:
    """Identifica candidatos que son hubs en varias configuraciones."""

    if min_configurations <= 0:
        raise ValueError(
            "min_configurations debe ser mayor que cero."
        )

    hubs = candidate_counts.loc[
        candidate_counts["is_top1_hub"]
    ].copy()

    columns = [
        "candidate_item_id",
        "candidate_museum",
        "configurations_as_hub",
        "configurations",
        "total_top1_count",
        "total_false_top1_count",
        "max_top1_count",
    ]

    if hubs.empty:
        return pd.DataFrame(
            columns=columns
        )

    persistent = (
        hubs.groupby(
            [
                "candidate_item_id",
                "candidate_museum",
            ],
            as_index=False,
        )
        .agg(
            configurations_as_hub=(
                "configuration",
                "nunique",
            ),
            configurations=(
                "configuration",
                lambda values: "|".join(
                    sorted(
                        set(
                            values.astype(str)
                        )
                    )
                ),
            ),
            total_top1_count=(
                "top1_count",
                "sum",
            ),
            total_false_top1_count=(
                "false_top1_count",
                "sum",
            ),
            max_top1_count=(
                "top1_count",
                "max",
            ),
        )
    )

    persistent = persistent.loc[
        persistent[
            "configurations_as_hub"
        ]
        >= min_configurations
    ]

    return persistent.sort_values(
        [
            "configurations_as_hub",
            "total_top1_count",
            "total_false_top1_count",
            "candidate_item_id",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)


def analyze_top1_hubness(
    *,
    per_query: pd.DataFrame,
    configurations: Iterable[str] | None = None,
    hub_min_count: int = 3,
    persistent_min_configurations: int = 2,
) -> Top1HubnessAnalysis:
    """Ejecuta el análisis Top-1 para varias configuraciones."""

    _require_columns(per_query)

    if configurations is None:
        selected_configurations = tuple(
            sorted(
                per_query[
                    "configuration"
                ].astype(str).unique()
            )
        )
    else:
        selected_configurations = tuple(
            str(configuration)
            for configuration in configurations
        )

    if not selected_configurations:
        raise ValueError(
            "Debe indicarse al menos una configuración."
        )

    counts_frames = []
    summary_frames = []
    flow_frames = []

    for configuration in selected_configurations:
        counts = build_top1_candidate_counts(
            per_query=per_query,
            configuration=configuration,
            hub_min_count=hub_min_count,
        )

        counts_frames.append(counts)
        summary_frames.append(
            build_top1_hubness_summary(
                counts
            )
        )
        flow_frames.append(
            build_top1_museum_flow(
                per_query=per_query,
                configuration=configuration,
            )
        )

    candidate_counts = pd.concat(
        counts_frames,
        ignore_index=True,
    )

    summary = pd.concat(
        summary_frames,
        ignore_index=True,
    )

    museum_flow = pd.concat(
        flow_frames,
        ignore_index=True,
    )

    persistent_hubs = build_persistent_hubs(
        candidate_counts,
        min_configurations=(
            persistent_min_configurations
        ),
    )

    return Top1HubnessAnalysis(
        candidate_counts=candidate_counts,
        summary=summary,
        museum_flow=museum_flow,
        persistent_hubs=persistent_hubs,
    )