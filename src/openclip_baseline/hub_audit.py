"""Auditoría enriquecida de hubs de recuperación OpenCLIP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OpenCLIPHubAudit:
    """Tablas resultantes de la auditoría de hubs."""

    hub_profiles: pd.DataFrame
    hub_configuration_counts: pd.DataFrame
    attraction_events: pd.DataFrame
    summary: pd.DataFrame


_CORPUS_REQUIRED_COLUMNS = {
    "item_id",
    "museum",
    "title",
    "title_original",
    "description",
    "culture",
    "culture_canonical",
    "period",
    "object_type",
    "material",
    "technique",
    "classification",
    "image_local_path",
    "image_url",
    "object_url",
    "image_width",
    "image_height",
    "image_bytes",
    "image_sha256",
}

_TEXT_REQUIRED_COLUMNS = {
    "item_id",
    "museum",
    "text_visual",
    "text_visual_version",
    "text_metadata",
    "text_metadata_version",
}

_PERSISTENT_REQUIRED_COLUMNS = {
    "candidate_item_id",
    "candidate_museum",
    "configurations_as_hub",
    "configurations",
    "total_top1_count",
    "total_false_top1_count",
    "max_top1_count",
}

_COUNTS_REQUIRED_COLUMNS = {
    "configuration",
    "candidate_rank",
    "candidate_item_id",
    "candidate_museum",
    "top1_count",
    "self_match_count",
    "false_top1_count",
    "top1_share",
    "is_top1_hub",
}

_PER_QUERY_REQUIRED_COLUMNS = {
    "configuration",
    "query_index",
    "item_id",
    "museum",
    "matched_rank",
    "matched_score",
    "top1_item_id",
    "top1_museum",
    "top1_score",
}


def _require_columns(
    frame: pd.DataFrame,
    *,
    required: set[str],
    frame_name: str,
) -> None:
    """Comprueba que una tabla contenga las columnas requeridas."""

    missing = required - set(frame.columns)

    if missing:
        missing_text = ", ".join(
            sorted(missing)
        )

        raise ValueError(
            f"{frame_name} no contiene las columnas "
            f"requeridas: {missing_text}"
        )


def _require_unique_ids(
    frame: pd.DataFrame,
    *,
    identifier: str,
    frame_name: str,
) -> None:
    """Comprueba que una tabla tenga identificadores únicos."""

    if frame[identifier].astype(str).duplicated().any():
        raise ValueError(
            f"{frame_name} contiene identificadores duplicados "
            f"en {identifier}."
        )


def _nonempty_equal(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:
    """Compara valores no vacíos evitando falsos iguales."""

    left_values = left.astype(str).str.strip()
    right_values = right.astype(str).str.strip()

    return (
        left_values.ne("")
        & right_values.ne("")
        & left_values.eq(right_values)
    )


def _resolve_image_exists(
    *,
    image_local_path: str,
    repository_root: Path,
) -> bool:
    """Comprueba la existencia de una imagen local."""

    path = Path(image_local_path)

    if not path.is_absolute():
        path = repository_root / path

    return path.is_file()


def _build_hub_profiles(
    *,
    corpus: pd.DataFrame,
    text_inputs: pd.DataFrame,
    persistent_hubs: pd.DataFrame,
    repository_root: Path,
) -> pd.DataFrame:
    """Enriquece los hubs con metadatos, textos e imagen."""

    corpus_columns = [
        "item_id",
        "museum",
        "title",
        "title_original",
        "description",
        "culture",
        "culture_canonical",
        "period",
        "object_type",
        "material",
        "technique",
        "classification",
        "image_local_path",
        "image_url",
        "object_url",
        "image_width",
        "image_height",
        "image_bytes",
        "image_sha256",
    ]

    corpus_profiles = corpus[
        corpus_columns
    ].copy()

    corpus_profiles = corpus_profiles.rename(
        columns={
            "item_id": "candidate_item_id",
            "museum": "corpus_museum",
        }
    )

    text_profiles = text_inputs[
        [
            "item_id",
            "museum",
            "text_visual",
            "text_visual_version",
            "text_metadata",
            "text_metadata_version",
        ]
    ].copy()

    text_profiles = text_profiles.rename(
        columns={
            "item_id": "candidate_item_id",
            "museum": "text_museum",
        }
    )

    profiles = persistent_hubs.merge(
        corpus_profiles,
        on="candidate_item_id",
        how="left",
        validate="one_to_one",
    )

    profiles = profiles.merge(
        text_profiles,
        on="candidate_item_id",
        how="left",
        validate="one_to_one",
    )

    missing_corpus = profiles[
        "corpus_museum"
    ].isna()

    if missing_corpus.any():
        missing = ", ".join(
            profiles.loc[
                missing_corpus,
                "candidate_item_id",
            ].astype(str)
        )

        raise ValueError(
            "Existen hubs ausentes en el corpus: "
            f"{missing}"
        )

    missing_texts = profiles[
        "text_museum"
    ].isna()

    if missing_texts.any():
        missing = ", ".join(
            profiles.loc[
                missing_texts,
                "candidate_item_id",
            ].astype(str)
        )

        raise ValueError(
            "Existen hubs ausentes en las entradas "
            f"textuales: {missing}"
        )

    candidate_museums = profiles[
        "candidate_museum"
    ].astype(str)

    if not candidate_museums.equals(
        profiles["corpus_museum"].astype(str)
    ):
        raise ValueError(
            "candidate_museum no coincide con el museo "
            "registrado en el corpus."
        )

    if not candidate_museums.equals(
        profiles["text_museum"].astype(str)
    ):
        raise ValueError(
            "candidate_museum no coincide con el museo "
            "registrado en las entradas textuales."
        )

    profiles = profiles.drop(
        columns=[
            "corpus_museum",
            "text_museum",
        ]
    )

    profiles["image_exists"] = [
        _resolve_image_exists(
            image_local_path=str(path),
            repository_root=repository_root,
        )
        for path in profiles["image_local_path"]
    ]

    profiles["description_available"] = (
        profiles["description"]
        .astype(str)
        .str.strip()
        .ne("")
    )

    profiles["text_visual_length"] = (
        profiles["text_visual"]
        .astype(str)
        .str.len()
    )

    profiles["text_metadata_length"] = (
        profiles["text_metadata"]
        .astype(str)
        .str.len()
    )

    return profiles.sort_values(
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


def _build_attraction_events(
    *,
    per_query: pd.DataFrame,
    corpus: pd.DataFrame,
    persistent_hubs: pd.DataFrame,
) -> pd.DataFrame:
    """Construye el detalle de consultas atraídas por los hubs."""

    hub_ids = set(
        persistent_hubs[
            "candidate_item_id"
        ].astype(str)
    )

    events = per_query.loc[
        per_query[
            "top1_item_id"
        ].astype(str).isin(hub_ids)
    ].copy()

    events = events.rename(
        columns={
            "item_id": "query_item_id",
            "museum": "query_museum",
            "top1_item_id": "candidate_item_id",
            "top1_museum": "candidate_museum",
        }
    )

    query_metadata = corpus[
        [
            "item_id",
            "title",
            "culture_canonical",
            "object_type",
            "image_local_path",
        ]
    ].rename(
        columns={
            "item_id": "query_item_id",
            "title": "query_title",
            "culture_canonical": (
                "query_culture_canonical"
            ),
            "object_type": "query_object_type",
            "image_local_path": (
                "query_image_local_path"
            ),
        }
    )

    candidate_metadata = corpus[
        [
            "item_id",
            "title",
            "culture_canonical",
            "object_type",
            "image_local_path",
        ]
    ].rename(
        columns={
            "item_id": "candidate_item_id",
            "title": "candidate_title",
            "culture_canonical": (
                "candidate_culture_canonical"
            ),
            "object_type": "candidate_object_type",
            "image_local_path": (
                "candidate_image_local_path"
            ),
        }
    )

    events = events.merge(
        query_metadata,
        on="query_item_id",
        how="left",
        validate="many_to_one",
    )

    events = events.merge(
        candidate_metadata,
        on="candidate_item_id",
        how="left",
        validate="many_to_one",
    )

    events["self_match"] = (
        events["query_item_id"].astype(str)
        == events["candidate_item_id"].astype(str)
    )

    events["same_museum"] = (
        events["query_museum"].astype(str)
        == events["candidate_museum"].astype(str)
    )

    events["cross_museum"] = ~events[
        "same_museum"
    ]

    events["same_culture_canonical"] = (
        _nonempty_equal(
            events["query_culture_canonical"],
            events["candidate_culture_canonical"],
        )
    )

    events["same_object_type"] = (
        _nonempty_equal(
            events["query_object_type"],
            events["candidate_object_type"],
        )
    )

    return events.sort_values(
        [
            "configuration",
            "candidate_item_id",
            "self_match",
            "matched_rank",
            "query_index",
        ],
        ascending=[
            True,
            True,
            False,
            True,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)


def _build_hub_configuration_counts(
    *,
    candidate_counts: pd.DataFrame,
    attraction_events: pd.DataFrame,
    persistent_hubs: pd.DataFrame,
) -> pd.DataFrame:
    """Resume cada hub dentro de cada configuración."""

    hub_ids = set(
        persistent_hubs[
            "candidate_item_id"
        ].astype(str)
    )

    counts = candidate_counts.loc[
        candidate_counts[
            "candidate_item_id"
        ].astype(str).isin(hub_ids)
    ].copy()

    event_flags = attraction_events.copy()

    event_flags["query_is_cma"] = (
        event_flags["query_museum"]
        .astype(str)
        .eq("CMA")
    )

    event_flags["query_is_met"] = (
        event_flags["query_museum"]
        .astype(str)
        .eq("MET")
    )

    event_summary = (
        event_flags.groupby(
            [
                "configuration",
                "candidate_item_id",
            ],
            as_index=False,
            sort=True,
        )
        .agg(
            attracted_query_count=(
                "query_index",
                "size",
            ),
            attracted_cma_queries=(
                "query_is_cma",
                "sum",
            ),
            attracted_met_queries=(
                "query_is_met",
                "sum",
            ),
            cross_museum_queries=(
                "cross_museum",
                "sum",
            ),
            same_culture_queries=(
                "same_culture_canonical",
                "sum",
            ),
            same_object_type_queries=(
                "same_object_type",
                "sum",
            ),
        )
    )

    result = counts.merge(
        event_summary,
        on=[
            "configuration",
            "candidate_item_id",
        ],
        how="left",
        validate="one_to_one",
    )

    integer_columns = [
        "attracted_query_count",
        "attracted_cma_queries",
        "attracted_met_queries",
        "cross_museum_queries",
        "same_culture_queries",
        "same_object_type_queries",
    ]

    result[integer_columns] = (
        result[integer_columns]
        .fillna(0)
        .astype(np.int64)
    )

    if not (
        result["top1_count"]
        == result["attracted_query_count"]
    ).all():
        raise ValueError(
            "Los eventos de atracción no coinciden con "
            "los conteos Top-1 materializados."
        )

    result["cross_museum_share"] = np.where(
        result["top1_count"] > 0,
        (
            result["cross_museum_queries"]
            / result["top1_count"]
        ),
        0.0,
    )

    result["same_culture_share"] = np.where(
        result["top1_count"] > 0,
        (
            result["same_culture_queries"]
            / result["top1_count"]
        ),
        0.0,
    )

    return result.sort_values(
        [
            "configuration",
            "top1_count",
            "false_top1_count",
            "candidate_item_id",
        ],
        ascending=[
            True,
            False,
            False,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)


def _build_audit_summary(
    *,
    per_query: pd.DataFrame,
    hub_configuration_counts: pd.DataFrame,
) -> pd.DataFrame:
    """Resume la captura de consultas por hubs persistentes."""

    total_queries = (
        per_query.groupby(
            "configuration",
            sort=True,
        )
        .size()
        .rename("total_queries")
        .reset_index()
    )

    hub_summary = (
        hub_configuration_counts.groupby(
            "configuration",
            as_index=False,
            sort=True,
        )
        .agg(
            persistent_hub_candidates=(
                "candidate_item_id",
                "nunique",
            ),
            active_persistent_hubs=(
                "top1_count",
                lambda values: int(
                    np.count_nonzero(
                        np.asarray(values) > 0
                    )
                ),
            ),
            top1_events_to_persistent_hubs=(
                "top1_count",
                "sum",
            ),
            false_top1_events_to_persistent_hubs=(
                "false_top1_count",
                "sum",
            ),
            cross_museum_events=(
                "cross_museum_queries",
                "sum",
            ),
        )
    )

    summary = total_queries.merge(
        hub_summary,
        on="configuration",
        how="left",
        validate="one_to_one",
    )

    summary[
        "persistent_hub_capture_share"
    ] = (
        summary[
            "top1_events_to_persistent_hubs"
        ]
        / summary["total_queries"]
    )

    summary[
        "persistent_hub_false_top1_share"
    ] = (
        summary[
            "false_top1_events_to_persistent_hubs"
        ]
        / summary["total_queries"]
    )

    return summary


def build_openclip_hub_audit(
    *,
    corpus: pd.DataFrame,
    text_inputs: pd.DataFrame,
    per_query: pd.DataFrame,
    persistent_hubs: pd.DataFrame,
    candidate_counts: pd.DataFrame,
    repository_root: Path,
) -> OpenCLIPHubAudit:
    """Construye una auditoría enriquecida de hubs persistentes."""

    _require_columns(
        corpus,
        required=_CORPUS_REQUIRED_COLUMNS,
        frame_name="corpus",
    )
    _require_columns(
        text_inputs,
        required=_TEXT_REQUIRED_COLUMNS,
        frame_name="text_inputs",
    )
    _require_columns(
        persistent_hubs,
        required=_PERSISTENT_REQUIRED_COLUMNS,
        frame_name="persistent_hubs",
    )
    _require_columns(
        candidate_counts,
        required=_COUNTS_REQUIRED_COLUMNS,
        frame_name="candidate_counts",
    )
    _require_columns(
        per_query,
        required=_PER_QUERY_REQUIRED_COLUMNS,
        frame_name="per_query",
    )

    _require_unique_ids(
        corpus,
        identifier="item_id",
        frame_name="corpus",
    )
    _require_unique_ids(
        text_inputs,
        identifier="item_id",
        frame_name="text_inputs",
    )
    _require_unique_ids(
        persistent_hubs,
        identifier="candidate_item_id",
        frame_name="persistent_hubs",
    )

    hub_profiles = _build_hub_profiles(
        corpus=corpus,
        text_inputs=text_inputs,
        persistent_hubs=persistent_hubs,
        repository_root=repository_root,
    )

    attraction_events = _build_attraction_events(
        per_query=per_query,
        corpus=corpus,
        persistent_hubs=persistent_hubs,
    )

    hub_configuration_counts = (
        _build_hub_configuration_counts(
            candidate_counts=candidate_counts,
            attraction_events=attraction_events,
            persistent_hubs=persistent_hubs,
        )
    )

    summary = _build_audit_summary(
        per_query=per_query,
        hub_configuration_counts=(
            hub_configuration_counts
        ),
    )

    return OpenCLIPHubAudit(
        hub_profiles=hub_profiles,
        hub_configuration_counts=(
            hub_configuration_counts
        ),
        attraction_events=attraction_events,
        summary=summary,
    )