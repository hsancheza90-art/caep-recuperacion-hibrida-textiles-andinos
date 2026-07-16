"""Construcción de plantillas para revisión manual de hubs OpenCLIP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OpenCLIPHubReviewConfig:
    """Configuración de la muestra para revisión manual."""

    configurations: tuple[str, ...] = (
        "text_visual",
        "text_metadata",
        "text_fused_alpha_0.50",
    )
    max_hubs: int = 16
    max_events_per_hub_configuration: int = 5
    include_self_matches: bool = False

    def __post_init__(self) -> None:
        """Valida la configuración de revisión."""

        if not self.configurations:
            raise ValueError(
                "configurations no puede estar vacío."
            )

        if len(set(self.configurations)) != len(
            self.configurations
        ):
            raise ValueError(
                "Las configuraciones deben ser únicas."
            )

        if self.max_hubs <= 0:
            raise ValueError(
                "max_hubs debe ser mayor que cero."
            )

        if self.max_events_per_hub_configuration <= 0:
            raise ValueError(
                "max_events_per_hub_configuration debe "
                "ser mayor que cero."
            )


_REVIEW_REQUIRED_COLUMNS = {
    "configuration",
    "query_index",
    "query_item_id",
    "query_museum",
    "matched_rank",
    "matched_score",
    "top1_score",
    "candidate_item_id",
    "candidate_museum",
    "self_match",
    "cross_museum",
    "same_culture_canonical",
    "same_object_type",
    "query_title",
    "query_culture_canonical",
    "query_object_type",
    "query_image_local_path",
    "candidate_title",
    "candidate_culture_canonical",
    "candidate_object_type",
    "candidate_image_local_path",
}

_PROFILE_REQUIRED_COLUMNS = {
    "candidate_item_id",
    "candidate_museum",
    "title",
    "culture_canonical",
    "object_type",
    "material",
    "text_visual",
    "text_metadata",
    "total_top1_count",
    "total_false_top1_count",
    "max_top1_count",
}


def _require_columns(
    frame: pd.DataFrame,
    *,
    required: set[str],
    frame_name: str,
) -> None:
    """Comprueba que una tabla contenga el esquema requerido."""

    missing = required - set(frame.columns)

    if missing:
        missing_text = ", ".join(
            sorted(missing)
        )

        raise ValueError(
            f"{frame_name} no contiene las columnas "
            f"requeridas: {missing_text}"
        )


def _as_bool(
    value: object,
) -> bool:
    """Convierte representaciones CSV comunes a booleano."""

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        return bool(value)

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "si",
        "sí",
    }


def _normalize_bool_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Normaliza columnas booleanas procedentes de CSV."""

    result = frame.copy()

    for column in (
        "self_match",
        "cross_museum",
        "same_culture_canonical",
        "same_object_type",
    ):
        result[column] = result[column].map(
            _as_bool
        )

    return result


def _selected_hub_ids(
    hub_profiles: pd.DataFrame,
    *,
    max_hubs: int,
) -> tuple[str, ...]:
    """Selecciona los hubs principales de forma estable."""

    ordered = hub_profiles.sort_values(
        [
            "total_false_top1_count",
            "total_top1_count",
            "max_top1_count",
            "candidate_item_id",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
        kind="stable",
    )

    return tuple(
        ordered[
            "candidate_item_id"
        ]
        .astype(str)
        .head(max_hubs)
    )


def _build_priority_columns(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula indicadores usados para priorizar la revisión."""

    result = events.copy()

    result["score_margin"] = (
        pd.to_numeric(
            result["top1_score"],
            errors="raise",
        )
        - pd.to_numeric(
            result["matched_score"],
            errors="raise",
        )
    )

    result["is_false_top1"] = ~result[
        "self_match"
    ]

    result["priority_cross_museum"] = (
        result["cross_museum"].astype(
            np.int64
        )
    )

    result["priority_different_culture"] = (
        ~result[
            "same_culture_canonical"
        ]
    ).astype(
        np.int64
    )

    result["priority_different_object_type"] = (
        ~result[
            "same_object_type"
        ]
    ).astype(
        np.int64
    )

    return result


def _select_group_events(
    group: pd.DataFrame,
    *,
    maximum: int,
) -> pd.DataFrame:
    """Selecciona los eventos prioritarios de un hub."""

    ordered = group.sort_values(
        [
            "is_false_top1",
            "score_margin",
            "matched_rank",
            "priority_cross_museum",
            "priority_different_culture",
            "priority_different_object_type",
            "query_index",
            "query_item_id",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
        ],
        kind="stable",
    )

    return ordered.head(
        maximum
    )


def _manual_columns() -> dict[str, object]:
    """Define las columnas vacías para revisión humana."""

    return {
        "review_status": "pendiente",
        "visual_similarity": "",
        "photographic_similarity": "",
        "semantic_plausibility": "",
        "dominant_cause": "",
        "duplicate_or_near_duplicate": "",
        "generic_text_influence": "",
        "museum_style_influence": "",
        "hubness_severity": "",
        "recommended_action": "",
        "reviewer": "",
        "review_date": "",
        "review_notes": "",
    }


def build_openclip_hub_review_template(
    *,
    hub_profiles: pd.DataFrame,
    attraction_events: pd.DataFrame,
    config: OpenCLIPHubReviewConfig,
) -> pd.DataFrame:
    """Construye una plantilla determinista de revisión manual."""

    _require_columns(
        hub_profiles,
        required=_PROFILE_REQUIRED_COLUMNS,
        frame_name="hub_profiles",
    )
    _require_columns(
        attraction_events,
        required=_REVIEW_REQUIRED_COLUMNS,
        frame_name="attraction_events",
    )

    normalized_events = _normalize_bool_columns(
        attraction_events
    )

    selected_hubs = _selected_hub_ids(
        hub_profiles,
        max_hubs=config.max_hubs,
    )

    selected = normalized_events.loc[
        normalized_events[
            "candidate_item_id"
        ].astype(str).isin(selected_hubs)
        & normalized_events[
            "configuration"
        ].astype(str).isin(
            config.configurations
        )
    ].copy()

    if not config.include_self_matches:
        selected = selected.loc[
            ~selected["self_match"]
        ].copy()

    if selected.empty:
        raise ValueError(
            "No existen eventos compatibles con la "
            "configuración de revisión."
        )

    selected = _build_priority_columns(
        selected
    )

    sampled_frames = []

    for (
        candidate_item_id,
        configuration,
    ), group in selected.groupby(
        [
            "candidate_item_id",
            "configuration",
        ],
        sort=True,
    ):
        sampled = _select_group_events(
            group,
            maximum=(
                config.max_events_per_hub_configuration
            ),
        ).copy()

        sampled_frames.append(
            sampled
        )

    review = pd.concat(
        sampled_frames,
        ignore_index=True,
    )

    hub_order = {
        item_id: index + 1
        for index, item_id in enumerate(
            selected_hubs
        )
    }

    configuration_order = {
        configuration: index + 1
        for index, configuration in enumerate(
            config.configurations
        )
    }

    review.insert(
        0,
        "hub_priority",
        review["candidate_item_id"].map(
            hub_order
        ),
    )

    review.insert(
        1,
        "configuration_priority",
        review["configuration"].map(
            configuration_order
        ),
    )

    review = review.sort_values(
        [
            "hub_priority",
            "configuration_priority",
            "is_false_top1",
            "score_margin",
            "matched_rank",
            "query_index",
        ],
        ascending=[
            True,
            True,
            False,
            False,
            False,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)

    review.insert(
        0,
        "review_id",
        [
            f"HR-{index:04d}"
            for index in range(
                1,
                len(review) + 1,
            )
        ],
    )

    profile_columns = hub_profiles[
        [
            "candidate_item_id",
            "material",
            "text_visual",
            "text_metadata",
            "total_top1_count",
            "total_false_top1_count",
            "max_top1_count",
        ]
    ].copy()

    profile_columns = profile_columns.rename(
        columns={
            "material": "candidate_material",
            "text_visual": (
                "candidate_text_visual"
            ),
            "text_metadata": (
                "candidate_text_metadata"
            ),
        }
    )

    review = review.merge(
        profile_columns,
        on="candidate_item_id",
        how="left",
        validate="many_to_one",
    )

    for column, default in (
        _manual_columns().items()
    ):
        review[column] = default

    output_columns = [
        "review_id",
        "hub_priority",
        "configuration_priority",
        "configuration",
        "candidate_item_id",
        "candidate_museum",
        "candidate_title",
        "candidate_culture_canonical",
        "candidate_object_type",
        "candidate_material",
        "candidate_image_local_path",
        "candidate_text_visual",
        "candidate_text_metadata",
        "total_top1_count",
        "total_false_top1_count",
        "max_top1_count",
        "query_index",
        "query_item_id",
        "query_museum",
        "query_title",
        "query_culture_canonical",
        "query_object_type",
        "query_image_local_path",
        "matched_rank",
        "matched_score",
        "top1_score",
        "score_margin",
        "is_false_top1",
        "cross_museum",
        "same_culture_canonical",
        "same_object_type",
        "review_status",
        "visual_similarity",
        "photographic_similarity",
        "semantic_plausibility",
        "dominant_cause",
        "duplicate_or_near_duplicate",
        "generic_text_influence",
        "museum_style_influence",
        "hubness_severity",
        "recommended_action",
        "reviewer",
        "review_date",
        "review_notes",
    ]

    return review[
        output_columns
    ]


def build_openclip_hub_review_codebook() -> pd.DataFrame:
    """Construye el vocabulario controlado de revisión."""

    records: list[
        dict[str, str]
    ] = []

    definitions: dict[
        str,
        tuple[
            tuple[str, str],
            ...,
        ]
    ] = {
        "review_status": (
            (
                "pendiente",
                "El caso todavía no fue revisado.",
            ),
            (
                "revisado",
                "La revisión manual fue completada.",
            ),
            (
                "requiere_segunda_revision",
                "El caso necesita validación adicional.",
            ),
        ),
        "visual_similarity": (
            (
                "alta",
                "Las estructuras visuales son claramente similares.",
            ),
            (
                "media",
                "Existe similitud parcial o localizada.",
            ),
            (
                "baja",
                "La similitud visual es débil.",
            ),
            (
                "ninguna",
                "No se observa semejanza visual relevante.",
            ),
        ),
        "photographic_similarity": (
            (
                "alta",
                "Fondo, encuadre, iluminación o escala son muy similares.",
            ),
            (
                "media",
                "Hay coincidencias fotográficas parciales.",
            ),
            (
                "baja",
                "La presentación museográfica influye poco.",
            ),
            (
                "ninguna",
                "No se observa influencia fotográfica.",
            ),
        ),
        "semantic_plausibility": (
            (
                "alta",
                "La recuperación es semánticamente defendible.",
            ),
            (
                "media",
                "La relación semántica es parcial.",
            ),
            (
                "baja",
                "La relación semántica es débil.",
            ),
            (
                "ninguna",
                "La recuperación no es semánticamente plausible.",
            ),
        ),
        "dominant_cause": (
            (
                "similitud_visual",
                "Predominan rasgos visuales compartidos.",
            ),
            (
                "estilo_fotografico",
                "Predomina la presentación museográfica.",
            ),
            (
                "texto_generico",
                "Predominan títulos o campos textuales genéricos.",
            ),
            (
                "coincidencia_cultural",
                "Predomina la cultura o periodo compartido.",
            ),
            (
                "coincidencia_tipo_objeto",
                "Predomina el tipo de objeto compartido.",
            ),
            (
                "duplicado_visual",
                "Existe duplicidad o casi duplicidad visual.",
            ),
            (
                "hubness_embedding",
                "No se observa una causa interpretable directa.",
            ),
            (
                "mixta",
                "Intervienen varias causas de forma comparable.",
            ),
        ),
        "duplicate_or_near_duplicate": (
            (
                "si",
                "Las imágenes parecen duplicadas o casi duplicadas.",
            ),
            (
                "no",
                "No se observa duplicidad.",
            ),
            (
                "dudoso",
                "La duplicidad requiere una segunda revisión.",
            ),
        ),
        "generic_text_influence": (
            (
                "alta",
                "El texto genérico parece influir notablemente.",
            ),
            (
                "media",
                "El texto genérico puede contribuir parcialmente.",
            ),
            (
                "baja",
                "La influencia del texto genérico es pequeña.",
            ),
            (
                "ninguna",
                "No se aprecia influencia del texto genérico.",
            ),
        ),
        "museum_style_influence": (
            (
                "alta",
                "El estilo fotográfico del museo parece dominante.",
            ),
            (
                "media",
                "Existe una influencia museográfica parcial.",
            ),
            (
                "baja",
                "La influencia museográfica es pequeña.",
            ),
            (
                "ninguna",
                "No se observa influencia museográfica.",
            ),
        ),
        "hubness_severity": (
            (
                "critica",
                "El caso afecta ampliamente la recuperación.",
            ),
            (
                "alta",
                "El efecto es frecuente y claramente problemático.",
            ),
            (
                "media",
                "El efecto existe pero es limitado.",
            ),
            (
                "baja",
                "El efecto es menor o semánticamente justificable.",
            ),
        ),
        "recommended_action": (
            (
                "mantener",
                "No se requiere una corrección inmediata.",
            ),
            (
                "revisar_texto",
                "Revisar título, descripción o campos textuales.",
            ),
            (
                "revisar_imagen",
                "Revisar recorte, fondo o calidad de imagen.",
            ),
            (
                "marcar_duplicado",
                "Registrar posible duplicidad.",
            ),
            (
                "excluir_evaluacion",
                "Considerar exclusión justificada de la evaluación.",
            ),
            (
                "aplicar_mitigacion_hubness",
                "Evaluar normalización o reordenamiento antihubness.",
            ),
            (
                "segunda_revision",
                "Solicitar revisión de otro evaluador.",
            ),
        ),
    }

    for field, values in definitions.items():
        for value, definition in values:
            records.append(
                {
                    "field": field,
                    "allowed_value": value,
                    "definition": definition,
                }
            )

    return pd.DataFrame.from_records(
        records
    )


def summarize_hub_review_template(
    review: pd.DataFrame,
) -> pd.DataFrame:
    """Resume la cobertura de la plantilla generada."""

    required = {
        "configuration",
        "candidate_item_id",
        "query_item_id",
        "cross_museum",
        "same_culture_canonical",
        "same_object_type",
    }

    _require_columns(
        review,
        required=required,
        frame_name="review",
    )

    records = []

    for configuration, group in review.groupby(
        "configuration",
        sort=True,
    ):
        records.append(
            {
                "configuration": configuration,
                "review_rows": len(group),
                "reviewed_hubs": int(
                    group[
                        "candidate_item_id"
                    ].nunique()
                ),
                "unique_queries": int(
                    group[
                        "query_item_id"
                    ].nunique()
                ),
                "cross_museum_rows": int(
                    group[
                        "cross_museum"
                    ].map(_as_bool).sum()
                ),
                "same_culture_rows": int(
                    group[
                        "same_culture_canonical"
                    ].map(_as_bool).sum()
                ),
                "same_object_type_rows": int(
                    group[
                        "same_object_type"
                    ].map(_as_bool).sum()
                ),
            }
        )

    return pd.DataFrame.from_records(
        records
    )