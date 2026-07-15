"""Construcción del corpus multimodal derivado.

Este módulo integra el corpus culturalmente enriquecido con el
manifiesto técnico de imágenes sin modificar ninguno de los dos
artefactos fuente.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

import pandas as pd


KEY_COLUMN = "item_id"

IMAGE_COLUMN_MAP: dict[str, str] = {
    "image_local_path": "image_local_path",
    "download_status": "image_download_status",
    "acquisition_action": "image_acquisition_action",
    "final_url": "image_final_url",
    "content_type": "image_content_type",
    "image_bytes": "image_bytes",
    "image_width": "image_width",
    "image_height": "image_height",
    "image_format": "image_format",
    "sha256": "image_sha256",
    "download_version": "image_download_version",
}

WINDOWS_ABSOLUTE_PATH = re.compile(
    r"^[A-Za-z]:[\\/]"
)


def _require_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    source_name: str,
) -> None:
    """Comprueba que un DataFrame contenga las columnas requeridas."""

    missing_columns = sorted(
        required_columns.difference(frame.columns)
    )

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(
            f"El {source_name} no contiene las columnas "
            f"requeridas: {missing}"
        )


def _prepare_source(
    frame: pd.DataFrame,
    source_name: str,
    required_columns: set[str],
) -> pd.DataFrame:
    """Copia y valida una fuente sin modificar el DataFrame original."""

    _require_columns(
        frame=frame,
        required_columns=required_columns,
        source_name=source_name,
    )

    prepared = frame.copy(deep=True)

    prepared[KEY_COLUMN] = (
        prepared[KEY_COLUMN]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    empty_keys = prepared[KEY_COLUMN].eq("")

    if empty_keys.any():
        raise ValueError(
            f"El {source_name} contiene valores item_id vacíos."
        )

    duplicated_keys = prepared[KEY_COLUMN].duplicated(
        keep=False
    )

    if duplicated_keys.any():
        duplicated_values = (
            prepared.loc[
                duplicated_keys,
                KEY_COLUMN,
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            f"El {source_name} contiene item_id duplicados: "
            f"{duplicated_values}"
        )

    return prepared


def _is_portable_image_path(value: str) -> bool:
    """Indica si una ruta es relativa, POSIX y está bajo data/images."""

    path_value = value.strip()

    if not path_value:
        return False

    if WINDOWS_ABSOLUTE_PATH.match(path_value):
        return False

    if "\\" in path_value:
        return False

    path = PurePosixPath(path_value)

    if path.is_absolute():
        return False

    if ".." in path.parts:
        return False

    if len(path.parts) < 3:
        return False

    if path.parts[:2] != ("data", "images"):
        return False

    return True


def _validate_image_paths(
    manifest: pd.DataFrame,
) -> None:
    """Valida que todas las rutas de imagen sean portables."""

    paths = (
        manifest["image_local_path"]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    portable = paths.map(_is_portable_image_path)

    if portable.all():
        return

    invalid_rows = manifest.loc[
        ~portable,
        [KEY_COLUMN, "image_local_path"],
    ]

    examples = invalid_rows.head(5).to_dict(
        orient="records"
    )

    raise ValueError(
        "El manifiesto contiene una ruta de imagen no portable. "
        f"Ejemplos: {examples}"
    )


def _validate_complete_coverage(
    cultural_corpus: pd.DataFrame,
    image_manifest: pd.DataFrame,
) -> None:
    """Exige correspondencia exacta entre las dos fuentes."""

    corpus_ids = set(
        cultural_corpus[KEY_COLUMN]
    )
    manifest_ids = set(
        image_manifest[KEY_COLUMN]
    )

    missing_in_manifest = sorted(
        corpus_ids.difference(manifest_ids)
    )

    extra_in_manifest = sorted(
        manifest_ids.difference(corpus_ids)
    )

    if missing_in_manifest or extra_in_manifest:
        raise ValueError(
            "Se detectó cobertura incompleta entre el corpus "
            "cultural y el manifiesto de imágenes. "
            f"Faltantes en manifiesto: "
            f"{len(missing_in_manifest)}; "
            f"adicionales en manifiesto: "
            f"{len(extra_in_manifest)}."
        )


def _validate_museum_consistency(
    cultural_corpus: pd.DataFrame,
    image_manifest: pd.DataFrame,
) -> None:
    """Comprueba que cada item_id conserve el mismo museo."""

    manifest_museum = (
        image_manifest
        .set_index(KEY_COLUMN)["museum"]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.upper()
    )

    corpus_museum = (
        cultural_corpus["museum"]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.upper()
    )

    mapped_museum = cultural_corpus[
        KEY_COLUMN
    ].map(manifest_museum)

    inconsistent = corpus_museum.ne(
        mapped_museum
    )

    if inconsistent.any():
        examples = (
            cultural_corpus.loc[
                inconsistent,
                [KEY_COLUMN, "museum"],
            ]
            .head(5)
            .to_dict(orient="records")
        )

        raise ValueError(
            "Se detectaron museos inconsistentes entre el corpus "
            f"y el manifiesto. Ejemplos: {examples}"
        )

def _normalize_technical_values(
    values: pd.Series,
    column: str,
) -> pd.Series:
    """Normaliza valores técnicos solo para compararlos."""

    normalized = (
        values
        .astype("string")
        .fillna("")
        .str.strip()
    )

    if column == "image_local_path":
        normalized = normalized.str.replace(
            "\\",
            "/",
            regex=False,
        )

    if column == "image_sha256":
        normalized = normalized.str.lower()

    return normalized


def _merge_technical_columns(
    cultural_corpus: pd.DataFrame,
    technical_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Integra columnas técnicas sin duplicarlas.

    Las columnas ausentes se agregan. Las columnas existentes y
    vacías se completan. Si existen valores no vacíos incompatibles,
    la construcción se rechaza.
    """

    derived = cultural_corpus.copy(deep=True)

    for column in IMAGE_COLUMN_MAP.values():
        incoming = derived[KEY_COLUMN].map(
            technical_manifest[column]
        )

        incoming = (
            incoming
            .astype("string")
            .fillna("")
        )

        if column not in derived.columns:
            derived[column] = incoming
            continue

        existing = (
            derived[column]
            .astype("string")
            .fillna("")
        )

        existing_normalized = _normalize_technical_values(
            existing,
            column,
        )

        incoming_normalized = _normalize_technical_values(
            incoming,
            column,
        )

        conflicts = (
            existing_normalized.ne("")
            & incoming_normalized.ne("")
            & existing_normalized.ne(
                incoming_normalized
            )
        )

        if conflicts.any():
            examples = pd.DataFrame(
                {
                    KEY_COLUMN: derived.loc[
                        conflicts,
                        KEY_COLUMN,
                    ],
                    "cultural_value": existing.loc[
                        conflicts
                    ],
                    "manifest_value": incoming.loc[
                        conflicts
                    ],
                }
            ).head(5)

            raise ValueError(
                "Se detectó un conflicto en la columna técnica "
                f"{column}. Ejemplos: "
                f"{examples.to_dict(orient='records')}"
            )

        merged = existing.copy()

        # Cuando el manifiesto contiene un valor, se utiliza como
        # representación técnica canónica.
        use_manifest = incoming_normalized.ne("")

        merged.loc[use_manifest] = incoming.loc[
            use_manifest
        ]

        derived[column] = merged

    return derived

def build_multimodal_corpus(
    cultural_corpus: pd.DataFrame,
    image_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Construye el corpus multimodal sin modificar las fuentes.

    La integración utiliza ``item_id`` como clave canónica y exige
    una correspondencia uno a uno y completa. El orden de las filas
    del corpus cultural se conserva.
    """

    cultural_required = {
        KEY_COLUMN,
        "museum",
    }

    manifest_required = {
        KEY_COLUMN,
        "museum",
        *IMAGE_COLUMN_MAP.keys(),
    }

    cultural = _prepare_source(
        frame=cultural_corpus,
        source_name="corpus cultural",
        required_columns=cultural_required,
    )

    manifest = _prepare_source(
        frame=image_manifest,
        source_name="manifiesto",
        required_columns=manifest_required,
    )

    _validate_complete_coverage(
        cultural_corpus=cultural,
        image_manifest=manifest,
    )

    _validate_museum_consistency(
        cultural_corpus=cultural,
        image_manifest=manifest,
    )

    _validate_image_paths(
        manifest=manifest,
    )


    technical = (
        manifest.loc[
            :,
            [
                KEY_COLUMN,
                *IMAGE_COLUMN_MAP.keys(),
            ],
        ]
        .rename(columns=IMAGE_COLUMN_MAP)
        .set_index(KEY_COLUMN)
    )

    derived = _merge_technical_columns(
        cultural_corpus=cultural,
        technical_manifest=technical,
    )

    if len(derived) != len(cultural):
        raise RuntimeError(
            "La construcción del corpus multimodal alteró "
            "inesperadamente el número de filas."
        )

    if derived[KEY_COLUMN].tolist() != cultural[
        KEY_COLUMN
    ].tolist():
        raise RuntimeError(
            "La construcción del corpus multimodal alteró "
            "inesperadamente el orden de las filas."
        )

    return derived