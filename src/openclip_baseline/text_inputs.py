"""Construcción reproducible de entradas textuales para OpenCLIP.

Se generan dos vistas:

- ``visual_semantic_v1``: descripción visual y material del objeto,
  sin etiquetas culturales utilizadas posteriormente en evaluación.
- ``metadata_rich_v1``: vista ampliada para una ablación explícita
  de metadatos curatoriales.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

import pandas as pd


VISUAL_TEXT_VERSION = "visual_semantic_v1"
METADATA_TEXT_VERSION = "metadata_rich_v1"

KEY_COLUMN = "item_id"
MUSEUM_COLUMN = "museum"

OUTPUT_COLUMNS: tuple[str, ...] = (
    KEY_COLUMN,
    MUSEUM_COLUMN,
    "text_visual",
    "text_visual_version",
    "text_visual_sha256",
    "text_metadata",
    "text_metadata_version",
    "text_metadata_sha256",
)

VISUAL_FIELDS: tuple[tuple[str, str], ...] = (
    ("title", "Title"),
    ("object_type", "Object type"),
    ("material", "Material"),
    ("technique", "Technique"),
    ("description", "Description"),
)

METADATA_FIELDS: tuple[tuple[str, str], ...] = (
    ("culture_canonical", "Culture"),
    ("period", "Period"),
    ("classification", "Classification"),
    ("country", "Country"),
    ("region", "Region"),
)

WHITESPACE_PATTERN = re.compile(r"\s+")


def _require_columns(
    corpus: pd.DataFrame,
    required_columns: Iterable[str],
) -> None:
    """Comprueba que existan las columnas estructurales requeridas."""

    missing_columns = sorted(
        set(required_columns).difference(corpus.columns)
    )

    if missing_columns:
        missing = ", ".join(missing_columns)

        raise ValueError(
            "El corpus no contiene las columnas requeridas: "
            f"{missing}"
        )


def _normalize_text_value(
    value: object,
) -> str:
    """Convierte un valor en texto limpio y estable."""

    if value is None or pd.isna(value):
        return ""

    text = str(value)

    # Evita que separadores internos rompan la plantilla externa.
    text = text.replace(";", ",")

    return WHITESPACE_PATTERN.sub(
        " ",
        text,
    ).strip()


def _resolve_culture_value(
    row: pd.Series,
) -> str:
    """Prioriza la cultura canónica y usa la original como respaldo."""

    canonical = _normalize_text_value(
        row.get("culture_canonical", "")
    )

    if canonical:
        return canonical

    return _normalize_text_value(
        row.get("culture", "")
    )


def _field_value(
    row: pd.Series,
    column: str,
) -> str:
    """Obtiene un campo normalizado, con reglas de respaldo."""

    if column == "culture_canonical":
        return _resolve_culture_value(row)

    return _normalize_text_value(
        row.get(column, "")
    )


def _build_segments(
    row: pd.Series,
    fields: Iterable[tuple[str, str]],
) -> list[str]:
    """Construye segmentos etiquetados y omite campos vacíos."""

    segments: list[str] = []

    for column, label in fields:
        value = _field_value(
            row,
            column,
        )

        if not value:
            continue

        segments.append(
            f"{label}: {value}"
        )

    return segments


def _build_visual_text(
    row: pd.Series,
) -> str:
    """Construye la vista visual-semántica sin ground truth cultural."""

    return "; ".join(
        _build_segments(
            row,
            VISUAL_FIELDS,
        )
    )


def _build_metadata_text(
    row: pd.Series,
) -> str:
    """Construye la vista ampliada para la ablación de metadatos."""

    visual_segments = _build_segments(
        row,
        VISUAL_FIELDS,
    )

    metadata_segments = _build_segments(
        row,
        METADATA_FIELDS,
    )

    return "; ".join(
        [
            *visual_segments,
            *metadata_segments,
        ]
    )


def _sha256_text(
    text: str,
) -> str:
    """Calcula un hash SHA-256 estable de una cadena UTF-8."""

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def _validate_identifiers(
    corpus: pd.DataFrame,
) -> None:
    """Valida identificadores y museo sin alterar la fuente."""

    item_ids = (
        corpus[KEY_COLUMN]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    if item_ids.eq("").any():
        raise ValueError(
            "El corpus contiene valores item_id vacíos."
        )

    duplicated = item_ids.duplicated(
        keep=False
    )

    if duplicated.any():
        examples = (
            item_ids.loc[duplicated]
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise ValueError(
            "El corpus contiene valores item_id duplicados: "
            f"{examples}"
        )


def build_openclip_text_inputs(
    corpus: pd.DataFrame,
) -> pd.DataFrame:
    """Construye vistas textuales sin modificar el corpus fuente.

    Parameters
    ----------
    corpus:
        Corpus multimodal con una fila por objeto cultural.

    Returns
    -------
    pandas.DataFrame
        Tabla con identificadores, las dos vistas textuales,
        versiones de plantilla y hashes SHA-256.
    """

    _require_columns(
        corpus,
        (
            KEY_COLUMN,
            MUSEUM_COLUMN,
        ),
    )

    _validate_identifiers(corpus)

    source = corpus.copy(deep=True)

    output = pd.DataFrame(
        {
            KEY_COLUMN: (
                source[KEY_COLUMN]
                .astype("string")
                .fillna("")
                .str.strip()
            ),
            MUSEUM_COLUMN: (
                source[MUSEUM_COLUMN]
                .astype("string")
                .fillna("")
                .str.strip()
                .str.upper()
            ),
        }
    )

    output["text_visual"] = source.apply(
        _build_visual_text,
        axis=1,
    )

    output["text_visual_version"] = (
        VISUAL_TEXT_VERSION
    )

    output["text_visual_sha256"] = output[
        "text_visual"
    ].map(_sha256_text)

    output["text_metadata"] = source.apply(
        _build_metadata_text,
        axis=1,
    )

    output["text_metadata_version"] = (
        METADATA_TEXT_VERSION
    )

    output["text_metadata_sha256"] = output[
        "text_metadata"
    ].map(_sha256_text)

    return output.loc[
        :,
        list(OUTPUT_COLUMNS),
    ].copy()