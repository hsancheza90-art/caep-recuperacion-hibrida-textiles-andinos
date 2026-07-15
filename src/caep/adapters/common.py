from __future__ import annotations

import re
from datetime import datetime, timezone

import pandas as pd


def clean_text(value: object) -> str:
    """Normaliza espacios sin alterar el contenido semántico."""
    if value is None:
        return ""

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null"}:
        return ""

    return re.sub(r"\s+", " ", text)


def first_non_empty(*values: object) -> str:
    """Devuelve el primer valor textual no vacío."""
    for value in values:
        cleaned = clean_text(value)
        if cleaned:
            return cleaned
    return ""


def build_item_id(museum: str, source_object_id: object) -> str:
    source_id = clean_text(source_object_id)

    if not source_id:
        raise ValueError(f"Identificador vacío para museo {museum}")

    return f"{museum}:{source_id}"


def utc_processing_timestamp() -> str:
    """Genera una marca temporal ISO 8601 en UTC."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def assert_required_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    source_name: str,
) -> None:
    missing = required_columns.difference(frame.columns)

    if missing:
        raise ValueError(
            f"{source_name}: faltan columnas requeridas: {sorted(missing)}"
        )


def validate_unique_non_empty(
    series: pd.Series,
    field_name: str,
) -> None:
    values = series.astype(str).map(clean_text)

    empty_count = values.eq("").sum()
    duplicate_count = values[values.ne("")].duplicated().sum()

    if empty_count:
        raise ValueError(
            f"{field_name}: contiene {empty_count} valores vacíos"
        )

    if duplicate_count:
        raise ValueError(
            f"{field_name}: contiene {duplicate_count} duplicados"
        )