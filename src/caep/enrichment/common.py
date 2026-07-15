from __future__ import annotations

import re

import pandas as pd


def clean_text(value: object) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null"}:
        return ""

    return re.sub(r"\s+", " ", text)


def normalize_key(series: pd.Series) -> pd.Series:
    return series.map(clean_text)


def coalesce_series(
    preferred: pd.Series,
    fallback: pd.Series,
) -> pd.Series:
    preferred_clean = preferred.map(clean_text)
    fallback_clean = fallback.map(clean_text)

    return preferred_clean.where(
        preferred_clean.ne(""),
        fallback_clean,
    )


def assert_unique_non_empty_key(
    frame: pd.DataFrame,
    key: str,
    source_name: str,
) -> None:
    values = normalize_key(frame[key])

    empty_count = int(values.eq("").sum())
    duplicate_count = int(
        values[values.ne("")].duplicated().sum()
    )

    if empty_count:
        raise ValueError(
            f"{source_name}.{key}: {empty_count} claves vacías"
        )

    if duplicate_count:
        raise ValueError(
            f"{source_name}.{key}: "
            f"{duplicate_count} claves duplicadas"
        )


def assert_complete_match(
    frame: pd.DataFrame,
    indicator_column: str,
    expected_rows: int,
    source_name: str,
) -> None:
    if len(frame) != expected_rows:
        raise ValueError(
            f"{source_name}: se esperaban {expected_rows} filas, "
            f"pero se obtuvieron {len(frame)}."
        )

    unmatched = int(
        frame[indicator_column].ne("both").sum()
    )

    if unmatched:
        raise ValueError(
            f"{source_name}: {unmatched} registros sin correspondencia."
        )