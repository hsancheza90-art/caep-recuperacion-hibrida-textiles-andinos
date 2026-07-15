from __future__ import annotations

import pandas as pd

from caep.evaluation.category_audit import (
    clean_raw_label,
    normalize_label,
)


def audit_field_pair(
    corpus: pd.DataFrame,
    left_field: str,
    right_field: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compara dos campos registro por registro y por museo.

    La comparación es únicamente lexical: Unicode, espacios y
    mayúsculas/minúsculas. No se interpretan equivalencias semánticas.
    """
    required = {
        "museum",
        "item_id",
        left_field,
        right_field,
    }

    missing = required.difference(corpus.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(missing)}"
        )

    summary_rows: list[dict[str, object]] = []
    pair_frames: list[pd.DataFrame] = []

    museum_groups = [
        ("ALL", corpus),
        ("MET", corpus.loc[corpus["museum"].eq("MET")]),
        ("CMA", corpus.loc[corpus["museum"].eq("CMA")]),
    ]

    for museum, subset in museum_groups:
        working = subset[
            ["item_id", left_field, right_field]
        ].copy()

        working["left_label"] = working[left_field].map(
            clean_raw_label
        )
        working["right_label"] = working[right_field].map(
            clean_raw_label
        )

        working["left_key"] = working[left_field].map(
            normalize_label
        )
        working["right_key"] = working[right_field].map(
            normalize_label
        )

        comparable = (
            working["left_key"].ne("")
            & working["right_key"].ne("")
        )

        equal = (
            comparable
            & working["left_key"].eq(
                working["right_key"]
            )
        )

        comparable_count = int(comparable.sum())
        equal_count = int(equal.sum())

        left_unique = int(
            working.loc[
                working["left_key"].ne(""),
                "left_key",
            ].nunique()
        )

        right_unique = int(
            working.loc[
                working["right_key"].ne(""),
                "right_key",
            ].nunique()
        )

        summary_rows.append(
            {
                "museum": museum,
                "left_field": left_field,
                "right_field": right_field,
                "total_rows": len(working),
                "comparable_rows": comparable_count,
                "equal_rows": equal_count,
                "different_rows": (
                    comparable_count - equal_count
                ),
                "equality_rate": (
                    equal_count / comparable_count
                    if comparable_count
                    else 0.0
                ),
                "left_unique_categories": left_unique,
                "right_unique_categories": right_unique,
                "left_is_constant": left_unique == 1,
                "right_is_constant": right_unique == 1,
            }
        )

        pairs = (
            working.loc[
                comparable,
                [
                    "left_label",
                    "left_key",
                    "right_label",
                    "right_key",
                ],
            ]
            .groupby(
                [
                    "left_label",
                    "left_key",
                    "right_label",
                    "right_key",
                ],
                dropna=False,
            )
            .size()
            .reset_index(name="count")
        )

        pairs.insert(0, "museum", museum)
        pairs.insert(1, "left_field", left_field)
        pairs.insert(2, "right_field", right_field)

        pairs["same_normalized_label"] = (
            pairs["left_key"].eq(pairs["right_key"])
        )

        pair_frames.append(pairs)

    summary = pd.DataFrame(summary_rows)

    pairs = pd.concat(
        pair_frames,
        ignore_index=True,
    )

    pairs = pairs.sort_values(
        [
            "museum",
            "count",
            "left_label",
            "right_label",
        ],
        ascending=[True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)

    return summary, pairs