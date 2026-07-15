from __future__ import annotations

import re
import unicodedata

import pandas as pd

from caep.evaluation.category_audit import (
    clean_raw_label,
    normalize_label,
)


def lexical_tokens(value: object) -> set[str]:
    """
    Genera tokens únicamente para sugerir posibles coincidencias.

    No convierte las sugerencias en equivalencias semánticas.
    """
    text = clean_raw_label(value)

    if not text:
        return set()

    decomposed = unicodedata.normalize("NFKD", text)

    without_accents = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )

    normalized = without_accents.casefold()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)

    return {
        token
        for token in normalized.split()
        if len(token) >= 2
    }


def join_examples(
    values: pd.Series,
    limit: int = 3,
) -> str:
    cleaned = [
        clean_raw_label(value)
        for value in values
        if clean_raw_label(value)
    ]

    unique_values = list(dict.fromkeys(cleaned))

    return " || ".join(unique_values[:limit])


def build_label_inventory(
    corpus: pd.DataFrame,
    fields: list[str],
    example_limit: int = 3,
) -> pd.DataFrame:
    required = {
        "museum",
        "item_id",
        "title",
        *fields,
    }

    missing = required.difference(corpus.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(missing)}"
        )

    rows: list[dict[str, object]] = []

    for field in fields:
        working = corpus[
            ["museum", "item_id", "title", field]
        ].copy()

        working["source_label"] = working[field].map(
            clean_raw_label
        )
        working["source_key"] = working[field].map(
            normalize_label
        )

        working = working.loc[
            working["source_key"].ne("")
        ].copy()

        grouped = working.groupby(
            ["museum", "source_key"],
            sort=True,
            dropna=False,
        )

        for (museum, source_key), group in grouped:
            label_counts = group["source_label"].value_counts()
            maximum = label_counts.max()

            representative_candidates = sorted(
                label_counts[
                    label_counts.eq(maximum)
                ].index.tolist()
            )

            source_label = representative_candidates[0]

            rows.append(
                {
                    "field": field,
                    "museum": museum,
                    "source_key": source_key,
                    "source_label": source_label,
                    "record_count": len(group),
                    "example_item_ids": join_examples(
                        group["item_id"],
                        limit=example_limit,
                    ),
                    "example_titles": join_examples(
                        group["title"],
                        limit=example_limit,
                    ),
                }
            )

    inventory = pd.DataFrame(rows)

    return (
        inventory
        .sort_values(
            [
                "field",
                "museum",
                "record_count",
                "source_label",
            ],
            ascending=[True, True, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def lexical_pair_metrics(
    left_label: str,
    right_label: str,
) -> dict[str, object]:
    left_tokens = lexical_tokens(left_label)
    right_tokens = lexical_tokens(right_label)

    union = left_tokens.union(right_tokens)
    intersection = left_tokens.intersection(right_tokens)

    jaccard = (
        len(intersection) / len(union)
        if union
        else 0.0
    )

    left_compact = " ".join(sorted(left_tokens))
    right_compact = " ".join(sorted(right_tokens))

    substring_match = (
        bool(left_compact)
        and bool(right_compact)
        and (
            left_compact in right_compact
            or right_compact in left_compact
        )
    )

    return {
        "token_jaccard": jaccard,
        "substring_match": substring_match,
        "common_tokens": " | ".join(
            sorted(intersection)
        ),
    }


def build_cross_museum_candidates(
    inventory: pd.DataFrame,
    top_n_per_source: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for field in sorted(inventory["field"].unique()):
        field_inventory = inventory.loc[
            inventory["field"].eq(field)
        ]

        met = field_inventory.loc[
            field_inventory["museum"].eq("MET")
        ]

        cma = field_inventory.loc[
            field_inventory["museum"].eq("CMA")
        ]

        for left in met.itertuples(index=False):
            candidate_rows: list[dict[str, object]] = []

            for right in cma.itertuples(index=False):
                metrics = lexical_pair_metrics(
                    left.source_label,
                    right.source_label,
                )

                candidate_score = max(
                    float(metrics["token_jaccard"]),
                    0.75
                    if metrics["substring_match"]
                    else 0.0,
                )

                candidate_rows.append(
                    {
                        "field": field,
                        "met_source_key": left.source_key,
                        "met_source_label": left.source_label,
                        "met_count": left.record_count,
                        "cma_source_key": right.source_key,
                        "cma_source_label": right.source_label,
                        "cma_count": right.record_count,
                        "token_jaccard": metrics[
                            "token_jaccard"
                        ],
                        "substring_match": metrics[
                            "substring_match"
                        ],
                        "common_tokens": metrics[
                            "common_tokens"
                        ],
                        "candidate_score": candidate_score,
                    }
                )

            candidate_rows = sorted(
                candidate_rows,
                key=lambda row: (
                    -float(row["candidate_score"]),
                    -float(row["token_jaccard"]),
                    str(row["cma_source_label"]),
                ),
            )

            rows.extend(candidate_rows[:top_n_per_source])

    candidates = pd.DataFrame(rows)

    if candidates.empty:
        return candidates

    return (
        candidates
        .sort_values(
            [
                "field",
                "candidate_score",
                "token_jaccard",
                "met_source_label",
                "cma_source_label",
            ],
            ascending=[True, False, False, True, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def build_mapping_template(
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    template = inventory.copy()

    template["canonical_label"] = ""
    template["canonical_level"] = ""
    template["mapping_decision"] = "pendiente"
    template["mapping_rationale"] = ""
    template["reviewer"] = ""
    template["review_date"] = ""

    return template