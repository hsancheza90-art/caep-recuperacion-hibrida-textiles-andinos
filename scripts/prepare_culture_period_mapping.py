from __future__ import annotations

from pathlib import Path

import pandas as pd

from caep.datasets.build_enriched_corpus import (
    build_enriched_corpus,
)
from caep.evaluation.label_inventory import (
    build_cross_museum_candidates,
    build_label_inventory,
    build_mapping_template,
)


FIELDS = [
    "culture",
    "period",
]

INVENTORY_OUTPUT = Path(
    "outputs/reports/culture_period_inventory_v1.csv"
)

CANDIDATES_OUTPUT = Path(
    "outputs/reports/culture_period_lexical_candidates_v1.csv"
)

SUMMARY_OUTPUT = Path(
    "outputs/reports/culture_period_inventory_summary_v1.csv"
)

MAPPING_OUTPUT = Path(
    "config/mappings/culture_period_mapping_v1.csv"
)


def build_summary(
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    return (
        inventory
        .groupby(
            ["field", "museum"],
            as_index=False,
        )
        .agg(
            source_categories=("source_key", "nunique"),
            represented_records=("record_count", "sum"),
        )
        .sort_values(
            ["field", "museum"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def print_inventory(
    inventory: pd.DataFrame,
    field: str,
    museum: str,
    limit: int = 25,
) -> None:
    subset = inventory.loc[
        inventory["field"].eq(field)
        & inventory["museum"].eq(museum)
    ].head(limit)

    print("\n" + "=" * 118)
    print(f"{field.upper()} — {museum}")
    print("=" * 118)

    print(
        subset[
            [
                "source_label",
                "record_count",
                "example_titles",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    corpus = build_enriched_corpus()

    inventory = build_label_inventory(
        corpus=corpus,
        fields=FIELDS,
        example_limit=3,
    )

    candidates = build_cross_museum_candidates(
        inventory=inventory,
        top_n_per_source=5,
    )

    mapping = build_mapping_template(inventory)
    summary = build_summary(inventory)

    INVENTORY_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MAPPING_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    inventory.to_csv(
        INVENTORY_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    candidates.to_csv(
        CANDIDATES_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    mapping.to_csv(
        MAPPING_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nRESUMEN")
    print("=" * 118)
    print(summary.to_string(index=False))

    for field in FIELDS:
        for museum in ["MET", "CMA"]:
            print_inventory(
                inventory,
                field,
                museum,
            )

    print("\nSUGERENCIAS LEXICALES PRINCIPALES")
    print("=" * 118)

    useful_candidates = candidates.loc[
        candidates["candidate_score"].gt(0)
    ].head(40)

    if useful_candidates.empty:
        print("No se encontraron coincidencias lexicales.")
    else:
        print(
            useful_candidates[
                [
                    "field",
                    "met_source_label",
                    "cma_source_label",
                    "met_count",
                    "cma_count",
                    "token_jaccard",
                    "substring_match",
                    "common_tokens",
                ]
            ].to_string(index=False)
        )

    print("\nARTEFACTOS")
    print("=" * 118)
    print(f"Inventario: {INVENTORY_OUTPUT}")
    print(f"Candidatos lexicales: {CANDIDATES_OUTPUT}")
    print(f"Resumen: {SUMMARY_OUTPUT}")
    print(f"Plantilla de revisión: {MAPPING_OUTPUT}")


if __name__ == "__main__":
    main()