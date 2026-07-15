from __future__ import annotations

from pathlib import Path

import pandas as pd

from caep.evaluation.culture_mapping import (
    build_culture_proposals,
    load_culture_taxonomy,
)


INVENTORY_PATH = Path(
    "outputs/reports/culture_period_inventory_v1.csv"
)

PROPOSALS_OUTPUT = Path(
    "config/mappings/culture_mapping_proposals_v1.csv"
)

SUMMARY_OUTPUT = Path(
    "outputs/reports/culture_mapping_proposals_summary_v1.csv"
)


def build_summary(
    proposals: pd.DataFrame,
) -> pd.DataFrame:
    return (
        proposals
        .groupby(
            ["museum", "proposal_decision"],
            as_index=False,
        )
        .agg(
            source_categories=("source_key", "nunique"),
            represented_records=("record_count", "sum"),
        )
        .sort_values(
            [
                "museum",
                "represented_records",
                "proposal_decision",
            ],
            ascending=[True, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def main() -> None:
    inventory = pd.read_csv(
        INVENTORY_PATH,
        dtype=str,
        keep_default_na=False,
    )

    inventory["record_count"] = pd.to_numeric(
        inventory["record_count"],
        errors="raise",
    )

    taxonomy = load_culture_taxonomy()

    proposals = build_culture_proposals(
        inventory,
        taxonomy,
    )

    summary = build_summary(proposals)

    PROPOSALS_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    proposals.to_csv(
        PROPOSALS_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nRESUMEN DE PROPUESTAS CULTURALES")
    print("=" * 118)
    print(summary.to_string(index=False))

    print("\nATRIBUCIONES ELEGIBLES ESTRICTAS")
    print("=" * 118)

    eligible = proposals.loc[
        proposals["strict_eligible"]
    ]

    print(
        eligible[
            [
                "museum",
                "source_label",
                "canonical_components",
                "record_count",
            ]
        ].to_string(index=False)
    )

    print("\nCASOS QUE REQUIEREN REVISIÓN")
    print("=" * 118)

    review = proposals.loc[
        ~proposals["strict_eligible"]
    ]

    print(
        review[
            [
                "museum",
                "source_label",
                "canonical_components",
                "record_count",
                "proposal_decision",
            ]
        ].to_string(index=False)
    )

    print(f"\nPropuestas: {PROPOSALS_OUTPUT}")
    print(f"Resumen: {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()