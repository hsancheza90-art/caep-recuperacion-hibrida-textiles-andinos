from __future__ import annotations

from pathlib import Path

from caep.datasets.build_enriched_corpus import (
    build_enriched_corpus,
)
from caep.evaluation.field_semantics import (
    audit_field_pair,
)


SUMMARY_OUTPUT = Path(
    "outputs/reports/field_semantics_summary_v1.csv"
)

PAIRS_OUTPUT = Path(
    "outputs/reports/field_semantics_pairs_v1.csv"
)


def main() -> None:
    corpus = build_enriched_corpus()

    summary, pairs = audit_field_pair(
        corpus=corpus,
        left_field="classification",
        right_field="object_type",
    )

    SUMMARY_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    pairs.to_csv(
        PAIRS_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nAUDITORÍA SEMÁNTICA")
    print("=" * 110)
    print(summary.to_string(index=False))

    print("\nPARES PRINCIPALES POR MUSEO")
    print("=" * 110)

    for museum in ["MET", "CMA"]:
        subset = pairs.loc[
            pairs["museum"].eq(museum)
        ].head(30)

        print(f"\n{museum}")
        print("-" * 110)

        print(
            subset[
                [
                    "left_label",
                    "right_label",
                    "count",
                    "same_normalized_label",
                ]
            ].to_string(index=False)
        )

    print(f"\nResumen: {SUMMARY_OUTPUT}")
    print(f"Pares: {PAIRS_OUTPUT}")


if __name__ == "__main__":
    main()