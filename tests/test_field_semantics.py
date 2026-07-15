from caep.datasets.build_enriched_corpus import (
    build_enriched_corpus,
)
from caep.evaluation.field_semantics import (
    audit_field_pair,
)


def test_field_semantics_summary_covers_all_groups() -> None:
    corpus = build_enriched_corpus()

    summary, _ = audit_field_pair(
        corpus,
        left_field="classification",
        right_field="object_type",
    )

    assert set(summary["museum"]) == {
        "ALL",
        "MET",
        "CMA",
    }


def test_field_semantics_row_counts_are_correct() -> None:
    corpus = build_enriched_corpus()

    summary, _ = audit_field_pair(
        corpus,
        left_field="classification",
        right_field="object_type",
    )

    counts = dict(
        zip(
            summary["museum"],
            summary["total_rows"],
            strict=True,
        )
    )

    assert counts == {
        "ALL": 215,
        "MET": 127,
        "CMA": 88,
    }


def test_pair_counts_cover_all_comparable_rows() -> None:
    corpus = build_enriched_corpus()

    summary, pairs = audit_field_pair(
        corpus,
        left_field="classification",
        right_field="object_type",
    )

    for museum in ["ALL", "MET", "CMA"]:
        comparable = int(
            summary.loc[
                summary["museum"].eq(museum),
                "comparable_rows",
            ].iloc[0]
        )

        pair_total = int(
            pairs.loc[
                pairs["museum"].eq(museum),
                "count",
            ].sum()
        )

        assert pair_total == comparable