from caep.datasets.build_enriched_corpus import (
    build_enriched_corpus,
)
from caep.evaluation.category_audit import (
    audit_categories,
    classify_category,
    load_audit_config,
    normalize_label,
)


def test_normalize_label_is_lexical_and_deterministic() -> None:
    assert (
        normalize_label("  Textil   Andino ")
        == "textil andino"
    )


def test_balanced_category_is_global_candidate() -> None:
    status = classify_category(
        field_role="core",
        total_count=10,
        museum_counts={"MET": 5, "CMA": 5},
        thresholds={
            "min_total_support": 5,
            "min_support_per_museum": 2,
            "min_museums": 2,
            "max_museum_dominance": 0.85,
        },
    )

    assert status == "candidate_global"


def test_single_museum_category_is_source_specific() -> None:
    status = classify_category(
        field_role="core",
        total_count=10,
        museum_counts={"MET": 10, "CMA": 0},
        thresholds={
            "min_total_support": 5,
            "min_support_per_museum": 2,
            "min_museums": 2,
            "max_museum_dominance": 0.85,
        },
    )

    assert status == "source_specific"


def test_core_fields_have_complete_coverage() -> None:
    corpus = build_enriched_corpus()
    config = load_audit_config()

    summary, _ = audit_categories(
        corpus,
        config,
    )

    core = summary.loc[
        summary["field_role"].eq("core")
    ]

    assert core["non_empty"].eq(215).all()
    assert core["coverage"].eq(1.0).all()


def test_detail_counts_match_non_empty_records() -> None:
    corpus = build_enriched_corpus()
    config = load_audit_config()

    summary, detail = audit_categories(
        corpus,
        config,
    )

    detail_totals = (
        detail
        .groupby("field")["total_count"]
        .sum()
        .to_dict()
    )

    for row in summary.itertuples(index=False):
        assert detail_totals[row.field] == row.non_empty