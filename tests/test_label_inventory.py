from caep.datasets.build_enriched_corpus import (
    build_enriched_corpus,
)
from caep.evaluation.label_inventory import (
    build_cross_museum_candidates,
    build_label_inventory,
    build_mapping_template,
    lexical_pair_metrics,
)


FIELDS = [
    "culture",
    "period",
]


def test_inventory_counts_cover_all_records() -> None:
    corpus = build_enriched_corpus()

    inventory = build_label_inventory(
        corpus,
        FIELDS,
    )

    totals = (
        inventory
        .groupby(["field", "museum"])["record_count"]
        .sum()
        .to_dict()
    )

    assert totals == {
        ("culture", "MET"): 127,
        ("culture", "CMA"): 88,
        ("period", "MET"): 127,
        ("period", "CMA"): 88,
    }


def test_inventory_has_expected_category_counts() -> None:
    corpus = build_enriched_corpus()

    inventory = build_label_inventory(
        corpus,
        FIELDS,
    )

    counts = (
        inventory
        .groupby(["field", "museum"])["source_key"]
        .nunique()
        .to_dict()
    )

    assert counts == {
        ("culture", "MET"): 14,
        ("culture", "CMA"): 43,
        ("period", "MET"): 39,
        ("period", "CMA"): 41,
    }


def test_mapping_rows_are_unique() -> None:
    corpus = build_enriched_corpus()

    inventory = build_label_inventory(
        corpus,
        FIELDS,
    )

    mapping = build_mapping_template(inventory)

    assert not mapping.duplicated(
        subset=[
            "field",
            "museum",
            "source_key",
        ]
    ).any()


def test_lexical_candidates_are_cross_museum() -> None:
    corpus = build_enriched_corpus()

    inventory = build_label_inventory(
        corpus,
        FIELDS,
    )

    candidates = build_cross_museum_candidates(
        inventory,
    )

    assert candidates["met_source_key"].ne("").all()
    assert candidates["cma_source_key"].ne("").all()


def test_identical_tokens_have_full_similarity() -> None:
    metrics = lexical_pair_metrics(
        "Chancay culture",
        "Chancay culture",
    )

    assert metrics["token_jaccard"] == 1.0
    assert metrics["substring_match"] is True