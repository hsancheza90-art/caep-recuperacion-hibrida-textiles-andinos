from caep.datasets.build_enriched_corpus import (
    build_enriched_corpus,
)


def test_enriched_corpus_has_expected_size() -> None:
    corpus = build_enriched_corpus()
    assert len(corpus) == 215


def test_enriched_corpus_has_expected_distribution() -> None:
    corpus = build_enriched_corpus()

    assert corpus["museum"].value_counts().to_dict() == {
        "MET": 127,
        "CMA": 88,
    }


def test_enriched_corpus_item_ids_are_unique() -> None:
    corpus = build_enriched_corpus()
    assert corpus["item_id"].is_unique


def test_core_metadata_is_complete() -> None:
    corpus = build_enriched_corpus()

    for column in (
        "title",
        "culture",
        "period",
        "object_type",
        "classification",
        "image_url",
        "object_url",
        "license",
    ):
        assert (
            corpus[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .all()
        ), column


def test_metadata_recovery_is_documented() -> None:
    corpus = build_enriched_corpus()

    assert corpus["metadata_recovery_source"].str.strip().ne("").all()
    assert corpus["metadata_recovery_version"].str.strip().ne("").all()
    assert corpus["enrichment_source_file"].str.strip().ne("").all()