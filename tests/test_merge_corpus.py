import pandas as pd

from caep.datasets.merge_corpus import (
    build_corpus,
    validate_combined_corpus,
)


def test_combined_corpus_has_expected_size() -> None:
    corpus = build_corpus()
    assert len(corpus) == 215


def test_combined_corpus_has_expected_museum_counts() -> None:
    corpus = build_corpus()
    counts = corpus["museum"].value_counts().to_dict()

    assert counts == {
        "MET": 127,
        "CMA": 88,
    }


def test_item_ids_are_unique() -> None:
    corpus = build_corpus()
    assert corpus["item_id"].is_unique


def test_source_keys_are_unique() -> None:
    corpus = build_corpus()

    assert not corpus.duplicated(
        subset=["museum", "source_object_id"]
    ).any()


def test_required_urls_are_complete() -> None:
    corpus = build_corpus()

    assert corpus["image_url"].str.strip().ne("").all()
    assert corpus["object_url"].str.strip().ne("").all()


def test_only_principal_approved_records_are_present() -> None:
    corpus = build_corpus()

    assert set(corpus["dataset_split"]) == {"principal"}
    assert set(corpus["review_status"]) == {"aprobado"}


def test_validation_rejects_duplicate_item_id() -> None:
    corpus = build_corpus()
    duplicated = pd.concat(
        [corpus, corpus.iloc[[0]]],
        ignore_index=True,
    )

    try:
        validate_combined_corpus(duplicated)
    except ValueError:
        return

    raise AssertionError(
        "La validación no rechazó un item_id duplicado."
    )