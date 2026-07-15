import pandas as pd

from caep.config import get_processing_timestamp
from caep.datasets.build_enriched_corpus import (
    build_enriched_corpus,
)


def test_build_is_deterministic() -> None:
    first = build_enriched_corpus()
    second = build_enriched_corpus()

    pd.testing.assert_frame_equal(
        first,
        second,
        check_dtype=True,
        check_like=False,
    )


def test_processing_timestamp_is_fixed() -> None:
    corpus = build_enriched_corpus()

    assert corpus["processing_timestamp"].nunique() == 1
    assert (
        corpus["processing_timestamp"].iloc[0]
        == get_processing_timestamp()
    )


def test_build_uses_documented_source_files() -> None:
    corpus = build_enriched_corpus()

    expected_source_files = {
        "data/metadata/met_corpus_principal_v2_revisado.csv",
        "data/metadata/cma_corpus_principal_revisado.csv",
    }

    expected_enrichment_files = {
        (
            "data/processed/"
            "corpus_met_textiles_andinos_v1_inventario_base.csv"
        ),
        "data/metadata/cma_andes_textiles_candidates.csv",
    }

    assert set(corpus["source_file"]) == expected_source_files
    assert (
        set(corpus["enrichment_source_file"])
        == expected_enrichment_files
    )