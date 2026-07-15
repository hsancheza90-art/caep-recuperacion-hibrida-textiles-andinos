"""Pruebas de materialización del corpus multimodal."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.corpus.materialize_multimodal_corpus import (
    DATASET_NAME,
    PIPELINE_VERSION,
    materialize_multimodal_corpus,
)


def make_cultural_corpus() -> pd.DataFrame:
    """Crea un corpus cultural mínimo."""

    return pd.DataFrame(
        [
            {
                "item_id": "MET:2",
                "museum": "MET",
                "title": "Textile fragment",
                "culture_normalized": "Wari",
            },
            {
                "item_id": "CMA:1",
                "museum": "CMA",
                "title": "Mantle",
                "culture_normalized": "Paracas",
            },
        ]
    )


def make_image_manifest() -> pd.DataFrame:
    """Crea un manifiesto técnico mínimo."""

    return pd.DataFrame(
        [
            {
                "item_id": "CMA:1",
                "museum": "CMA",
                "image_local_path": (
                    "data/images/cma/CMA_1.jpg"
                ),
                "download_status": "available",
                "acquisition_action": "reused",
                "final_url": "https://example.org/cma.jpg",
                "content_type": "image/jpeg",
                "image_bytes": "200",
                "image_width": "20",
                "image_height": "10",
                "image_format": "JPEG",
                "sha256": "b" * 64,
                "download_version": "image_download_v1",
            },
            {
                "item_id": "MET:2",
                "museum": "MET",
                "image_local_path": (
                    "data/images/met/MET_2.jpg"
                ),
                "download_status": "available",
                "acquisition_action": "downloaded",
                "final_url": "https://example.org/met.jpg",
                "content_type": "image/jpeg",
                "image_bytes": "100",
                "image_width": "10",
                "image_height": "20",
                "image_format": "JPEG",
                "sha256": "a" * 64,
                "download_version": "image_download_v1",
            },
        ]
    )


def create_test_paths(
    repository_root: Path,
) -> dict[str, Path]:
    """Define rutas reproducibles dentro de un repositorio temporal."""

    return {
        "cultural": (
            repository_root
            / "data"
            / "processed"
            / "paper_corpus_culture_enriched_v1.csv"
        ),
        "manifest": (
            repository_root
            / "outputs"
            / "reports"
            / "image_download_manifest_v1.csv"
        ),
        "output_csv": (
            repository_root
            / "data"
            / "derived"
            / "paper_corpus_multimodal_v1.csv"
        ),
        "output_parquet": (
            repository_root
            / "data"
            / "derived"
            / "paper_corpus_multimodal_v1.parquet"
        ),
        "summary": (
            repository_root
            / "outputs"
            / "reports"
            / "multimodal_corpus_summary_v1.csv"
        ),
        "provenance": (
            repository_root
            / "outputs"
            / "reports"
            / "multimodal_corpus_provenance_v1.json"
        ),
    }


def write_test_sources(
    repository_root: Path,
) -> dict[str, Path]:
    """Escribe las dos fuentes de prueba."""

    paths = create_test_paths(repository_root)

    paths["cultural"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    paths["manifest"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    make_cultural_corpus().to_csv(
        paths["cultural"],
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    make_image_manifest().to_csv(
        paths["manifest"],
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    return paths


def run_materialization(
    repository_root: Path,
    paths: dict[str, Path],
) -> None:
    """Ejecuta la materialización con las rutas de prueba."""

    materialize_multimodal_corpus(
        cultural_corpus_path=paths["cultural"],
        image_manifest_path=paths["manifest"],
        output_csv_path=paths["output_csv"],
        output_parquet_path=paths["output_parquet"],
        summary_path=paths["summary"],
        provenance_path=paths["provenance"],
        repository_root=repository_root,
    )


def test_materialization_writes_all_artifacts(
    tmp_path: Path,
) -> None:
    paths = write_test_sources(tmp_path)

    run_materialization(
        repository_root=tmp_path,
        paths=paths,
    )

    assert paths["output_csv"].is_file()
    assert paths["output_parquet"].is_file()
    assert paths["summary"].is_file()
    assert paths["provenance"].is_file()


def test_materialization_preserves_source_files(
    tmp_path: Path,
) -> None:
    paths = write_test_sources(tmp_path)

    cultural_before = paths["cultural"].read_bytes()
    manifest_before = paths["manifest"].read_bytes()

    run_materialization(
        repository_root=tmp_path,
        paths=paths,
    )

    assert paths["cultural"].read_bytes() == cultural_before
    assert paths["manifest"].read_bytes() == manifest_before


def test_materialized_csv_and_parquet_are_equivalent(
    tmp_path: Path,
) -> None:
    paths = write_test_sources(tmp_path)

    run_materialization(
        repository_root=tmp_path,
        paths=paths,
    )

    csv_frame = pd.read_csv(
        paths["output_csv"],
        dtype=str,
        keep_default_na=False,
    )

    parquet_frame = (
        pd.read_parquet(
            paths["output_parquet"]
        )
        .fillna("")
        .astype(str)
    )

    pd.testing.assert_frame_equal(
        csv_frame,
        parquet_frame,
        check_dtype=False,
    )

    assert csv_frame["item_id"].tolist() == [
        "MET:2",
        "CMA:1",
    ]
    assert csv_frame["image_local_path"].tolist() == [
        "data/images/met/MET_2.jpg",
        "data/images/cma/CMA_1.jpg",
    ]


def test_materialization_builds_summary_by_museum(
    tmp_path: Path,
) -> None:
    paths = write_test_sources(tmp_path)

    run_materialization(
        repository_root=tmp_path,
        paths=paths,
    )

    summary = pd.read_csv(
        paths["summary"],
        keep_default_na=False,
    )

    assert summary["museum"].tolist() == [
        "ALL",
        "CMA",
        "MET",
    ]

    all_row = summary.loc[
        summary["museum"].eq("ALL")
    ].iloc[0]

    assert int(all_row["total_records"]) == 2
    assert int(all_row["records_with_image"]) == 2
    assert float(all_row["image_coverage_rate"]) == 1.0
    assert int(all_row["unique_image_hashes"]) == 2
    assert int(all_row["total_image_bytes"]) == 300


def test_materialization_writes_portable_provenance(
    tmp_path: Path,
) -> None:
    paths = write_test_sources(tmp_path)

    run_materialization(
        repository_root=tmp_path,
        paths=paths,
    )

    provenance = json.loads(
        paths["provenance"].read_text(
            encoding="utf-8"
        )
    )

    assert provenance["dataset_name"] == DATASET_NAME
    assert (
        provenance["pipeline_version"]
        == PIPELINE_VERSION
    )

    assert provenance["join"] == {
        "key": "item_id",
        "type": "left",
        "cardinality": "one_to_one",
        "coverage_required": True,
    }

    assert provenance["coverage"] == {
        "cultural_records": 2,
        "manifest_records": 2,
        "matched_records": 2,
        "coverage_rate": 1.0,
    }

    assert (
        provenance["inputs"]["cultural_corpus"]["path"]
        == (
            "data/processed/"
            "paper_corpus_culture_enriched_v1.csv"
        )
    )

    assert (
        provenance["outputs"]["csv"]["path"]
        == (
            "data/derived/"
            "paper_corpus_multimodal_v1.csv"
        )
    )

    for section in ("inputs", "outputs"):
        for artifact in provenance[section].values():
            assert not Path(artifact["path"]).is_absolute()
            assert "\\" not in artifact["path"]
            assert len(artifact["sha256"]) == 64


def test_materialization_rejects_overwriting_source(
    tmp_path: Path,
) -> None:
    paths = write_test_sources(tmp_path)

    with pytest.raises(
        ValueError,
        match="sobrescribir.*fuente",
    ):
        materialize_multimodal_corpus(
            cultural_corpus_path=paths["cultural"],
            image_manifest_path=paths["manifest"],
            output_csv_path=paths["cultural"],
            output_parquet_path=paths["output_parquet"],
            summary_path=paths["summary"],
            provenance_path=paths["provenance"],
            repository_root=tmp_path,
        )


def test_materialization_requires_distinct_output_paths(
    tmp_path: Path,
) -> None:
    paths = write_test_sources(tmp_path)

    with pytest.raises(
        ValueError,
        match="rutas de salida.*distintas",
    ):
        materialize_multimodal_corpus(
            cultural_corpus_path=paths["cultural"],
            image_manifest_path=paths["manifest"],
            output_csv_path=paths["output_csv"],
            output_parquet_path=paths["output_csv"],
            summary_path=paths["summary"],
            provenance_path=paths["provenance"],
            repository_root=tmp_path,
        )