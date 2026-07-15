"""Pruebas de materialización de entradas textuales OpenCLIP."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.openclip_baseline.materialize_text_inputs import (
    DATASET_NAME,
    PIPELINE_VERSION,
    materialize_openclip_text_inputs,
)


def make_corpus() -> pd.DataFrame:
    """Construye un corpus mínimo para las pruebas."""

    return pd.DataFrame(
        [
            {
                "item_id": "MET:2",
                "museum": "MET",
                "title": "Textile fragment",
                "object_type": "Textile fragment",
                "material": "Cotton, camelid fiber",
                "technique": "Tapestry weave",
                "description": "Fragment with geometric motifs.",
                "culture": "Wari",
                "culture_canonical": "Wari",
                "period": "Middle Horizon",
                "classification": "Textiles",
                "country": "Peru",
                "region": "Central Andes",
            },
            {
                "item_id": "CMA:1",
                "museum": "CMA",
                "title": "Mantle",
                "object_type": "Mantle",
                "material": "Cotton",
                "technique": "",
                "description": "",
                "culture": "Paracas",
                "culture_canonical": "Paracas",
                "period": "Early Intermediate Period",
                "classification": "Textiles",
                "country": "Peru",
                "region": "",
            },
        ]
    )


def make_paths(root: Path) -> dict[str, Path]:
    """Define rutas dentro de un repositorio temporal."""

    return {
        "corpus": (
            root
            / "data"
            / "derived"
            / "paper_corpus_multimodal_v1.csv"
        ),
        "output_csv": (
            root
            / "data"
            / "derived"
            / "openclip_text_inputs_v1.csv"
        ),
        "output_parquet": (
            root
            / "data"
            / "derived"
            / "openclip_text_inputs_v1.parquet"
        ),
        "summary": (
            root
            / "outputs"
            / "reports"
            / "openclip_text_inputs_summary_v1.csv"
        ),
        "provenance": (
            root
            / "outputs"
            / "reports"
            / "openclip_text_inputs_provenance_v1.json"
        ),
    }


def write_source(root: Path) -> dict[str, Path]:
    """Escribe el corpus de prueba."""

    paths = make_paths(root)

    paths["corpus"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    make_corpus().to_csv(
        paths["corpus"],
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    return paths


def run_materialization(
    root: Path,
    paths: dict[str, Path],
) -> None:
    """Ejecuta la materialización de prueba."""

    materialize_openclip_text_inputs(
        corpus_path=paths["corpus"],
        output_csv_path=paths["output_csv"],
        output_parquet_path=paths["output_parquet"],
        summary_path=paths["summary"],
        provenance_path=paths["provenance"],
        repository_root=root,
    )


def test_materialization_writes_all_artifacts(
    tmp_path: Path,
) -> None:
    paths = write_source(tmp_path)

    run_materialization(
        tmp_path,
        paths,
    )

    assert paths["output_csv"].is_file()
    assert paths["output_parquet"].is_file()
    assert paths["summary"].is_file()
    assert paths["provenance"].is_file()


def test_materialization_preserves_source(
    tmp_path: Path,
) -> None:
    paths = write_source(tmp_path)
    before = paths["corpus"].read_bytes()

    run_materialization(
        tmp_path,
        paths,
    )

    assert paths["corpus"].read_bytes() == before


def test_materialized_csv_and_parquet_are_equivalent(
    tmp_path: Path,
) -> None:
    paths = write_source(tmp_path)

    run_materialization(
        tmp_path,
        paths,
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

    assert csv_frame.shape == (2, 8)
    assert csv_frame["item_id"].tolist() == [
        "MET:2",
        "CMA:1",
    ]


def test_materialization_builds_summary(
    tmp_path: Path,
) -> None:
    paths = write_source(tmp_path)

    run_materialization(
        tmp_path,
        paths,
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
    assert int(all_row["visual_text_nonempty"]) == 2
    assert int(all_row["metadata_text_nonempty"]) == 2
    assert int(all_row["unique_visual_hashes"]) == 2
    assert int(all_row["unique_metadata_hashes"]) == 2


def test_materialization_writes_portable_provenance(
    tmp_path: Path,
) -> None:
    paths = write_source(tmp_path)

    run_materialization(
        tmp_path,
        paths,
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

    assert provenance["coverage"] == {
        "source_records": 2,
        "output_records": 2,
        "visual_text_nonempty": 2,
        "metadata_text_nonempty": 2,
    }

    assert (
        provenance["input"]["path"]
        == (
            "data/derived/"
            "paper_corpus_multimodal_v1.csv"
        )
    )

    assert (
        provenance["outputs"]["csv"]["path"]
        == (
            "data/derived/"
            "openclip_text_inputs_v1.csv"
        )
    )

    for metadata in [
        provenance["input"],
        *provenance["outputs"].values(),
    ]:
        assert not Path(
            metadata["path"]
        ).is_absolute()

        assert "\\" not in metadata["path"]
        assert len(metadata["sha256"]) == 64


def test_materialization_rejects_source_overwrite(
    tmp_path: Path,
) -> None:
    paths = write_source(tmp_path)

    with pytest.raises(
        ValueError,
        match="sobrescribir.*fuente",
    ):
        materialize_openclip_text_inputs(
            corpus_path=paths["corpus"],
            output_csv_path=paths["corpus"],
            output_parquet_path=paths["output_parquet"],
            summary_path=paths["summary"],
            provenance_path=paths["provenance"],
            repository_root=tmp_path,
        )


def test_materialization_requires_distinct_outputs(
    tmp_path: Path,
) -> None:
    paths = write_source(tmp_path)

    with pytest.raises(
        ValueError,
        match="rutas de salida.*distintas",
    ):
        materialize_openclip_text_inputs(
            corpus_path=paths["corpus"],
            output_csv_path=paths["output_csv"],
            output_parquet_path=paths["output_csv"],
            summary_path=paths["summary"],
            provenance_path=paths["provenance"],
            repository_root=tmp_path,
        )