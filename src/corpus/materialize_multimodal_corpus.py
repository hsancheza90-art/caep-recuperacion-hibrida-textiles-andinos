"""Materialización reproducible del corpus multimodal derivado.

Este módulo lee las dos fuentes congeladas, construye el corpus
multimodal y genera artefactos portables de datos, resumen y
procedencia.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.corpus.build_multimodal_corpus import (
    build_multimodal_corpus,
)


DATASET_NAME = "paper_corpus_multimodal_v1"
PIPELINE_VERSION = "multimodal_corpus_materialization_v1"


def _resolve_path(
    path: Path,
    repository_root: Path,
) -> Path:
    """Devuelve una ruta absoluta resuelta respecto del repositorio."""

    if path.is_absolute():
        return path.resolve()

    return (repository_root / path).resolve()


def _portable_relative_path(
    path: Path,
    repository_root: Path,
) -> str:
    """Convierte una ruta del repositorio a formato POSIX relativo."""

    resolved_path = path.resolve()
    resolved_root = repository_root.resolve()

    try:
        relative_path = resolved_path.relative_to(
            resolved_root
        )
    except ValueError as exc:
        raise ValueError(
            "El artefacto debe ubicarse dentro de la raíz "
            f"del repositorio: {resolved_path}"
        ) from exc

    return relative_path.as_posix()


def _validate_materialization_paths(
    *,
    cultural_corpus_path: Path,
    image_manifest_path: Path,
    output_csv_path: Path,
    output_parquet_path: Path,
    summary_path: Path,
    provenance_path: Path,
) -> None:
    """Impide colisiones entre fuentes y artefactos de salida."""

    source_paths = {
        cultural_corpus_path.resolve(),
        image_manifest_path.resolve(),
    }

    output_paths = [
        output_csv_path.resolve(),
        output_parquet_path.resolve(),
        summary_path.resolve(),
        provenance_path.resolve(),
    ]

    if len(set(output_paths)) != len(output_paths):
        raise ValueError(
            "Las rutas de salida deben ser distintas."
        )

    overwritten_sources = source_paths.intersection(
        output_paths
    )

    if overwritten_sources:
        rendered_paths = ", ".join(
            sorted(
                path.as_posix()
                for path in overwritten_sources
            )
        )

        raise ValueError(
            "No se permite sobrescribir un archivo fuente. "
            f"Rutas en conflicto: {rendered_paths}"
        )


def _require_source_file(
    path: Path,
    source_name: str,
) -> None:
    """Comprueba que un archivo fuente exista."""

    if not path.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo fuente "
            f"{source_name}: {path}"
        )


def _sha256_file(
    path: Path,
) -> str:
    """Calcula el hash SHA-256 de un archivo."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _temporary_path(
    destination: Path,
) -> Path:
    """Genera una ruta temporal junto al archivo de destino."""

    return destination.with_name(
        f".{destination.name}.{uuid4().hex}.tmp"
    )


def _atomic_write_csv(
    frame: pd.DataFrame,
    destination: Path,
) -> None:
    """Escribe un CSV mediante reemplazo atómico."""

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = _temporary_path(destination)

    try:
        frame.to_csv(
            temporary,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )

        os.replace(
            temporary,
            destination,
        )
    finally:
        temporary.unlink(
            missing_ok=True
        )


def _atomic_write_parquet(
    frame: pd.DataFrame,
    destination: Path,
) -> None:
    """Escribe un archivo Parquet mediante reemplazo atómico."""

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = _temporary_path(destination)

    try:
        frame.to_parquet(
            temporary,
            index=False,
        )

        os.replace(
            temporary,
            destination,
        )
    finally:
        temporary.unlink(
            missing_ok=True
        )


def _atomic_write_json(
    payload: dict[str, object],
    destination: Path,
) -> None:
    """Escribe JSON UTF-8 mediante reemplazo atómico."""

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = _temporary_path(destination)

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )

    try:
        temporary.write_text(
            serialized + "\n",
            encoding="utf-8",
            newline="\n",
        )

        os.replace(
            temporary,
            destination,
        )
    finally:
        temporary.unlink(
            missing_ok=True
        )


def _summarize_partition(
    frame: pd.DataFrame,
    museum: str,
) -> dict[str, object]:
    """Resume la cobertura técnica de una partición."""

    image_paths = (
        frame["image_local_path"]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    image_hashes = (
        frame["image_sha256"]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    image_bytes = pd.to_numeric(
        frame["image_bytes"],
        errors="coerce",
    ).fillna(0)

    total_records = len(frame)
    records_with_image = int(
        image_paths.ne("").sum()
    )

    coverage_rate = (
        records_with_image / total_records
        if total_records
        else 0.0
    )

    unique_image_hashes = int(
        image_hashes[
            image_hashes.ne("")
        ].nunique()
    )

    return {
        "museum": museum,
        "total_records": total_records,
        "records_with_image": records_with_image,
        "records_without_image": (
            total_records - records_with_image
        ),
        "image_coverage_rate": coverage_rate,
        "unique_image_hashes": unique_image_hashes,
        "total_image_bytes": int(
            image_bytes.sum()
        ),
    }


def build_multimodal_summary(
    corpus: pd.DataFrame,
) -> pd.DataFrame:
    """Construye un resumen total y por museo."""

    rows = [
        _summarize_partition(
            frame=corpus,
            museum="ALL",
        )
    ]

    museums = sorted(
        corpus["museum"]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.upper()
        .unique()
        .tolist()
    )

    normalized_museum = (
        corpus["museum"]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.upper()
    )

    for museum in museums:
        partition = corpus.loc[
            normalized_museum.eq(museum)
        ]

        rows.append(
            _summarize_partition(
                frame=partition,
                museum=museum,
            )
        )

    return pd.DataFrame(
        rows,
        columns=[
            "museum",
            "total_records",
            "records_with_image",
            "records_without_image",
            "image_coverage_rate",
            "unique_image_hashes",
            "total_image_bytes",
        ],
    )


def _artifact_metadata(
    path: Path,
    repository_root: Path,
) -> dict[str, object]:
    """Describe un artefacto mediante ruta, tamaño y hash."""

    return {
        "path": _portable_relative_path(
            path,
            repository_root,
        ),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _build_provenance(
    *,
    cultural_corpus_path: Path,
    image_manifest_path: Path,
    output_csv_path: Path,
    output_parquet_path: Path,
    summary_path: Path,
    repository_root: Path,
    cultural_records: int,
    manifest_records: int,
    matched_records: int,
) -> dict[str, object]:
    """Construye la declaración portable de procedencia."""

    coverage_rate = (
        matched_records / cultural_records
        if cultural_records
        else 0.0
    )

    return {
        "dataset_name": DATASET_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "join": {
            "key": "item_id",
            "type": "left",
            "cardinality": "one_to_one",
            "coverage_required": True,
        },
        "coverage": {
            "cultural_records": cultural_records,
            "manifest_records": manifest_records,
            "matched_records": matched_records,
            "coverage_rate": coverage_rate,
        },
        "inputs": {
            "cultural_corpus": _artifact_metadata(
                cultural_corpus_path,
                repository_root,
            ),
            "image_manifest": _artifact_metadata(
                image_manifest_path,
                repository_root,
            ),
        },
        "outputs": {
            "csv": _artifact_metadata(
                output_csv_path,
                repository_root,
            ),
            "parquet": _artifact_metadata(
                output_parquet_path,
                repository_root,
            ),
            "summary": _artifact_metadata(
                summary_path,
                repository_root,
            ),
        },
    }


def materialize_multimodal_corpus(
    *,
    cultural_corpus_path: Path,
    image_manifest_path: Path,
    output_csv_path: Path,
    output_parquet_path: Path,
    summary_path: Path,
    provenance_path: Path,
    repository_root: Path,
) -> None:
    """Materializa el corpus y sus artefactos de trazabilidad.

    Las fuentes se leen sin modificarlas. Los resultados se escriben
    en archivos independientes mediante reemplazo atómico.
    """

    repository_root = repository_root.resolve()

    cultural_corpus_path = _resolve_path(
        cultural_corpus_path,
        repository_root,
    )

    image_manifest_path = _resolve_path(
        image_manifest_path,
        repository_root,
    )

    output_csv_path = _resolve_path(
        output_csv_path,
        repository_root,
    )

    output_parquet_path = _resolve_path(
        output_parquet_path,
        repository_root,
    )

    summary_path = _resolve_path(
        summary_path,
        repository_root,
    )

    provenance_path = _resolve_path(
        provenance_path,
        repository_root,
    )

    _validate_materialization_paths(
        cultural_corpus_path=cultural_corpus_path,
        image_manifest_path=image_manifest_path,
        output_csv_path=output_csv_path,
        output_parquet_path=output_parquet_path,
        summary_path=summary_path,
        provenance_path=provenance_path,
    )

    for path in (
        cultural_corpus_path,
        image_manifest_path,
        output_csv_path,
        output_parquet_path,
        summary_path,
        provenance_path,
    ):
        _portable_relative_path(
            path,
            repository_root,
        )

    _require_source_file(
        cultural_corpus_path,
        "corpus cultural",
    )

    _require_source_file(
        image_manifest_path,
        "manifiesto de imágenes",
    )

    cultural_corpus = pd.read_csv(
        cultural_corpus_path,
        dtype=str,
        keep_default_na=False,
    )

    image_manifest = pd.read_csv(
        image_manifest_path,
        dtype=str,
        keep_default_na=False,
    )

    multimodal_corpus = build_multimodal_corpus(
        cultural_corpus=cultural_corpus,
        image_manifest=image_manifest,
    )

    summary = build_multimodal_summary(
        multimodal_corpus
    )

    _atomic_write_csv(
        multimodal_corpus,
        output_csv_path,
    )

    _atomic_write_parquet(
        multimodal_corpus,
        output_parquet_path,
    )

    _atomic_write_csv(
        summary,
        summary_path,
    )

    provenance = _build_provenance(
        cultural_corpus_path=cultural_corpus_path,
        image_manifest_path=image_manifest_path,
        output_csv_path=output_csv_path,
        output_parquet_path=output_parquet_path,
        summary_path=summary_path,
        repository_root=repository_root,
        cultural_records=len(cultural_corpus),
        manifest_records=len(image_manifest),
        matched_records=len(multimodal_corpus),
    )

    _atomic_write_json(
        provenance,
        provenance_path,
    )