"""Materialización reproducible de entradas textuales OpenCLIP."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.openclip_baseline.text_inputs import (
    build_openclip_text_inputs,
)


DATASET_NAME = "openclip_text_inputs_v1"
PIPELINE_VERSION = "openclip_text_materialization_v1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "paper_corpus_multimodal_v1.csv"
)

DEFAULT_OUTPUT_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "openclip_text_inputs_v1.csv"
)

DEFAULT_OUTPUT_PARQUET_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "openclip_text_inputs_v1.parquet"
)

DEFAULT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "openclip_text_inputs_summary_v1.csv"
)

DEFAULT_PROVENANCE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "openclip_text_inputs_provenance_v1.json"
)

def _resolve_path(
    path: Path,
    repository_root: Path,
) -> Path:
    """Resuelve una ruta respecto de la raíz del repositorio."""

    if path.is_absolute():
        return path.resolve()

    return (repository_root / path).resolve()


def _portable_relative_path(
    path: Path,
    repository_root: Path,
) -> str:
    """Devuelve una ruta POSIX relativa al repositorio."""

    resolved_path = path.resolve()
    resolved_root = repository_root.resolve()

    try:
        relative_path = resolved_path.relative_to(
            resolved_root
        )
    except ValueError as exc:
        raise ValueError(
            "El artefacto debe estar dentro de la raíz "
            f"del repositorio: {resolved_path}"
        ) from exc

    return relative_path.as_posix()


def _validate_paths(
    *,
    corpus_path: Path,
    output_csv_path: Path,
    output_parquet_path: Path,
    summary_path: Path,
    provenance_path: Path,
) -> None:
    """Impide sobrescrituras y colisiones entre salidas."""

    source_path = corpus_path.resolve()

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

    if source_path in output_paths:
        raise ValueError(
            "No se permite sobrescribir el archivo fuente."
        )


def _require_source_file(
    path: Path,
) -> None:
    """Comprueba que el corpus fuente exista."""

    if not path.is_file():
        raise FileNotFoundError(
            f"No se encontró el corpus fuente: {path}"
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
    """Genera una ruta temporal junto al destino."""

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
    """Escribe Parquet mediante reemplazo atómico."""

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
    """Resume la cobertura textual de una partición."""

    visual_text = (
        frame["text_visual"]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    metadata_text = (
        frame["text_metadata"]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    visual_hashes = (
        frame["text_visual_sha256"]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    metadata_hashes = (
        frame["text_metadata_sha256"]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    return {
        "museum": museum,
        "total_records": len(frame),
        "visual_text_nonempty": int(
            visual_text.ne("").sum()
        ),
        "metadata_text_nonempty": int(
            metadata_text.ne("").sum()
        ),
        "unique_visual_hashes": int(
            visual_hashes[
                visual_hashes.ne("")
            ].nunique()
        ),
        "unique_metadata_hashes": int(
            metadata_hashes[
                metadata_hashes.ne("")
            ].nunique()
        ),
    }


def build_text_inputs_summary(
    text_inputs: pd.DataFrame,
) -> pd.DataFrame:
    """Construye el resumen total y por museo."""

    rows = [
        _summarize_partition(
            frame=text_inputs,
            museum="ALL",
        )
    ]

    normalized_museum = (
        text_inputs["museum"]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.upper()
    )

    museums = sorted(
        normalized_museum.unique().tolist()
    )

    for museum in museums:
        partition = text_inputs.loc[
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
            "visual_text_nonempty",
            "metadata_text_nonempty",
            "unique_visual_hashes",
            "unique_metadata_hashes",
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
    corpus_path: Path,
    output_csv_path: Path,
    output_parquet_path: Path,
    summary_path: Path,
    repository_root: Path,
    source_records: int,
    text_inputs: pd.DataFrame,
) -> dict[str, object]:
    """Construye la declaración de procedencia."""

    visual_nonempty = int(
        text_inputs["text_visual"]
        .astype("string")
        .fillna("")
        .str.strip()
        .ne("")
        .sum()
    )

    metadata_nonempty = int(
        text_inputs["text_metadata"]
        .astype("string")
        .fillna("")
        .str.strip()
        .ne("")
        .sum()
    )

    return {
        "dataset_name": DATASET_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "coverage": {
            "source_records": source_records,
            "output_records": len(text_inputs),
            "visual_text_nonempty": visual_nonempty,
            "metadata_text_nonempty": metadata_nonempty,
        },
        "input": _artifact_metadata(
            corpus_path,
            repository_root,
        ),
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


def materialize_openclip_text_inputs(
    *,
    corpus_path: Path,
    output_csv_path: Path,
    output_parquet_path: Path,
    summary_path: Path,
    provenance_path: Path,
    repository_root: Path,
) -> None:
    """Materializa las vistas textuales y su trazabilidad."""

    repository_root = repository_root.resolve()

    corpus_path = _resolve_path(
        corpus_path,
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

    _validate_paths(
        corpus_path=corpus_path,
        output_csv_path=output_csv_path,
        output_parquet_path=output_parquet_path,
        summary_path=summary_path,
        provenance_path=provenance_path,
    )

    for path in (
        corpus_path,
        output_csv_path,
        output_parquet_path,
        summary_path,
        provenance_path,
    ):
        _portable_relative_path(
            path,
            repository_root,
        )

    _require_source_file(corpus_path)

    corpus = pd.read_csv(
        corpus_path,
        dtype=str,
        keep_default_na=False,
    )

    text_inputs = build_openclip_text_inputs(
        corpus
    )

    summary = build_text_inputs_summary(
        text_inputs
    )

    _atomic_write_csv(
        text_inputs,
        output_csv_path,
    )

    _atomic_write_parquet(
        text_inputs,
        output_parquet_path,
    )

    _atomic_write_csv(
        summary,
        summary_path,
    )

    provenance = _build_provenance(
        corpus_path=corpus_path,
        output_csv_path=output_csv_path,
        output_parquet_path=output_parquet_path,
        summary_path=summary_path,
        repository_root=repository_root,
        source_records=len(corpus),
        text_inputs=text_inputs,
    )

    _atomic_write_json(
        provenance,
        provenance_path,
    )

def _print_materialization_summary(
    *,
    output_csv_path: Path,
    summary_path: Path,
    output_parquet_path: Path,
    provenance_path: Path,
) -> None:
    """Muestra un resumen de los artefactos textuales generados."""

    text_inputs = pd.read_csv(
        output_csv_path,
        dtype=str,
        keep_default_na=False,
    )

    summary = pd.read_csv(
        summary_path,
        keep_default_na=False,
    )

    print("\nENTRADAS TEXTUALES OPENCLIP")
    print("=" * 100)
    print(f"Registros: {len(text_inputs)}")
    print(f"Columnas: {len(text_inputs.columns)}")

    print("\nCOBERTURA")
    print("=" * 100)
    print(summary.to_string(index=False))

    print("\nARTEFACTOS")
    print("=" * 100)

    for path in (
        output_csv_path,
        output_parquet_path,
        summary_path,
        provenance_path,
    ):
        print(
            path.resolve()
            .relative_to(PROJECT_ROOT.resolve())
        )

def main() -> None:
    """Materializa las entradas textuales OpenCLIP v1."""

    materialize_openclip_text_inputs(
        corpus_path=DEFAULT_CORPUS_PATH,
        output_csv_path=DEFAULT_OUTPUT_CSV_PATH,
        output_parquet_path=DEFAULT_OUTPUT_PARQUET_PATH,
        summary_path=DEFAULT_SUMMARY_PATH,
        provenance_path=DEFAULT_PROVENANCE_PATH,
        repository_root=PROJECT_ROOT,
    )

    _print_materialization_summary(
        output_csv_path=DEFAULT_OUTPUT_CSV_PATH,
        output_parquet_path=DEFAULT_OUTPUT_PARQUET_PATH,
        summary_path=DEFAULT_SUMMARY_PATH,
        provenance_path=DEFAULT_PROVENANCE_PATH,
    )


if __name__ == "__main__":
    main()
