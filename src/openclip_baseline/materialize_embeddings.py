"""Materialización reproducible de embeddings OpenCLIP."""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from src.openclip_baseline.embeddings import (
    OpenCLIPEmbeddingConfig,
    extract_openclip_embeddings,
)


DATASET_NAME = "openclip_embeddings_v1"
PIPELINE_VERSION = "openclip_embedding_materialization_v1"

NPZ_ARRAY_ORDER = (
    "item_ids",
    "museums",
    "image_embeddings",
    "text_visual_embeddings",
    "text_metadata_embeddings",
)

FIXED_ZIP_TIMESTAMP = (
    1980,
    1,
    1,
    0,
    0,
    0,
)


def _resolve_path(
    path: Path,
    repository_root: Path,
) -> Path:
    """Resuelve una ruta respecto de la raíz del repositorio."""

    if path.is_absolute():
        return path.resolve()

    return (
        repository_root
        / path
    ).resolve()


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
            "El artefacto debe permanecer dentro de la raíz "
            f"del repositorio: {resolved_path}"
        ) from exc

    return relative_path.as_posix()


def _validate_paths(
    *,
    corpus_path: Path,
    text_inputs_path: Path,
    output_npz_path: Path,
    output_index_path: Path,
    summary_path: Path,
    provenance_path: Path,
) -> None:
    """Impide sobrescrituras de fuentes y colisiones entre salidas."""

    source_paths = {
        corpus_path.resolve(),
        text_inputs_path.resolve(),
    }

    output_paths = [
        output_npz_path.resolve(),
        output_index_path.resolve(),
        summary_path.resolve(),
        provenance_path.resolve(),
    ]

    if len(set(output_paths)) != len(output_paths):
        raise ValueError(
            "Las rutas de salida deben ser distintas."
        )

    conflicting_sources = source_paths.intersection(
        output_paths
    )

    if conflicting_sources:
        raise ValueError(
            "No se permite sobrescribir un archivo fuente."
        )


def _require_source_file(
    path: Path,
) -> None:
    """Comprueba que una fuente exista."""

    if not path.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo fuente: {path}"
        )


def _temporary_path(
    destination: Path,
) -> Path:
    """Genera una ruta temporal junto al destino."""

    return destination.with_name(
        f".{destination.name}.{uuid4().hex}.tmp"
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


def _atomic_write_csv(
    frame: pd.DataFrame,
    destination: Path,
) -> None:
    """Escribe un CSV mediante reemplazo atómico."""

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = _temporary_path(
        destination
    )

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


def _atomic_write_json(
    payload: dict[str, object],
    destination: Path,
) -> None:
    """Escribe JSON UTF-8 mediante reemplazo atómico."""

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = _temporary_path(
        destination
    )

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


def _array_to_npy_bytes(
    array: np.ndarray,
) -> bytes:
    """Serializa un arreglo NumPy sin permitir objetos Python."""

    buffer = io.BytesIO()

    np.save(
        buffer,
        array,
        allow_pickle=False,
    )

    return buffer.getvalue()


def _write_deterministic_npz(
    *,
    destination: Path,
    arrays: dict[str, np.ndarray],
) -> None:
    """Escribe un NPZ determinista con metadatos ZIP fijos.

    ``numpy.savez`` incorpora metadatos ZIP que pueden variar entre
    ejecuciones. Esta función fija el orden de los miembros, sus marcas
    de tiempo y sus permisos para permitir comparación byte a byte.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = _temporary_path(
        destination
    )

    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_STORED,
            strict_timestamps=False,
        ) as archive:
            for name in NPZ_ARRAY_ORDER:
                if name not in arrays:
                    raise ValueError(
                        "Falta un arreglo requerido para el NPZ: "
                        f"{name}"
                    )

                array = np.asarray(
                    arrays[name]
                )

                information = zipfile.ZipInfo(
                    filename=f"{name}.npy",
                    date_time=FIXED_ZIP_TIMESTAMP,
                )

                information.compress_type = (
                    zipfile.ZIP_STORED
                )

                information.create_system = 3

                information.external_attr = (
                    0o600
                    << 16
                )

                archive.writestr(
                    information,
                    _array_to_npy_bytes(
                        array
                    ),
                )

        os.replace(
            temporary,
            destination,
        )
    finally:
        temporary.unlink(
            missing_ok=True
        )


def _build_index(
    *,
    item_ids: tuple[str, ...],
    museums: tuple[str, ...],
) -> pd.DataFrame:
    """Construye el índice tabular del archivo NPZ."""

    if len(item_ids) != len(museums):
        raise ValueError(
            "item_ids y museums deben tener la misma longitud."
        )

    return pd.DataFrame(
        {
            "row_index": np.arange(
                len(item_ids),
                dtype=np.int64,
            ),
            "item_id": list(
                item_ids
            ),
            "museum": list(
                museums
            ),
        }
    )


def _summarize_matrix(
    *,
    modality: str,
    matrix: np.ndarray,
) -> dict[str, object]:
    """Resume las propiedades numéricas de una matriz."""

    norms = np.linalg.norm(
        matrix,
        axis=1,
    )

    finite_values = bool(
        np.isfinite(
            matrix
        ).all()
    )

    normalized_l2 = bool(
        finite_values
        and np.allclose(
            norms,
            np.ones(
                len(norms),
                dtype=np.float32,
            ),
            atol=1e-5,
            rtol=0.0,
        )
    )

    return {
        "modality": modality,
        "total_records": int(
            matrix.shape[0]
        ),
        "embedding_dimension": int(
            matrix.shape[1]
        ),
        "dtype": str(
            matrix.dtype
        ),
        "finite_values": finite_values,
        "normalized_l2": normalized_l2,
        "mean_l2_norm": float(
            norms.mean()
        ),
        "min_l2_norm": float(
            norms.min()
        ),
        "max_l2_norm": float(
            norms.max()
        ),
    }


def build_embeddings_summary(
    *,
    image_embeddings: np.ndarray,
    text_visual_embeddings: np.ndarray,
    text_metadata_embeddings: np.ndarray,
) -> pd.DataFrame:
    """Construye el resumen de las tres modalidades."""

    rows = [
        _summarize_matrix(
            modality="image",
            matrix=image_embeddings,
        ),
        _summarize_matrix(
            modality="text_visual",
            matrix=text_visual_embeddings,
        ),
        _summarize_matrix(
            modality="text_metadata",
            matrix=text_metadata_embeddings,
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "modality",
            "total_records",
            "embedding_dimension",
            "dtype",
            "finite_values",
            "normalized_l2",
            "mean_l2_norm",
            "min_l2_norm",
            "max_l2_norm",
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
        "sha256": _sha256_file(
            path
        ),
        "size_bytes": path.stat().st_size,
    }


def _build_provenance(
    *,
    corpus_path: Path,
    text_inputs_path: Path,
    output_npz_path: Path,
    output_index_path: Path,
    summary_path: Path,
    repository_root: Path,
    corpus_records: int,
    text_records: int,
    output_records: int,
    embedding_dimension: int,
    config: OpenCLIPEmbeddingConfig,
) -> dict[str, object]:
    """Construye la declaración portable de procedencia."""

    return {
        "dataset_name": DATASET_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "coverage": {
            "corpus_records": corpus_records,
            "text_records": text_records,
            "output_records": output_records,
        },
        "model": {
            "model_name": config.model_name,
            "pretrained": config.pretrained,
            "device": config.device,
            "precision": config.precision,
            "batch_size": config.batch_size,
            "normalize": config.normalize,
            "embedding_dimension": embedding_dimension,
        },
        "inputs": {
            "corpus": _artifact_metadata(
                corpus_path,
                repository_root,
            ),
            "text_inputs": _artifact_metadata(
                text_inputs_path,
                repository_root,
            ),
        },
        "outputs": {
            "npz": _artifact_metadata(
                output_npz_path,
                repository_root,
            ),
            "index": _artifact_metadata(
                output_index_path,
                repository_root,
            ),
            "summary": _artifact_metadata(
                summary_path,
                repository_root,
            ),
        },
    }


def materialize_openclip_embeddings(
    *,
    corpus_path: Path,
    text_inputs_path: Path,
    output_npz_path: Path,
    output_index_path: Path,
    summary_path: Path,
    provenance_path: Path,
    repository_root: Path,
    model: Any,
    preprocess: Any,
    tokenizer: Any,
    config: OpenCLIPEmbeddingConfig,
) -> None:
    """Extrae y materializa embeddings con trazabilidad."""

    repository_root = (
        repository_root.resolve()
    )

    corpus_path = _resolve_path(
        corpus_path,
        repository_root,
    )

    text_inputs_path = _resolve_path(
        text_inputs_path,
        repository_root,
    )

    output_npz_path = _resolve_path(
        output_npz_path,
        repository_root,
    )

    output_index_path = _resolve_path(
        output_index_path,
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
        text_inputs_path=text_inputs_path,
        output_npz_path=output_npz_path,
        output_index_path=output_index_path,
        summary_path=summary_path,
        provenance_path=provenance_path,
    )

    for path in (
        corpus_path,
        text_inputs_path,
        output_npz_path,
        output_index_path,
        summary_path,
        provenance_path,
    ):
        _portable_relative_path(
            path,
            repository_root,
        )

    _require_source_file(
        corpus_path
    )

    _require_source_file(
        text_inputs_path
    )

    corpus = pd.read_csv(
        corpus_path,
        dtype=str,
        keep_default_na=False,
    )

    text_inputs = pd.read_csv(
        text_inputs_path,
        dtype=str,
        keep_default_na=False,
    )

    result = extract_openclip_embeddings(
        corpus=corpus,
        text_inputs=text_inputs,
        repository_root=repository_root,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        config=config,
    )

    arrays = {
        "item_ids": np.asarray(
            result.item_ids,
            dtype=np.str_,
        ),
        "museums": np.asarray(
            result.museums,
            dtype=np.str_,
        ),
        "image_embeddings": (
            result.image_embeddings
        ),
        "text_visual_embeddings": (
            result.text_visual_embeddings
        ),
        "text_metadata_embeddings": (
            result.text_metadata_embeddings
        ),
    }

    index = _build_index(
        item_ids=result.item_ids,
        museums=result.museums,
    )

    summary = build_embeddings_summary(
        image_embeddings=(
            result.image_embeddings
        ),
        text_visual_embeddings=(
            result.text_visual_embeddings
        ),
        text_metadata_embeddings=(
            result.text_metadata_embeddings
        ),
    )

    _write_deterministic_npz(
        destination=output_npz_path,
        arrays=arrays,
    )

    _atomic_write_csv(
        index,
        output_index_path,
    )

    _atomic_write_csv(
        summary,
        summary_path,
    )

    provenance = _build_provenance(
        corpus_path=corpus_path,
        text_inputs_path=text_inputs_path,
        output_npz_path=output_npz_path,
        output_index_path=output_index_path,
        summary_path=summary_path,
        repository_root=repository_root,
        corpus_records=len(corpus),
        text_records=len(text_inputs),
        output_records=len(
            result.item_ids
        ),
        embedding_dimension=(
            result.embedding_dimension
        ),
        config=config,
    )

    _atomic_write_json(
        provenance,
        provenance_path,
    )