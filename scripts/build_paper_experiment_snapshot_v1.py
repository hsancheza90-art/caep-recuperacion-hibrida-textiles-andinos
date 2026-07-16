"""Construye el snapshot reproducible del experimento del paper."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd


SNAPSHOT_VERSION = "paper_experiment_snapshot_v1"

ARTIFACT_PATHS = (
    "data/derived/paper_corpus_multimodal_v1.csv",
    "data/derived/openclip_text_inputs_v1.csv",
    "data/derived/openclip_embeddings_v1.npz",
    "data/derived/openclip_embeddings_index_v1.csv",
    "outputs/reports/openclip_embeddings_summary_v1.csv",
    "outputs/reports/openclip_embeddings_provenance_v1.json",
    "outputs/reports/openclip_retrieval_summary_v1.csv",
    "outputs/reports/openclip_retrieval_per_query_v1.csv",
    "outputs/reports/openclip_alpha_grid_v1.csv",
    "outputs/reports/openclip_comparison_vs_random_v1.csv",
    "outputs/reports/openclip_retrieval_provenance_v1.json",
    "outputs/reports/openclip_bootstrap_replicates_v1.csv",
    "outputs/reports/openclip_bootstrap_summary_v1.csv",
    "outputs/reports/openclip_bootstrap_provenance_v1.json",
    "outputs/reports/openclip_top1_candidate_counts_v1.csv",
    "outputs/reports/openclip_top1_hubness_summary_v1.csv",
    "outputs/reports/openclip_top1_museum_flow_v1.csv",
    "outputs/reports/openclip_top1_persistent_hubs_v1.csv",
    "outputs/reports/openclip_top1_hubness_provenance_v1.json",
    "outputs/reports/openclip_hub_audit_profiles_v1.csv",
    "outputs/reports/openclip_hub_audit_configuration_counts_v1.csv",
    "outputs/reports/openclip_hub_audit_attraction_events_v1.csv",
    "outputs/reports/openclip_hub_audit_summary_v1.csv",
    "outputs/reports/openclip_hub_audit_provenance_v1.json",
)


def _sha256_file(
    path: Path,
) -> str:
    """Calcula el hash SHA-256 de un archivo."""

    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _run_git(
    root: Path,
    *arguments: str,
) -> str:
    """Ejecuta Git y devuelve una cadena vacía si falla."""

    process = subprocess.run(
        [
            "git",
            *arguments,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    if process.returncode != 0:
        return ""

    return process.stdout.strip()


def _inspect_csv(
    path: Path,
) -> dict[str, Any]:
    """Registra dimensiones y esquema de un CSV."""

    frame = pd.read_csv(
        path,
        keep_default_na=False,
        low_memory=False,
    )

    return {
        "format": "csv",
        "rows": len(frame),
        "columns": len(frame.columns),
        "column_names": list(
            frame.columns.astype(str)
        ),
    }


def _inspect_json(
    path: Path,
) -> dict[str, Any]:
    """Valida un JSON y registra sus claves principales."""

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if isinstance(payload, dict):
        top_level_keys = sorted(
            str(key)
            for key in payload
        )
        root_type = "object"
    elif isinstance(payload, list):
        top_level_keys = []
        root_type = "array"
    else:
        top_level_keys = []
        root_type = type(
            payload
        ).__name__

    return {
        "format": "json",
        "valid_json": True,
        "root_type": root_type,
        "top_level_keys": top_level_keys,
    }


def _inspect_npz(
    path: Path,
) -> dict[str, Any]:
    """Registra los arreglos almacenados en un NPZ."""

    arrays: dict[str, Any] = {}

    with np.load(
        path,
        allow_pickle=False,
    ) as archive:
        for name in sorted(archive.files):
            array = archive[name]

            record: dict[str, Any] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "element_count": int(
                    array.size
                ),
            }

            if np.issubdtype(
                array.dtype,
                np.number,
            ):
                record["data_type"] = "numeric"
                record["finite"] = bool(
                    np.isfinite(array).all()
                )
            elif np.issubdtype(
                array.dtype,
                np.bool_,
            ):
                record["data_type"] = "boolean"
                record["finite"] = None
            elif np.issubdtype(
                array.dtype,
                np.str_,
            ):
                values = array.astype(str)

                record["data_type"] = "string"
                record["finite"] = None
                record["nonempty_elements"] = int(
                    np.count_nonzero(
                        np.char.str_len(values) > 0
                    )
                )
                record["unique_elements"] = int(
                    np.unique(values).size
                )
            elif np.issubdtype(
                array.dtype,
                np.bytes_,
            ):
                record["data_type"] = "bytes"
                record["finite"] = None
                record["unique_elements"] = int(
                    np.unique(array).size
                )
            else:
                record["data_type"] = "other"
                record["finite"] = None

            arrays[name] = record

    return {
        "format": "npz",
        "arrays": arrays,
    }

def _inspect_artifact(
    path: Path,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    """Construye el registro completo de un artefacto."""

    relative_path = path.relative_to(
        repository_root
    ).as_posix()

    record: dict[str, Any] = {
        "path": relative_path,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }

    suffix = path.suffix.lower()

    if suffix == ".csv":
        record.update(
            _inspect_csv(path)
        )
    elif suffix == ".json":
        record.update(
            _inspect_json(path)
        )
    elif suffix == ".npz":
        record.update(
            _inspect_npz(path)
        )
    else:
        record["format"] = (
            suffix.lstrip(".")
            or "unknown"
        )

    return record


def _build_corpus_summary(
    corpus_path: Path,
) -> dict[str, Any]:
    """Resume la composición y consistencia del corpus."""

    corpus = pd.read_csv(
        corpus_path,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    required = {
        "item_id",
        "museum",
        "image_local_path",
        "image_download_status",
    }

    missing = required - set(
        corpus.columns
    )

    if missing:
        raise ValueError(
            "El corpus no contiene las columnas "
            f"requeridas: {sorted(missing)}"
        )

    museum_counts = (
        corpus["museum"]
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict()
    )

    local_paths = corpus[
        "image_local_path"
    ].astype(str)

    repository_root = (
        corpus_path.parents[2]
    )

    existing_images = 0

    for value in local_paths:
        image_path = Path(value)

        if not image_path.is_absolute():
            image_path = (
                repository_root
                / image_path
            )

        if image_path.is_file():
            existing_images += 1

    return {
        "records": len(corpus),
        "unique_item_ids": int(
            corpus["item_id"].nunique()
        ),
        "item_ids_are_unique": bool(
            corpus["item_id"].is_unique
        ),
        "museum_counts": museum_counts,
        "records_with_local_path": int(
            local_paths.str.strip().ne("").sum()
        ),
        "existing_local_images": (
            existing_images
        ),
        "download_status_counts": (
            corpus[
                "image_download_status"
            ]
            .value_counts()
            .sort_index()
            .astype(int)
            .to_dict()
        ),
    }


def build_snapshot(
    repository_root: Path,
) -> dict[str, Any]:
    """Construye el snapshot completo del experimento."""

    missing_artifacts = []

    for relative in ARTIFACT_PATHS:
        path = repository_root / relative

        if not path.is_file():
            missing_artifacts.append(
                relative
            )

    if missing_artifacts:
        formatted = "\n".join(
            f"- {path}"
            for path in missing_artifacts
        )

        raise FileNotFoundError(
            "Faltan artefactos requeridos:\n"
            f"{formatted}"
        )

    artifacts = [
        _inspect_artifact(
            repository_root / relative,
            repository_root=repository_root,
        )
        for relative in ARTIFACT_PATHS
    ]

    branch = _run_git(
        repository_root,
        "branch",
        "--show-current",
    )

    commit = _run_git(
        repository_root,
        "rev-parse",
        "HEAD",
    )

    status_text = _run_git(
        repository_root,
        "status",
        "--porcelain",
    )

    dirty_entries = (
        status_text.splitlines()
        if status_text
        else []
    )

    corpus_path = (
        repository_root
        / "data"
        / "derived"
        / "paper_corpus_multimodal_v1.csv"
    )

    return {
        "snapshot_version": SNAPSHOT_VERSION,
        "generated_at_utc": (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        ),
        "repository": {
            "root": (
                repository_root.resolve()
                .as_posix()
            ),
            "branch": branch,
            "commit": commit,
            "working_tree_clean": (
                len(dirty_entries) == 0
            ),
            "dirty_entries": dirty_entries,
        },
        "environment": {
            "python_version": (
                sys.version.split()[0]
            ),
            "python_implementation": (
                platform.python_implementation()
            ),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
        "corpus": _build_corpus_summary(
            corpus_path
        ),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def _atomic_write_json(
    destination: Path,
    payload: dict[str, Any],
) -> None:
    """Escribe el JSON mediante reemplazo atómico."""

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_name(
        f".{destination.name}.{uuid4().hex}.tmp"
    )

    content = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )

    try:
        temporary.write_text(
            f"{content}\n",
            encoding="utf-8",
            newline="\n",
        )

        os.replace(
            temporary,
            destination,
        )
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    """Construye y valida el snapshot del paper."""

    repository_root = Path.cwd()

    output_path = (
        repository_root
        / "outputs"
        / "paper"
        / "paper_experiment_snapshot_v1.json"
    )

    snapshot = build_snapshot(
        repository_root
    )

    _atomic_write_json(
        output_path,
        snapshot,
    )

    corpus = snapshot["corpus"]
    repository = snapshot[
        "repository"
    ]

    print()
    print("SNAPSHOT DEL EXPERIMENTO GENERADO")
    print("=" * 100)
    print(
        f"Rama:                "
        f"{repository['branch']}"
    )
    print(
        f"Commit:              "
        f"{repository['commit']}"
    )
    print(
        "Working tree limpio: "
        f"{repository['working_tree_clean']}"
    )
    print(
        f"Registros corpus:     "
        f"{corpus['records']}"
    )
    print(
        f"IDs únicos:           "
        f"{corpus['unique_item_ids']}"
    )
    print(
        f"Imágenes existentes:  "
        f"{corpus['existing_local_images']}"
    )
    print(
        f"Distribución museo:   "
        f"{corpus['museum_counts']}"
    )
    print(
        f"Artefactos auditados: "
        f"{snapshot['artifact_count']}"
    )
    print()
    print(output_path)


if __name__ == "__main__":
    main()