"""Materialización reproducible de la auditoría de hubs OpenCLIP."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from src.openclip_baseline.hub_audit import (
    OpenCLIPHubAudit,
    build_openclip_hub_audit,
)


DATASET_NAME = "openclip_hub_audit_v1"
PIPELINE_VERSION = "openclip_hub_audit_v1"


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


def _relative_path(
    path: Path,
    repository_root: Path,
) -> str:
    """Representa una ruta relativa cuando es posible."""

    resolved = path.resolve()
    root = repository_root.resolve()

    try:
        return resolved.relative_to(
            root
        ).as_posix()
    except ValueError:
        return resolved.as_posix()


def _file_record(
    path: Path,
    repository_root: Path,
) -> dict[str, object]:
    """Construye un registro de trazabilidad."""

    return {
        "path": _relative_path(
            path,
            repository_root,
        ),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _atomic_write_text(
    destination: Path,
    content: str,
) -> None:
    """Escribe texto mediante reemplazo atómico."""

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_name(
        f".{destination.name}.{uuid4().hex}.tmp"
    )

    try:
        temporary.write_text(
            content,
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


def _atomic_write_csv(
    frame: pd.DataFrame,
    destination: Path,
) -> None:
    """Escribe una tabla CSV estable."""

    content = frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    )

    _atomic_write_text(
        destination,
        content,
    )


def _atomic_write_json(
    payload: dict[str, Any],
    destination: Path,
) -> None:
    """Escribe un documento JSON estable."""

    content = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )

    _atomic_write_text(
        destination,
        f"{content}\n",
    )


def build_openclip_hub_audit_artifacts(
    *,
    corpus: pd.DataFrame,
    text_inputs: pd.DataFrame,
    per_query: pd.DataFrame,
    persistent_hubs: pd.DataFrame,
    candidate_counts: pd.DataFrame,
    repository_root: Path,
) -> OpenCLIPHubAudit:
    """Construye las tablas de auditoría enriquecida."""

    return build_openclip_hub_audit(
        corpus=corpus,
        text_inputs=text_inputs,
        per_query=per_query,
        persistent_hubs=persistent_hubs,
        candidate_counts=candidate_counts,
        repository_root=repository_root,
    )


def _require_source_file(
    path: Path,
    *,
    label: str,
) -> None:
    """Valida la existencia de un archivo de entrada."""

    if not path.is_file():
        raise FileNotFoundError(
            f"No existe el archivo {label}: {path}"
        )


def materialize_openclip_hub_audit(
    *,
    corpus_path: Path,
    text_inputs_path: Path,
    per_query_path: Path,
    persistent_hubs_path: Path,
    candidate_counts_path: Path,
    hub_profiles_path: Path,
    hub_configuration_counts_path: Path,
    attraction_events_path: Path,
    summary_path: Path,
    provenance_path: Path,
    repository_root: Path,
) -> None:
    """Construye y materializa la auditoría de hubs."""

    for path, label in (
        (
            corpus_path,
            "del corpus",
        ),
        (
            text_inputs_path,
            "de entradas textuales",
        ),
        (
            per_query_path,
            "de resultados por consulta",
        ),
        (
            persistent_hubs_path,
            "de hubs persistentes",
        ),
        (
            candidate_counts_path,
            "de conteos de candidatos",
        ),
    ):
        _require_source_file(
            path,
            label=label,
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

    per_query = pd.read_csv(
        per_query_path,
        dtype={
            "configuration": str,
            "item_id": str,
            "museum": str,
            "top1_item_id": str,
            "top1_museum": str,
        },
        keep_default_na=False,
    )

    persistent_hubs = pd.read_csv(
        persistent_hubs_path,
        dtype={
            "candidate_item_id": str,
            "candidate_museum": str,
        },
        keep_default_na=False,
    )

    candidate_counts = pd.read_csv(
        candidate_counts_path,
        dtype={
            "configuration": str,
            "candidate_item_id": str,
            "candidate_museum": str,
        },
        keep_default_na=False,
    )

    audit = build_openclip_hub_audit_artifacts(
        corpus=corpus,
        text_inputs=text_inputs,
        per_query=per_query,
        persistent_hubs=persistent_hubs,
        candidate_counts=candidate_counts,
        repository_root=repository_root,
    )

    _atomic_write_csv(
        audit.hub_profiles,
        hub_profiles_path,
    )
    _atomic_write_csv(
        audit.hub_configuration_counts,
        hub_configuration_counts_path,
    )
    _atomic_write_csv(
        audit.attraction_events,
        attraction_events_path,
    )
    _atomic_write_csv(
        audit.summary,
        summary_path,
    )

    provenance = {
        "dataset_name": DATASET_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "coverage": {
            "corpus_records": len(corpus),
            "text_records": len(text_inputs),
            "per_query_records": len(per_query),
            "persistent_hub_records": len(
                persistent_hubs
            ),
            "hub_profile_rows": len(
                audit.hub_profiles
            ),
            "hub_configuration_rows": len(
                audit.hub_configuration_counts
            ),
            "attraction_event_rows": len(
                audit.attraction_events
            ),
            "summary_rows": len(
                audit.summary
            ),
            "available_hub_images": int(
                audit.hub_profiles[
                    "image_exists"
                ].sum()
            ),
        },
        "configuration": {
            "hub_scope": "persistent_top1_hubs",
            "event_definition": (
                "query_whose_top1_candidate_is_a_persistent_hub"
            ),
            "same_culture_definition": (
                "nonempty_equal_culture_canonical"
            ),
            "same_object_type_definition": (
                "nonempty_equal_object_type"
            ),
        },
        "inputs": {
            "corpus": _file_record(
                corpus_path,
                repository_root,
            ),
            "text_inputs": _file_record(
                text_inputs_path,
                repository_root,
            ),
            "per_query": _file_record(
                per_query_path,
                repository_root,
            ),
            "persistent_hubs": _file_record(
                persistent_hubs_path,
                repository_root,
            ),
            "candidate_counts": _file_record(
                candidate_counts_path,
                repository_root,
            ),
        },
        "outputs": {
            "hub_profiles": _file_record(
                hub_profiles_path,
                repository_root,
            ),
            "hub_configuration_counts": (
                _file_record(
                    hub_configuration_counts_path,
                    repository_root,
                )
            ),
            "attraction_events": _file_record(
                attraction_events_path,
                repository_root,
            ),
            "summary": _file_record(
                summary_path,
                repository_root,
            ),
        },
    }

    _atomic_write_json(
        provenance,
        provenance_path,
    )