"""Materialización reproducible del análisis de hubness OpenCLIP."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from src.openclip_baseline.hubness import (
    Top1HubnessAnalysis,
    analyze_top1_hubness,
)


DATASET_NAME = "openclip_top1_hubness_v1"
PIPELINE_VERSION = "openclip_top1_hubness_v1"


@dataclass(frozen=True)
class OpenCLIPHubnessConfig:
    """Configuración del análisis de hubness Top-1."""

    configurations: tuple[str, ...] = (
        "text_visual",
        "text_metadata",
        "text_fused_alpha_0.50",
    )
    hub_min_count: int = 3
    persistent_min_configurations: int = 2

    def __post_init__(self) -> None:
        """Valida los parámetros del análisis."""

        if not self.configurations:
            raise ValueError(
                "configurations no puede estar vacío."
            )

        if any(
            not configuration
            for configuration in self.configurations
        ):
            raise ValueError(
                "Las configuraciones no pueden estar vacías."
            )

        if len(set(self.configurations)) != len(
            self.configurations
        ):
            raise ValueError(
                "Las configuraciones deben ser únicas."
            )

        if self.hub_min_count <= 0:
            raise ValueError(
                "hub_min_count debe ser mayor que cero."
            )

        if self.persistent_min_configurations <= 0:
            raise ValueError(
                "persistent_min_configurations debe "
                "ser mayor que cero."
            )


def build_openclip_hubness_artifacts(
    *,
    per_query: pd.DataFrame,
    config: OpenCLIPHubnessConfig,
) -> Top1HubnessAnalysis:
    """Construye las tablas de hubness configuradas."""

    return analyze_top1_hubness(
        per_query=per_query,
        configurations=config.configurations,
        hub_min_count=config.hub_min_count,
        persistent_min_configurations=(
            config.persistent_min_configurations
        ),
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
    """Construye trazabilidad básica para un archivo."""

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


def materialize_openclip_hubness(
    *,
    per_query_path: Path,
    candidate_counts_path: Path,
    summary_path: Path,
    museum_flow_path: Path,
    persistent_hubs_path: Path,
    provenance_path: Path,
    repository_root: Path,
    config: OpenCLIPHubnessConfig,
) -> None:
    """Ejecuta y materializa el análisis de hubness."""

    if not per_query_path.is_file():
        raise FileNotFoundError(
            "No existe el archivo de resultados "
            f"por consulta: {per_query_path}"
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

    analysis = build_openclip_hubness_artifacts(
        per_query=per_query,
        config=config,
    )

    _atomic_write_csv(
        analysis.candidate_counts,
        candidate_counts_path,
    )
    _atomic_write_csv(
        analysis.summary,
        summary_path,
    )
    _atomic_write_csv(
        analysis.museum_flow,
        museum_flow_path,
    )
    _atomic_write_csv(
        analysis.persistent_hubs,
        persistent_hubs_path,
    )

    provenance = {
        "dataset_name": DATASET_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "configuration": {
            "configurations": list(
                config.configurations
            ),
            "hub_definition": (
                "top1_count_greater_or_equal_to_threshold"
            ),
            "hub_min_count": config.hub_min_count,
            "persistent_min_configurations": (
                config.persistent_min_configurations
            ),
            "ranking_position": 1,
        },
        "coverage": {
            "input_rows": len(per_query),
            "candidate_count_rows": len(
                analysis.candidate_counts
            ),
            "summary_rows": len(
                analysis.summary
            ),
            "museum_flow_rows": len(
                analysis.museum_flow
            ),
            "persistent_hub_rows": len(
                analysis.persistent_hubs
            ),
        },
        "input": {
            "per_query": _file_record(
                per_query_path,
                repository_root,
            )
        },
        "outputs": {
            "candidate_counts": _file_record(
                candidate_counts_path,
                repository_root,
            ),
            "summary": _file_record(
                summary_path,
                repository_root,
            ),
            "museum_flow": _file_record(
                museum_flow_path,
                repository_root,
            ),
            "persistent_hubs": _file_record(
                persistent_hubs_path,
                repository_root,
            ),
        },
    }

    _atomic_write_json(
        provenance,
        provenance_path,
    )