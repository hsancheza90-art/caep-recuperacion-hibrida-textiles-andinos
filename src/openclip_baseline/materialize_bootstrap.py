"""Materialización reproducible del bootstrap OpenCLIP."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from src.openclip_baseline.bootstrap import (
    PairedBootstrapResult,
    paired_stratified_bootstrap,
)


DATASET_NAME = "openclip_paired_bootstrap_v1"
PIPELINE_VERSION = "openclip_paired_bootstrap_v1"


@dataclass(frozen=True)
class OpenCLIPBootstrapConfig:
    """Configuración del bootstrap pareado."""

    configuration_a: str = "text_metadata"
    configuration_b: str = "text_fused_alpha_0.50"
    cutoffs: tuple[int, ...] = (
        1,
        5,
        10,
    )
    n_resamples: int = 5000
    confidence_level: float = 0.95
    random_seed: int = 20260715

    def __post_init__(self) -> None:
        """Valida los parámetros del experimento."""

        if not self.configuration_a:
            raise ValueError(
                "configuration_a no puede estar vacía."
            )

        if not self.configuration_b:
            raise ValueError(
                "configuration_b no puede estar vacía."
            )

        if self.configuration_a == self.configuration_b:
            raise ValueError(
                "Las configuraciones deben ser diferentes."
            )

        if not self.cutoffs:
            raise ValueError(
                "cutoffs no puede estar vacío."
            )

        if any(
            cutoff <= 0
            for cutoff in self.cutoffs
        ):
            raise ValueError(
                "Todos los cutoffs deben ser positivos."
            )

        if self.n_resamples <= 0:
            raise ValueError(
                "n_resamples debe ser mayor que cero."
            )

        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError(
                "confidence_level debe estar entre 0 y 1."
            )


def build_openclip_bootstrap_artifacts(
    *,
    per_query: pd.DataFrame,
    config: OpenCLIPBootstrapConfig,
) -> PairedBootstrapResult:
    """Ejecuta el bootstrap con una configuración reproducible."""

    return paired_stratified_bootstrap(
        per_query=per_query,
        configuration_a=config.configuration_a,
        configuration_b=config.configuration_b,
        cutoffs=config.cutoffs,
        n_resamples=config.n_resamples,
        confidence_level=config.confidence_level,
        random_seed=config.random_seed,
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
    """Devuelve una ruta relativa cuando es posible."""

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


def materialize_openclip_bootstrap(
    *,
    per_query_path: Path,
    replicates_path: Path,
    summary_path: Path,
    provenance_path: Path,
    repository_root: Path,
    config: OpenCLIPBootstrapConfig,
) -> None:
    """Ejecuta y materializa el bootstrap pareado."""

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

    result = build_openclip_bootstrap_artifacts(
        per_query=per_query,
        config=config,
    )

    _atomic_write_csv(
        result.replicates,
        replicates_path,
    )
    _atomic_write_csv(
        result.summary,
        summary_path,
    )

    difference_rows = result.summary.loc[
        result.summary["estimate_type"]
        == "difference_b_minus_a"
    ]

    provenance = {
        "dataset_name": DATASET_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "configuration": {
            "configuration_a": config.configuration_a,
            "configuration_b": config.configuration_b,
            "difference_definition": (
                "configuration_b_minus_configuration_a"
            ),
            "cutoffs": list(config.cutoffs),
            "n_resamples": config.n_resamples,
            "confidence_level": (
                config.confidence_level
            ),
            "random_seed": config.random_seed,
            "paired": True,
            "stratified_by": "museum",
            "interval_method": "percentile",
        },
        "coverage": {
            "input_rows": len(per_query),
            "replicate_rows": len(
                result.replicates
            ),
            "summary_rows": len(
                result.summary
            ),
            "difference_summary_rows": len(
                difference_rows
            ),
        },
        "input": {
            "per_query": _file_record(
                per_query_path,
                repository_root,
            )
        },
        "outputs": {
            "replicates": _file_record(
                replicates_path,
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