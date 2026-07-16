"""Materialización reproducible de la evaluación OpenCLIP."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from src.openclip_baseline.evaluation import (
    build_random_paired_baseline,
    evaluate_alpha_grid,
    evaluate_fused_text_retrieval,
    evaluate_paired_retrieval,
)
from src.openclip_baseline.retrieval import (
    OpenCLIPEmbeddingStore,
    load_openclip_embedding_store,
)


DATASET_NAME = "openclip_retrieval_evaluation_v1"
PIPELINE_VERSION = "openclip_retrieval_evaluation_v1"


@dataclass(frozen=True)
class OpenCLIPEvaluationConfig:
    """Configuración de evaluación y comparación."""

    cutoffs: tuple[int, ...] = (
        1,
        5,
        10,
    )
    alphas: tuple[float, ...] = (
        0.00,
        0.25,
        0.50,
        0.75,
        1.00,
    )
    selected_alpha: float = 0.50

    def __post_init__(self) -> None:
        """Valida los parámetros independientes del corpus."""

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

        if not self.alphas:
            raise ValueError(
                "alphas no puede estar vacío."
            )

        if any(
            not 0.0 <= alpha <= 1.0
            for alpha in self.alphas
        ):
            raise ValueError(
                "Todos los valores de alpha deben "
                "estar entre 0 y 1."
            )

        if not 0.0 <= self.selected_alpha <= 1.0:
            raise ValueError(
                "selected_alpha debe estar entre 0 y 1."
            )


@dataclass(frozen=True)
class OpenCLIPEvaluationArtifacts:
    """Tablas resultantes de la evaluación."""

    summary: pd.DataFrame
    per_query: pd.DataFrame
    alpha_grid: pd.DataFrame
    comparison_vs_random: pd.DataFrame


def _configuration_name(
    alpha: float,
) -> str:
    """Construye un nombre estable para una fusión."""

    return f"text_fused_alpha_{alpha:.2f}"


def _label_summary(
    frame: pd.DataFrame,
    *,
    configuration: str,
) -> pd.DataFrame:
    """Añade una etiqueta de configuración al resumen."""

    labeled = frame.copy()

    labeled.insert(
        0,
        "configuration",
        configuration,
    )

    return labeled


def _label_per_query(
    frame: pd.DataFrame,
    *,
    configuration: str,
    alpha: float | None,
) -> pd.DataFrame:
    """Añade la configuración a resultados por consulta."""

    labeled = frame.copy()

    labeled.insert(
        0,
        "configuration",
        configuration,
    )
    labeled.insert(
        1,
        "alpha",
        (
            np.nan
            if alpha is None
            else float(alpha)
        ),
    )

    return labeled


def _build_comparison_vs_random(
    *,
    metadata_summary: pd.DataFrame,
    fused_summary: pd.DataFrame,
    random_baseline: pd.DataFrame,
    selected_alpha: float,
) -> pd.DataFrame:
    """Compara las configuraciones seleccionadas con el azar."""

    metadata_all = metadata_summary.loc[
        metadata_summary["scope"] == "ALL"
    ].copy()

    metadata_all.insert(
        0,
        "configuration",
        "text_metadata",
    )

    fused_all = fused_summary.loc[
        fused_summary["scope"] == "ALL"
    ].copy()

    fused_all.insert(
        0,
        "configuration",
        _configuration_name(
            selected_alpha
        ),
    )

    comparison = pd.concat(
        [
            metadata_all,
            fused_all,
        ],
        ignore_index=True,
    )

    baseline = random_baseline.iloc[0]

    for metric in (
        "mrr",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
    ):
        if metric not in comparison.columns:
            continue

        comparison[
            f"{metric}_lift_vs_random"
        ] = (
            comparison[metric]
            / float(baseline[metric])
        )

    comparison[
        "mean_rank_reduction_vs_random"
    ] = (
        1.0
        - comparison["mean_rank"]
        / float(baseline["mean_rank"])
    )

    comparison[
        "median_rank_reduction_vs_random"
    ] = (
        1.0
        - comparison["median_rank"]
        / float(baseline["median_rank"])
    )

    return comparison


def build_openclip_evaluation_artifacts(
    *,
    store: OpenCLIPEmbeddingStore,
    config: OpenCLIPEvaluationConfig,
) -> OpenCLIPEvaluationArtifacts:
    """Construye todas las tablas de evaluación."""

    if any(
        cutoff > store.total_records
        for cutoff in config.cutoffs
    ):
        raise ValueError(
            "Ningún cutoff puede superar el número "
            "de registros."
        )

    visual = evaluate_paired_retrieval(
        store=store,
        query_modality="text_visual",
        candidate_modality="image",
        cutoffs=config.cutoffs,
    )

    metadata = evaluate_paired_retrieval(
        store=store,
        query_modality="text_metadata",
        candidate_modality="image",
        cutoffs=config.cutoffs,
    )

    fused = evaluate_fused_text_retrieval(
        store=store,
        alpha=config.selected_alpha,
        candidate_modality="image",
        cutoffs=config.cutoffs,
    )

    alpha_grid = evaluate_alpha_grid(
        store=store,
        alphas=config.alphas,
        candidate_modality="image",
        cutoffs=config.cutoffs,
    )

    alpha_grid = alpha_grid.copy()

    alpha_grid.insert(
        0,
        "configuration",
        alpha_grid["alpha"].map(
            _configuration_name
        ),
    )

    random_baseline = build_random_paired_baseline(
        candidate_count=store.total_records,
        cutoffs=config.cutoffs,
    )

    summary = pd.concat(
        [
            _label_summary(
                visual.summary,
                configuration="text_visual",
            ),
            _label_summary(
                metadata.summary,
                configuration="text_metadata",
            ),
            alpha_grid,
            _label_summary(
                random_baseline,
                configuration="random",
            ),
        ],
        ignore_index=True,
    )

    per_query = pd.concat(
        [
            _label_per_query(
                visual.per_query,
                configuration="text_visual",
                alpha=None,
            ),
            _label_per_query(
                metadata.per_query,
                configuration="text_metadata",
                alpha=None,
            ),
            _label_per_query(
                fused.per_query,
                configuration=_configuration_name(
                    config.selected_alpha
                ),
                alpha=config.selected_alpha,
            ),
        ],
        ignore_index=True,
    )

    comparison = _build_comparison_vs_random(
        metadata_summary=metadata.summary,
        fused_summary=fused.summary,
        random_baseline=random_baseline,
        selected_alpha=config.selected_alpha,
    )

    return OpenCLIPEvaluationArtifacts(
        summary=summary,
        per_query=per_query,
        alpha_grid=alpha_grid,
        comparison_vs_random=comparison,
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
    """Escribe una tabla CSV de forma estable."""

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
    """Escribe JSON válido, estable y legible."""

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
        "sha256": _sha256_file(
            path
        ),
        "size_bytes": path.stat().st_size,
    }


def materialize_openclip_evaluation(
    *,
    embeddings_path: Path,
    summary_path: Path,
    per_query_path: Path,
    alpha_grid_path: Path,
    comparison_path: Path,
    provenance_path: Path,
    repository_root: Path,
    config: OpenCLIPEvaluationConfig,
) -> None:
    """Evalúa embeddings y materializa resultados trazables."""

    store = load_openclip_embedding_store(
        embeddings_path
    )

    artifacts = build_openclip_evaluation_artifacts(
        store=store,
        config=config,
    )

    _atomic_write_csv(
        artifacts.summary,
        summary_path,
    )
    _atomic_write_csv(
        artifacts.per_query,
        per_query_path,
    )
    _atomic_write_csv(
        artifacts.alpha_grid,
        alpha_grid_path,
    )
    _atomic_write_csv(
        artifacts.comparison_vs_random,
        comparison_path,
    )

    provenance = {
        "dataset_name": DATASET_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "coverage": {
            "total_records": store.total_records,
            "embedding_dimension": (
                store.embedding_dimension
            ),
            "per_query_rows": len(
                artifacts.per_query
            ),
            "summary_rows": len(
                artifacts.summary
            ),
        },
        "configuration": {
            "cutoffs": list(
                config.cutoffs
            ),
            "alphas": list(
                config.alphas
            ),
            "selected_alpha": (
                config.selected_alpha
            ),
            "query_modalities": [
                "text_visual",
                "text_metadata",
                "text_fused",
            ],
            "candidate_modality": "image",
            "paired_candidate_included": True,
        },
        "input": {
            "embeddings": _file_record(
                embeddings_path,
                repository_root,
            )
        },
        "outputs": {
            "summary": _file_record(
                summary_path,
                repository_root,
            ),
            "per_query": _file_record(
                per_query_path,
                repository_root,
            ),
            "alpha_grid": _file_record(
                alpha_grid_path,
                repository_root,
            ),
            "comparison_vs_random": _file_record(
                comparison_path,
                repository_root,
            ),
        },
    }

    _atomic_write_json(
        provenance,
        provenance_path,
    )
    