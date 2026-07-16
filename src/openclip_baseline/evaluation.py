"""Evaluación de recuperación cruzada sobre embeddings OpenCLIP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.openclip_baseline.retrieval import (
    EmbeddingModality,
    OpenCLIPEmbeddingStore,
    fuse_text_embeddings,
)


@dataclass(frozen=True)
class PairedRetrievalEvaluation:
    """Resultados detallados y agregados de recuperación."""

    per_query: pd.DataFrame
    summary: pd.DataFrame


def _validate_cutoffs(
    cutoffs: Iterable[int],
    *,
    total_records: int,
) -> tuple[int, ...]:
    """Valida y ordena los valores de K."""

    normalized = tuple(
        sorted(
            {
                int(cutoff)
                for cutoff in cutoffs
            }
        )
    )

    if not normalized:
        raise ValueError(
            "cutoffs debe contener al menos un valor."
        )

    if any(cutoff <= 0 for cutoff in normalized):
        raise ValueError(
            "Todos los valores de cutoffs deben ser positivos."
        )

    if any(
        cutoff > total_records
        for cutoff in normalized
    ):
        raise ValueError(
            "Ningún cutoff puede superar el número "
            "total de candidatos."
        )

    return normalized


def _validate_query_candidate_matrices(
    *,
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    total_records: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Valida matrices alineadas de consultas y candidatos."""

    queries = np.asarray(
        query_embeddings,
        dtype=np.float32,
    )
    candidates = np.asarray(
        candidate_embeddings,
        dtype=np.float32,
    )

    if queries.ndim != 2:
        raise ValueError(
            "query_embeddings debe ser una matriz."
        )

    if candidates.ndim != 2:
        raise ValueError(
            "candidate_embeddings debe ser una matriz."
        )

    if queries.shape[0] != total_records:
        raise ValueError(
            "Debe existir una consulta por cada registro."
        )

    if candidates.shape[0] != total_records:
        raise ValueError(
            "Debe existir un candidato por cada registro."
        )

    if queries.shape[1] != candidates.shape[1]:
        raise ValueError(
            "Consultas y candidatos deben tener "
            "la misma dimensión."
        )

    if not np.isfinite(queries).all():
        raise ValueError(
            "query_embeddings contiene valores no finitos."
        )

    if not np.isfinite(candidates).all():
        raise ValueError(
            "candidate_embeddings contiene valores "
            "no finitos."
        )

    return queries, candidates


def _evaluate_aligned_matrices(
    *,
    store: OpenCLIPEmbeddingStore,
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    query_modality: str,
    candidate_modality: str,
    cutoffs: Iterable[int],
    alpha: float | None = None,
) -> PairedRetrievalEvaluation:
    """Evalúa la posición de cada candidato emparejado."""

    normalized_cutoffs = _validate_cutoffs(
        cutoffs,
        total_records=store.total_records,
    )

    queries, candidates = (
        _validate_query_candidate_matrices(
            query_embeddings=query_embeddings,
            candidate_embeddings=candidate_embeddings,
            total_records=store.total_records,
        )
    )

    similarity_matrix = (
        queries
        @ candidates.T
    )

    candidate_indices = np.arange(
        store.total_records,
        dtype=np.int64,
    )

    records: list[dict[str, object]] = []

    for query_index in range(
        store.total_records
    ):
        scores = np.asarray(
            similarity_matrix[query_index],
            dtype=np.float64,
        )

        order = np.lexsort(
            (
                candidate_indices,
                -scores,
            )
        )

        matched_position = np.flatnonzero(
            order == query_index
        )

        if len(matched_position) != 1:
            raise RuntimeError(
                "No se pudo identificar unívocamente "
                "el candidato emparejado."
            )

        matched_rank = int(
            matched_position[0] + 1
        )
        top1_index = int(order[0])

        records.append(
            {
                "query_index": query_index,
                "item_id": store.item_ids[
                    query_index
                ],
                "museum": store.museums[
                    query_index
                ],
                "matched_rank": matched_rank,
                "reciprocal_rank": (
                    1.0 / matched_rank
                ),
                "matched_score": float(
                    scores[query_index]
                ),
                "top1_index": top1_index,
                "top1_item_id": store.item_ids[
                    top1_index
                ],
                "top1_museum": store.museums[
                    top1_index
                ],
                "top1_score": float(
                    scores[top1_index]
                ),
                "top1_is_match": (
                    top1_index == query_index
                ),
            }
        )

    per_query = pd.DataFrame.from_records(
        records
    )

    summary = _build_evaluation_summary(
        per_query=per_query,
        query_modality=query_modality,
        candidate_modality=candidate_modality,
        cutoffs=normalized_cutoffs,
        alpha=alpha,
    )

    return PairedRetrievalEvaluation(
        per_query=per_query,
        summary=summary,
    )


def _build_evaluation_summary(
    *,
    per_query: pd.DataFrame,
    query_modality: str,
    candidate_modality: str,
    cutoffs: tuple[int, ...],
    alpha: float | None,
) -> pd.DataFrame:
    """Construye métricas globales y por museo de consulta."""

    groups: list[
        tuple[str, pd.DataFrame]
    ] = [
        (
            "ALL",
            per_query,
        )
    ]

    for museum, group in per_query.groupby(
        "museum",
        sort=True,
    ):
        groups.append(
            (
                str(museum),
                group,
            )
        )

    summary_records: list[
        dict[str, object]
    ] = []

    for scope, group in groups:
        ranks = group[
            "matched_rank"
        ].to_numpy(
            dtype=np.int64
        )

        record: dict[str, object] = {
            "scope": scope,
            "query_modality": query_modality,
            "candidate_modality": (
                candidate_modality
            ),
            "alpha": (
                np.nan
                if alpha is None
                else float(alpha)
            ),
            "total_queries": len(group),
            "mrr": float(
                group[
                    "reciprocal_rank"
                ].mean()
            ),
            "mean_rank": float(
                np.mean(ranks)
            ),
            "median_rank": float(
                np.median(ranks)
            ),
            "mean_matched_score": float(
                group[
                    "matched_score"
                ].mean()
            ),
        }

        for cutoff in cutoffs:
            record[
                f"recall_at_{cutoff}"
            ] = float(
                np.mean(
                    ranks <= cutoff
                )
            )

        summary_records.append(
            record
        )

    return pd.DataFrame.from_records(
        summary_records
    )


def evaluate_paired_retrieval(
    *,
    store: OpenCLIPEmbeddingStore,
    query_modality: EmbeddingModality,
    candidate_modality: EmbeddingModality = "image",
    cutoffs: Iterable[int] = (1, 5, 10),
) -> PairedRetrievalEvaluation:
    """Evalúa recuperación emparejada entre dos modalidades."""

    return _evaluate_aligned_matrices(
        store=store,
        query_embeddings=store.matrix(
            query_modality
        ),
        candidate_embeddings=store.matrix(
            candidate_modality
        ),
        query_modality=query_modality,
        candidate_modality=candidate_modality,
        cutoffs=cutoffs,
    )


def build_fused_text_matrix(
    *,
    store: OpenCLIPEmbeddingStore,
    alpha: float,
) -> np.ndarray:
    """Construye consultas fusionadas normalizadas."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "alpha debe estar comprendido entre 0 y 1."
        )

    fused_rows = [
        fuse_text_embeddings(
            text_visual_embedding=(
                store.text_visual_embeddings[
                    row_index
                ]
            ),
            text_metadata_embedding=(
                store.text_metadata_embeddings[
                    row_index
                ]
            ),
            alpha=alpha,
        )
        for row_index in range(
            store.total_records
        )
    ]

    return np.vstack(
        fused_rows
    ).astype(
        np.float32,
        copy=False,
    )


def evaluate_fused_text_retrieval(
    *,
    store: OpenCLIPEmbeddingStore,
    alpha: float,
    candidate_modality: EmbeddingModality = "image",
    cutoffs: Iterable[int] = (1, 5, 10),
) -> PairedRetrievalEvaluation:
    """Evalúa texto fusionado contra candidatos emparejados."""

    fused_queries = build_fused_text_matrix(
        store=store,
        alpha=alpha,
    )

    return _evaluate_aligned_matrices(
        store=store,
        query_embeddings=fused_queries,
        candidate_embeddings=store.matrix(
            candidate_modality
        ),
        query_modality="text_fused",
        candidate_modality=candidate_modality,
        cutoffs=cutoffs,
        alpha=alpha,
    )


def evaluate_alpha_grid(
    *,
    store: OpenCLIPEmbeddingStore,
    alphas: Iterable[float],
    candidate_modality: EmbeddingModality = "image",
    cutoffs: Iterable[int] = (1, 5, 10),
) -> pd.DataFrame:
    """Evalúa una cuadrícula de pesos de fusión textual."""

    normalized_alphas = tuple(
        float(alpha)
        for alpha in alphas
    )

    if not normalized_alphas:
        raise ValueError(
            "alphas debe contener al menos un valor."
        )

    summaries = []

    for alpha in normalized_alphas:
        evaluation = (
            evaluate_fused_text_retrieval(
                store=store,
                alpha=alpha,
                candidate_modality=(
                    candidate_modality
                ),
                cutoffs=cutoffs,
            )
        )

        summaries.append(
            evaluation.summary
        )

    return pd.concat(
        summaries,
        ignore_index=True,
    )

def build_random_paired_baseline(
    *,
    candidate_count: int,
    cutoffs: Iterable[int] = (1, 5, 10),
) -> pd.DataFrame:
    """Calcula la expectativa teórica de un ranking aleatorio."""

    if candidate_count <= 0:
        raise ValueError(
            "candidate_count debe ser mayor que cero."
        )

    normalized_cutoffs = _validate_cutoffs(
        cutoffs,
        total_records=candidate_count,
    )

    ranks = np.arange(
        1,
        candidate_count + 1,
        dtype=np.float64,
    )

    record: dict[str, object] = {
        "scope": "RANDOM",
        "query_modality": "random",
        "candidate_modality": "image",
        "alpha": np.nan,
        "total_queries": candidate_count,
        "mrr": float(
            np.mean(
                1.0 / ranks
            )
        ),
        "mean_rank": float(
            np.mean(ranks)
        ),
        "median_rank": float(
            np.median(ranks)
        ),
        "mean_matched_score": np.nan,
    }

    for cutoff in normalized_cutoffs:
        record[
            f"recall_at_{cutoff}"
        ] = float(
            cutoff / candidate_count
        )

    return pd.DataFrame.from_records(
        [record]
    )