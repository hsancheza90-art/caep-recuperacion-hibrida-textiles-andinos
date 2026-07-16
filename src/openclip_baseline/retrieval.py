"""Recuperación reproducible sobre embeddings OpenCLIP materializados."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


EmbeddingModality = Literal[
    "image",
    "text_visual",
    "text_metadata",
]

_REQUIRED_ARRAYS = {
    "item_ids",
    "museums",
    "image_embeddings",
    "text_visual_embeddings",
    "text_metadata_embeddings",
}


@dataclass(frozen=True)
class OpenCLIPEmbeddingStore:
    """Conjunto alineado de embeddings OpenCLIP."""

    item_ids: np.ndarray
    museums: np.ndarray
    image_embeddings: np.ndarray
    text_visual_embeddings: np.ndarray
    text_metadata_embeddings: np.ndarray

    def __post_init__(self) -> None:
        """Valida la coherencia estructural del conjunto."""

        _validate_metadata_vector(
            self.item_ids,
            name="item_ids",
        )
        _validate_metadata_vector(
            self.museums,
            name="museums",
        )

        _validate_embedding_matrix(
            self.image_embeddings,
            name="image_embeddings",
        )
        _validate_embedding_matrix(
            self.text_visual_embeddings,
            name="text_visual_embeddings",
        )
        _validate_embedding_matrix(
            self.text_metadata_embeddings,
            name="text_metadata_embeddings",
        )

        total_records = len(self.item_ids)

        if len(self.museums) != total_records:
            raise ValueError(
                "item_ids y museums deben tener la misma longitud."
            )

        matrices = (
            self.image_embeddings,
            self.text_visual_embeddings,
            self.text_metadata_embeddings,
        )

        if any(
            matrix.shape[0] != total_records
            for matrix in matrices
        ):
            raise ValueError(
                "Todas las matrices deben tener una fila "
                "por cada item_id."
            )

        shapes = {
            matrix.shape
            for matrix in matrices
        }

        if len(shapes) != 1:
            raise ValueError(
                "Las tres matrices de embeddings deben "
                "tener la misma forma."
            )

        if len(np.unique(self.item_ids)) != total_records:
            raise ValueError(
                "Los item_ids deben ser únicos."
            )

    @property
    def total_records(self) -> int:
        """Devuelve el número de objetos representados."""

        return len(self.item_ids)

    @property
    def embedding_dimension(self) -> int:
        """Devuelve la dimensión común de los embeddings."""

        return int(self.image_embeddings.shape[1])

    def matrix(
        self,
        modality: EmbeddingModality,
    ) -> np.ndarray:
        """Selecciona la matriz correspondiente a una modalidad."""

        if modality == "image":
            return self.image_embeddings

        if modality == "text_visual":
            return self.text_visual_embeddings

        if modality == "text_metadata":
            return self.text_metadata_embeddings

        raise ValueError(
            f"Modalidad no soportada: {modality}"
        )


def _validate_metadata_vector(
    values: np.ndarray,
    *,
    name: str,
) -> None:
    """Valida un vector unidimensional de metadatos."""

    if values.ndim != 1:
        raise ValueError(
            f"{name} debe ser un vector unidimensional."
        )

    if len(values) == 0:
        raise ValueError(
            f"{name} no puede estar vacío."
        )


def _validate_embedding_matrix(
    matrix: np.ndarray,
    *,
    name: str,
) -> None:
    """Valida una matriz bidimensional y numéricamente finita."""

    if matrix.ndim != 2:
        raise ValueError(
            f"{name} debe ser una matriz bidimensional."
        )

    if matrix.shape[0] == 0:
        raise ValueError(
            f"{name} no puede estar vacía."
        )

    if matrix.shape[1] == 0:
        raise ValueError(
            f"{name} debe tener al menos una dimensión."
        )

    if not np.issubdtype(
        matrix.dtype,
        np.number,
    ):
        raise ValueError(
            f"{name} debe contener valores numéricos."
        )

    if not np.isfinite(matrix).all():
        raise ValueError(
            f"{name} contiene NaN o valores infinitos."
        )


def load_openclip_embedding_store(
    path: Path,
) -> OpenCLIPEmbeddingStore:
    """Carga y valida un archivo NPZ de embeddings OpenCLIP."""

    if not path.is_file():
        raise FileNotFoundError(
            f"No existe el archivo de embeddings: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        available_arrays = set(data.files)
        missing_arrays = (
            _REQUIRED_ARRAYS - available_arrays
        )

        if missing_arrays:
            missing = ", ".join(
                sorted(missing_arrays)
            )

            raise ValueError(
                "El archivo NPZ no contiene todos los "
                f"arreglos requeridos. Faltan: {missing}"
            )

        item_ids = np.asarray(
            data["item_ids"],
            dtype=np.str_,
        ).copy()

        museums = np.asarray(
            data["museums"],
            dtype=np.str_,
        ).copy()

        image_embeddings = np.asarray(
            data["image_embeddings"],
            dtype=np.float32,
        ).copy()

        text_visual_embeddings = np.asarray(
            data["text_visual_embeddings"],
            dtype=np.float32,
        ).copy()

        text_metadata_embeddings = np.asarray(
            data["text_metadata_embeddings"],
            dtype=np.float32,
        ).copy()

    return OpenCLIPEmbeddingStore(
        item_ids=item_ids,
        museums=museums,
        image_embeddings=image_embeddings,
        text_visual_embeddings=(
            text_visual_embeddings
        ),
        text_metadata_embeddings=(
            text_metadata_embeddings
        ),
    )


def compute_similarity_scores(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
) -> np.ndarray:
    """Calcula productos punto entre una consulta y candidatos."""

    query = np.asarray(
        query_embedding,
        dtype=np.float32,
    )
    candidates = np.asarray(
        candidate_embeddings,
        dtype=np.float32,
    )

    if query.ndim != 1:
        raise ValueError(
            "query_embedding debe ser un vector."
        )

    if candidates.ndim != 2:
        raise ValueError(
            "candidate_embeddings debe ser una matriz."
        )

    if candidates.shape[1] != query.shape[0]:
        raise ValueError(
            "La consulta y los candidatos tienen "
            "dimensiones incompatibles."
        )

    if not np.isfinite(query).all():
        raise ValueError(
            "query_embedding contiene valores no finitos."
        )

    if not np.isfinite(candidates).all():
        raise ValueError(
            "candidate_embeddings contiene valores "
            "no finitos."
        )

    return candidates @ query

def normalize_embedding(
    embedding: np.ndarray,
) -> np.ndarray:
    """Normaliza un embedding individual mediante norma L2."""

    vector = np.asarray(
        embedding,
        dtype=np.float32,
    )

    if vector.ndim != 1:
        raise ValueError(
            "embedding debe ser un vector."
        )

    if not np.isfinite(vector).all():
        raise ValueError(
            "embedding contiene valores no finitos."
        )

    norm = float(
        np.linalg.norm(vector)
    )

    if norm <= 0.0:
        raise ValueError(
            "No se puede normalizar un vector de norma cero."
        )

    return np.asarray(
        vector / norm,
        dtype=np.float32,
    )


def fuse_text_embeddings(
    *,
    text_visual_embedding: np.ndarray,
    text_metadata_embedding: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Fusiona dos embeddings textuales y normaliza el resultado."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "alpha debe estar comprendido entre 0 y 1."
        )

    visual = np.asarray(
        text_visual_embedding,
        dtype=np.float32,
    )
    metadata = np.asarray(
        text_metadata_embedding,
        dtype=np.float32,
    )

    if visual.ndim != 1:
        raise ValueError(
            "text_visual_embedding debe ser un vector."
        )

    if metadata.ndim != 1:
        raise ValueError(
            "text_metadata_embedding debe ser un vector."
        )

    if visual.shape != metadata.shape:
        raise ValueError(
            "Los embeddings textuales deben tener "
            "la misma dimensión."
        )

    if not np.isfinite(visual).all():
        raise ValueError(
            "text_visual_embedding contiene valores "
            "no finitos."
        )

    if not np.isfinite(metadata).all():
        raise ValueError(
            "text_metadata_embedding contiene valores "
            "no finitos."
        )

    fused = (
        np.float32(alpha) * visual
        + np.float32(1.0 - alpha) * metadata
    )

    return normalize_embedding(
        fused
    )


def retrieve_fused_text_by_index(
    *,
    store: OpenCLIPEmbeddingStore,
    query_index: int,
    alpha: float = 0.5,
    candidate_modality: EmbeddingModality = "image",
    top_k: int = 10,
    exclude_same_item: bool = True,
) -> pd.DataFrame:
    """Recupera candidatos usando una consulta textual fusionada."""

    if not 0 <= query_index < store.total_records:
        raise IndexError(
            "query_index está fuera de rango."
        )

    query_embedding = fuse_text_embeddings(
        text_visual_embedding=(
            store.text_visual_embeddings[
                query_index
            ]
        ),
        text_metadata_embedding=(
            store.text_metadata_embeddings[
                query_index
            ]
        ),
        alpha=alpha,
    )

    candidate_embeddings = store.matrix(
        candidate_modality
    )

    scores = compute_similarity_scores(
        query_embedding=query_embedding,
        candidate_embeddings=candidate_embeddings,
    )

    exclude_index = (
        query_index
        if exclude_same_item
        else None
    )

    return rank_similarity_scores(
        store=store,
        scores=scores,
        top_k=top_k,
        exclude_index=exclude_index,
    )

def rank_similarity_scores(
    *,
    store: OpenCLIPEmbeddingStore,
    scores: np.ndarray,
    top_k: int,
    exclude_index: int | None = None,
) -> pd.DataFrame:
    """Ordena puntuaciones con desempate estable por fila."""

    score_vector = np.asarray(
        scores,
        dtype=np.float64,
    )

    if score_vector.ndim != 1:
        raise ValueError(
            "scores debe ser un vector."
        )

    if len(score_vector) != store.total_records:
        raise ValueError(
            "scores debe contener una puntuación "
            "por cada registro."
        )

    if not np.isfinite(score_vector).all():
        raise ValueError(
            "scores contiene valores no finitos."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k debe ser mayor que cero."
        )

    available = np.ones(
        store.total_records,
        dtype=bool,
    )

    if exclude_index is not None:
        if not 0 <= exclude_index < store.total_records:
            raise IndexError(
                "exclude_index está fuera de rango."
            )

        available[exclude_index] = False

    candidate_indices = np.flatnonzero(
        available
    )
    candidate_scores = score_vector[
        candidate_indices
    ]

    order = np.lexsort(
        (
            candidate_indices,
            -candidate_scores,
        ),
    )

    selected_indices = candidate_indices[
        order[:top_k]
    ]

    selected_scores = score_vector[
        selected_indices
    ]

    return pd.DataFrame(
        {
            "rank": np.arange(
                1,
                len(selected_indices) + 1,
                dtype=np.int64,
            ),
            "row_index": selected_indices,
            "item_id": store.item_ids[
                selected_indices
            ],
            "museum": store.museums[
                selected_indices
            ],
            "score": selected_scores,
        },
    )


def retrieve_by_index(
    *,
    store: OpenCLIPEmbeddingStore,
    query_index: int,
    query_modality: EmbeddingModality,
    candidate_modality: EmbeddingModality = "image",
    top_k: int = 10,
    exclude_same_item: bool = True,
) -> pd.DataFrame:
    """Recupera los objetos más próximos a una fila del corpus."""

    if not 0 <= query_index < store.total_records:
        raise IndexError(
            "query_index está fuera de rango."
        )

    query_matrix = store.matrix(
        query_modality
    )
    candidate_matrix = store.matrix(
        candidate_modality
    )

    scores = compute_similarity_scores(
        query_embedding=query_matrix[
            query_index
        ],
        candidate_embeddings=candidate_matrix,
    )

    exclude_index = (
        query_index
        if exclude_same_item
        else None
    )

    return rank_similarity_scores(
        store=store,
        scores=scores,
        top_k=top_k,
        exclude_index=exclude_index,
    )