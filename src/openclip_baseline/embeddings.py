"""Extracción reproducible de embeddings OpenCLIP."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as functional


MODEL_NAME = "ViT-B-32"
PRETRAINED_NAME = "laion2b_s34b_b79k"
EMBEDDING_PIPELINE_VERSION = "openclip_embedding_extraction_v1"

VALID_DEVICES = frozenset(
    {
        "cpu",
        "cuda",
    }
)

VALID_PRECISIONS = frozenset(
    {
        "fp32",
        "amp_fp16",
        "amp_bf16",
    }
)

CORPUS_REQUIRED_COLUMNS = (
    "item_id",
    "museum",
    "image_local_path",
)

TEXT_REQUIRED_COLUMNS = (
    "item_id",
    "museum",
    "text_visual",
    "text_metadata",
)


@dataclass(frozen=True)
class OpenCLIPEmbeddingConfig:
    """Configuración reproducible para la extracción de embeddings."""

    model_name: str = MODEL_NAME
    pretrained: str = PRETRAINED_NAME
    device: str = "cuda"
    precision: str = "amp_fp16"
    batch_size: int = 32
    normalize: bool = True


@dataclass(frozen=True)
class EmbeddingExtractionResult:
    """Resultado ordenado de la extracción multimodal."""

    item_ids: tuple[str, ...]
    museums: tuple[str, ...]
    image_embeddings: np.ndarray
    text_visual_embeddings: np.ndarray
    text_metadata_embeddings: np.ndarray
    embedding_dimension: int


def _require_columns(
    frame: pd.DataFrame,
    required_columns: tuple[str, ...],
    frame_name: str,
) -> None:
    """Comprueba las columnas mínimas de una tabla."""

    missing_columns = sorted(
        set(required_columns).difference(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            f"La tabla {frame_name} no contiene las columnas requeridas: "
            f"{missing_columns}"
        )


def _normalize_string_series(
    series: pd.Series,
) -> pd.Series:
    """Convierte una serie a cadenas limpias y no nulas."""

    return (
        series.astype("string")
        .fillna("")
        .str.strip()
    )


def _validate_identifiers(
    frame: pd.DataFrame,
    frame_name: str,
) -> pd.Series:
    """Valida que los identificadores sean completos y únicos."""

    item_ids = _normalize_string_series(
        frame["item_id"]
    )

    if item_ids.eq("").any():
        raise ValueError(
            f"La tabla {frame_name} contiene item_id vacíos."
        )

    duplicated = item_ids.duplicated(
        keep=False
    )

    if duplicated.any():
        examples = (
            item_ids.loc[duplicated]
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"La tabla {frame_name} contiene item_id duplicados: "
            f"{examples}"
        )

    return item_ids


def _validate_config(
    config: OpenCLIPEmbeddingConfig,
) -> None:
    """Valida los parámetros de ejecución del extractor."""

    if (
        not isinstance(config.batch_size, int)
        or isinstance(config.batch_size, bool)
        or config.batch_size <= 0
    ):
        raise ValueError(
            "batch_size debe ser un entero mayor que cero."
        )

    if config.device not in VALID_DEVICES:
        raise ValueError(
            "device debe ser uno de los siguientes valores: "
            f"{sorted(VALID_DEVICES)}"
        )

    if config.precision not in VALID_PRECISIONS:
        raise ValueError(
            "precision debe ser uno de los siguientes valores: "
            f"{sorted(VALID_PRECISIONS)}"
        )

    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "device='cuda' fue solicitado, pero CUDA no está disponible."
        )

    if (
        config.precision == "amp_fp16"
        and config.device != "cuda"
    ):
        raise ValueError(
            "precision='amp_fp16' requiere device='cuda'."
        )

    if not isinstance(config.normalize, bool):
        raise ValueError(
            "normalize debe ser un valor booleano."
        )


def _prepare_records(
    corpus: pd.DataFrame,
    text_inputs: pd.DataFrame,
) -> pd.DataFrame:
    """Une corpus y textos preservando el orden canónico."""

    _require_columns(
        corpus,
        CORPUS_REQUIRED_COLUMNS,
        "corpus",
    )

    _require_columns(
        text_inputs,
        TEXT_REQUIRED_COLUMNS,
        "text_inputs",
    )

    corpus_copy = corpus.copy(
        deep=True
    )

    text_copy = text_inputs.copy(
        deep=True
    )

    corpus_ids = _validate_identifiers(
        corpus_copy,
        "corpus",
    )

    text_ids = _validate_identifiers(
        text_copy,
        "text_inputs",
    )

    corpus_copy["item_id"] = corpus_ids
    text_copy["item_id"] = text_ids

    corpus_id_set = set(
        corpus_ids.tolist()
    )

    text_id_set = set(
        text_ids.tolist()
    )

    if corpus_id_set != text_id_set:
        only_corpus = sorted(
            corpus_id_set.difference(
                text_id_set
            )
        )

        only_text = sorted(
            text_id_set.difference(
                corpus_id_set
            )
        )

        raise ValueError(
            "La cobertura textual no coincide con el corpus. "
            f"Sin texto: {only_corpus[:10]}; "
            f"sin registro de corpus: {only_text[:10]}"
        )

    corpus_copy["_row_order"] = np.arange(
        len(corpus_copy),
        dtype=np.int64,
    )

    text_selection = text_copy[
        [
            "item_id",
            "museum",
            "text_visual",
            "text_metadata",
        ]
    ].rename(
        columns={
            "museum": "text_museum",
        }
    )

    records = corpus_copy[
        [
            "item_id",
            "museum",
            "image_local_path",
            "_row_order",
        ]
    ].merge(
        text_selection,
        on="item_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    records = records.sort_values(
        "_row_order",
        kind="stable",
    ).reset_index(
        drop=True
    )

    corpus_museum = (
        _normalize_string_series(
            records["museum"]
        )
        .str.upper()
    )

    text_museum = (
        _normalize_string_series(
            records["text_museum"]
        )
        .str.upper()
    )

    museum_mismatch = corpus_museum.ne(
        text_museum
    )

    if museum_mismatch.any():
        examples = (
            records.loc[
                museum_mismatch,
                "item_id",
            ]
            .head(10)
            .tolist()
        )

        raise ValueError(
            "Existen museos incompatibles entre el corpus y "
            f"las entradas textuales: {examples}"
        )

    records["museum"] = corpus_museum

    for column in (
        "text_visual",
        "text_metadata",
    ):
        records[column] = _normalize_string_series(
            records[column]
        )

        if records[column].eq("").any():
            examples = (
                records.loc[
                    records[column].eq(""),
                    "item_id",
                ]
                .head(10)
                .tolist()
            )

            raise ValueError(
                "La cobertura textual contiene valores vacíos "
                f"en {column}: {examples}"
            )

    return records[
        [
            "item_id",
            "museum",
            "image_local_path",
            "text_visual",
            "text_metadata",
        ]
    ].copy()


def _resolve_image_path(
    value: object,
    repository_root: Path,
) -> Path:
    """Resuelve y valida una imagen dentro del repositorio."""

    relative_text = str(
        value
        if value is not None
        else ""
    ).strip()

    if not relative_text:
        raise FileNotFoundError(
            "Se encontró una ruta de imagen vacía."
        )

    candidate = Path(
        relative_text
    )

    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (
            repository_root
            / candidate
        ).resolve()

    try:
        resolved.relative_to(
            repository_root
        )
    except ValueError as exc:
        raise ValueError(
            "La ruta de imagen debe permanecer dentro "
            f"del repositorio: {resolved}"
        ) from exc

    if not resolved.is_file():
        raise FileNotFoundError(
            f"No se encontró la imagen requerida: {resolved}"
        )

    return resolved


def _autocast_context(
    config: OpenCLIPEmbeddingConfig,
):
    """Construye el contexto de precisión configurado."""

    if config.precision == "fp32":
        return nullcontext()

    if config.precision == "amp_fp16":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    return torch.autocast(
        device_type=config.device,
        dtype=torch.bfloat16,
    )


def _normalize_features(
    features: torch.Tensor,
    normalize: bool,
) -> torch.Tensor:
    """Convierte embeddings a float32 y aplica normalización L2."""

    output = features.float()

    if normalize:
        output = functional.normalize(
            output,
            p=2,
            dim=-1,
        )

    return output


def _tensor_to_numpy(
    tensor: torch.Tensor,
) -> np.ndarray:
    """Convierte un tensor a matriz NumPy float32 independiente."""

    return (
        tensor.detach()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
        .copy()
    )


def _iter_batch_bounds(
    total_records: int,
    batch_size: int,
):
    """Produce límites de lotes consecutivos."""

    for start in range(
        0,
        total_records,
        batch_size,
    ):
        stop = min(
            start + batch_size,
            total_records,
        )

        yield start, stop


def _extract_image_embeddings(
    *,
    records: pd.DataFrame,
    repository_root: Path,
    model: Any,
    preprocess: Any,
    config: OpenCLIPEmbeddingConfig,
    device: torch.device,
) -> np.ndarray:
    """Extrae embeddings visuales por lotes."""

    batches: list[np.ndarray] = []

    for start, stop in _iter_batch_bounds(
        len(records),
        config.batch_size,
    ):
        image_tensors: list[torch.Tensor] = []

        for value in records.iloc[
            start:stop
        ]["image_local_path"]:
            image_path = _resolve_image_path(
                value,
                repository_root,
            )

            with Image.open(
                image_path
            ) as source_image:
                processed = preprocess(
                    source_image.convert("RGB")
                )

            if not isinstance(
                processed,
                torch.Tensor,
            ):
                raise TypeError(
                    "preprocess debe devolver un tensor de PyTorch."
                )

            image_tensors.append(
                processed
            )

        images = torch.stack(
            image_tensors,
            dim=0,
        ).to(
            device,
            non_blocking=(
                device.type == "cuda"
            ),
        )

        with torch.inference_mode():
            with _autocast_context(config):
                features = model.encode_image(
                    images
                )

        features = _normalize_features(
            features,
            config.normalize,
        )

        batches.append(
            _tensor_to_numpy(
                features
            )
        )

    return np.concatenate(
        batches,
        axis=0,
    )


def _extract_text_embeddings(
    *,
    texts: list[str],
    model: Any,
    tokenizer: Any,
    config: OpenCLIPEmbeddingConfig,
    device: torch.device,
) -> np.ndarray:
    """Extrae embeddings textuales por lotes."""

    batches: list[np.ndarray] = []

    for start, stop in _iter_batch_bounds(
        len(texts),
        config.batch_size,
    ):
        tokens = tokenizer(
            texts[start:stop]
        )

        if not isinstance(
            tokens,
            torch.Tensor,
        ):
            raise TypeError(
                "tokenizer debe devolver un tensor de PyTorch."
            )

        tokens = tokens.to(
            device,
            non_blocking=(
                device.type == "cuda"
            ),
        )

        with torch.inference_mode():
            with _autocast_context(config):
                features = model.encode_text(
                    tokens
                )

        features = _normalize_features(
            features,
            config.normalize,
        )

        batches.append(
            _tensor_to_numpy(
                features
            )
        )

    return np.concatenate(
        batches,
        axis=0,
    )


def _validate_embedding_matrices(
    *,
    expected_records: int,
    image_embeddings: np.ndarray,
    text_visual_embeddings: np.ndarray,
    text_metadata_embeddings: np.ndarray,
) -> int:
    """Comprueba forma, dimensión y tipo de los embeddings."""

    matrices = {
        "image": image_embeddings,
        "text_visual": text_visual_embeddings,
        "text_metadata": text_metadata_embeddings,
    }

    for name, matrix in matrices.items():
        if matrix.ndim != 2:
            raise ValueError(
                f"La matriz {name} debe tener dos dimensiones."
            )

        if matrix.shape[0] != expected_records:
            raise ValueError(
                f"La matriz {name} contiene {matrix.shape[0]} "
                f"registros; se esperaban {expected_records}."
            )

        if matrix.dtype != np.float32:
            raise ValueError(
                f"La matriz {name} debe utilizar float32."
            )

        if not np.isfinite(
            matrix
        ).all():
            raise ValueError(
                f"La matriz {name} contiene valores no finitos."
            )

    dimensions = {
        matrix.shape[1]
        for matrix in matrices.values()
    }

    if len(dimensions) != 1:
        shapes = {
            name: matrix.shape
            for name, matrix in matrices.items()
        }

        raise ValueError(
            "Los embeddings presentan dimensiones incompatibles: "
            f"{shapes}"
        )

    embedding_dimension = dimensions.pop()

    if embedding_dimension <= 0:
        raise ValueError(
            "La dimensión de los embeddings debe ser positiva."
        )

    return int(
        embedding_dimension
    )


def extract_openclip_embeddings(
    *,
    corpus: pd.DataFrame,
    text_inputs: pd.DataFrame,
    repository_root: Path,
    model: Any,
    preprocess: Any,
    tokenizer: Any,
    config: OpenCLIPEmbeddingConfig,
) -> EmbeddingExtractionResult:
    """Extrae embeddings visuales y textuales preservando el orden.

    El extractor conserva el orden del corpus multimodal, exige una
    correspondencia uno a uno mediante ``item_id`` y devuelve matrices
    NumPy ``float32`` listas para calcular similitud coseno.
    """

    _validate_config(
        config
    )

    repository_root = (
        repository_root.resolve()
    )

    records = _prepare_records(
        corpus,
        text_inputs,
    )

    device = torch.device(
        config.device
    )

    evaluation_method = getattr(
        model,
        "eval",
        None,
    )

    if callable(
        evaluation_method
    ):
        evaluation_method()

    image_embeddings = (
        _extract_image_embeddings(
            records=records,
            repository_root=repository_root,
            model=model,
            preprocess=preprocess,
            config=config,
            device=device,
        )
    )

    text_visual_embeddings = (
        _extract_text_embeddings(
            texts=records[
                "text_visual"
            ].tolist(),
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
        )
    )

    text_metadata_embeddings = (
        _extract_text_embeddings(
            texts=records[
                "text_metadata"
            ].tolist(),
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
        )
    )

    embedding_dimension = (
        _validate_embedding_matrices(
            expected_records=len(
                records
            ),
            image_embeddings=image_embeddings,
            text_visual_embeddings=(
                text_visual_embeddings
            ),
            text_metadata_embeddings=(
                text_metadata_embeddings
            ),
        )
    )

    return EmbeddingExtractionResult(
        item_ids=tuple(
            records["item_id"].tolist()
        ),
        museums=tuple(
            records["museum"].tolist()
        ),
        image_embeddings=image_embeddings,
        text_visual_embeddings=(
            text_visual_embeddings
        ),
        text_metadata_embeddings=(
            text_metadata_embeddings
        ),
        embedding_dimension=(
            embedding_dimension
        ),
    )