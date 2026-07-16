"""Pruebas del núcleo de recuperación OpenCLIP."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.openclip_baseline.retrieval import (
    OpenCLIPEmbeddingStore,
    compute_similarity_scores,
    fuse_text_embeddings,
    load_openclip_embedding_store,
    normalize_embedding,
    rank_similarity_scores,
    retrieve_by_index,
    retrieve_fused_text_by_index,
)


def _build_store() -> OpenCLIPEmbeddingStore:
    """Construye un conjunto sintético normalizado."""

    diagonal = np.float32(
        1.0 / np.sqrt(2.0)
    )

    return OpenCLIPEmbeddingStore(
        item_ids=np.asarray(
            ["item-a", "item-b", "item-c"],
            dtype=np.str_,
        ),
        museums=np.asarray(
            ["CMA", "MET", "MET"],
            dtype=np.str_,
        ),
        image_embeddings=np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [diagonal, diagonal],
            ],
            dtype=np.float32,
        ),
        text_visual_embeddings=np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [diagonal, diagonal],
            ],
            dtype=np.float32,
        ),
        text_metadata_embeddings=np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )


def test_load_openclip_embedding_store(
    tmp_path: Path,
) -> None:
    """El cargador debe reconstruir las matrices del NPZ."""

    store = _build_store()
    path = tmp_path / "embeddings.npz"

    np.savez(
        path,
        item_ids=store.item_ids,
        museums=store.museums,
        image_embeddings=store.image_embeddings,
        text_visual_embeddings=(
            store.text_visual_embeddings
        ),
        text_metadata_embeddings=(
            store.text_metadata_embeddings
        ),
    )

    loaded = load_openclip_embedding_store(
        path
    )

    assert loaded.total_records == 3
    assert loaded.embedding_dimension == 2

    np.testing.assert_array_equal(
        loaded.item_ids,
        store.item_ids,
    )
    np.testing.assert_allclose(
        loaded.image_embeddings,
        store.image_embeddings,
    )


def test_loader_rejects_missing_arrays(
    tmp_path: Path,
) -> None:
    """El NPZ debe contener todas las modalidades requeridas."""

    path = tmp_path / "incomplete.npz"

    np.savez(
        path,
        item_ids=np.asarray(["item-a"]),
        museums=np.asarray(["CMA"]),
        image_embeddings=np.ones(
            (1, 2),
            dtype=np.float32,
        ),
        text_visual_embeddings=np.ones(
            (1, 2),
            dtype=np.float32,
        ),
    )

    with pytest.raises(
        ValueError,
        match="text_metadata_embeddings",
    ):
        load_openclip_embedding_store(
            path
        )


def test_store_rejects_duplicate_item_ids() -> None:
    """Los identificadores deben mantener relación uno a uno."""

    store = _build_store()

    with pytest.raises(
        ValueError,
        match="item_ids deben ser únicos",
    ):
        OpenCLIPEmbeddingStore(
            item_ids=np.asarray(
                ["item-a", "item-a", "item-c"],
            ),
            museums=store.museums,
            image_embeddings=(
                store.image_embeddings
            ),
            text_visual_embeddings=(
                store.text_visual_embeddings
            ),
            text_metadata_embeddings=(
                store.text_metadata_embeddings
            ),
        )


def test_compute_similarity_scores() -> None:
    """El producto punto debe coincidir con el coseno normalizado."""

    store = _build_store()

    scores = compute_similarity_scores(
        query_embedding=np.asarray(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        candidate_embeddings=(
            store.image_embeddings
        ),
    )

    expected = np.asarray(
        [
            1.0,
            0.0,
            1.0 / np.sqrt(2.0),
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        scores,
        expected,
        atol=1e-6,
    )


def test_ranking_uses_row_index_to_break_ties() -> None:
    """Los empates deben resolverse con orden estable."""

    store = _build_store()

    ranking = rank_similarity_scores(
        store=store,
        scores=np.asarray(
            [0.5, 0.9, 0.9],
            dtype=np.float32,
        ),
        top_k=3,
    )

    assert ranking["row_index"].tolist() == [
        1,
        2,
        0,
    ]
    assert ranking["rank"].tolist() == [
        1,
        2,
        3,
    ]


def test_retrieve_by_index_excludes_same_item() -> None:
    """La recuperación imagen-imagen debe excluir la consulta."""

    store = _build_store()

    ranking = retrieve_by_index(
        store=store,
        query_index=0,
        query_modality="image",
        candidate_modality="image",
        top_k=2,
        exclude_same_item=True,
    )

    assert ranking["item_id"].tolist() == [
        "item-c",
        "item-b",
    ]
    assert "item-a" not in set(
        ranking["item_id"]
    )


def test_retrieve_by_index_can_keep_same_item() -> None:
    """La recuperación cruzada puede conservar el mismo objeto."""

    store = _build_store()

    ranking = retrieve_by_index(
        store=store,
        query_index=0,
        query_modality="text_metadata",
        candidate_modality="image",
        top_k=1,
        exclude_same_item=False,
    )

    assert ranking.iloc[0]["item_id"] == "item-a"
    assert ranking.iloc[0]["score"] == pytest.approx(
        1.0
    )


def test_retrieval_rejects_invalid_arguments() -> None:
    """Los índices y tamaños de ranking deben ser válidos."""

    store = _build_store()

    with pytest.raises(
        IndexError,
        match="query_index",
    ):
        retrieve_by_index(
            store=store,
            query_index=10,
            query_modality="image",
        )

    with pytest.raises(
        ValueError,
        match="top_k",
    ):
        rank_similarity_scores(
            store=store,
            scores=np.ones(
                store.total_records,
                dtype=np.float32,
            ),
            top_k=0,
        )

def test_normalize_embedding() -> None:
    """La normalización debe producir norma L2 unitaria."""

    normalized = normalize_embedding(
        np.asarray(
            [3.0, 4.0],
            dtype=np.float32,
        )
    )

    np.testing.assert_allclose(
        normalized,
        np.asarray(
            [0.6, 0.8],
            dtype=np.float32,
        ),
        atol=1e-6,
    )

    assert np.linalg.norm(
        normalized
    ) == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_normalize_embedding_rejects_zero_vector() -> None:
    """Un vector nulo no posee dirección normalizable."""

    with pytest.raises(
        ValueError,
        match="norma cero",
    ):
        normalize_embedding(
            np.zeros(
                2,
                dtype=np.float32,
            )
        )


def test_fuse_text_embeddings_at_midpoint() -> None:
    """La fusión equilibrada debe combinar ambas direcciones."""

    fused = fuse_text_embeddings(
        text_visual_embedding=np.asarray(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        text_metadata_embedding=np.asarray(
            [0.0, 1.0],
            dtype=np.float32,
        ),
        alpha=0.5,
    )

    diagonal = np.float32(
        1.0 / np.sqrt(2.0)
    )

    np.testing.assert_allclose(
        fused,
        np.asarray(
            [diagonal, diagonal],
            dtype=np.float32,
        ),
        atol=1e-6,
    )


def test_fuse_text_embeddings_rejects_invalid_alpha() -> None:
    """El peso de fusión debe pertenecer al intervalo unitario."""

    with pytest.raises(
        ValueError,
        match="alpha",
    ):
        fuse_text_embeddings(
            text_visual_embedding=np.asarray(
                [1.0, 0.0],
                dtype=np.float32,
            ),
            text_metadata_embedding=np.asarray(
                [0.0, 1.0],
                dtype=np.float32,
            ),
            alpha=1.2,
        )


def test_retrieve_fused_text_by_index() -> None:
    """La consulta fusionada debe recuperar el candidato diagonal."""

    store = _build_store()

    ranking = retrieve_fused_text_by_index(
        store=store,
        query_index=1,
        alpha=0.5,
        candidate_modality="image",
        top_k=2,
        exclude_same_item=True,
    )

    assert ranking.iloc[0]["item_id"] == "item-c"
    assert ranking.iloc[0]["score"] == pytest.approx(
        1.0,
        abs=1e-6,
    )