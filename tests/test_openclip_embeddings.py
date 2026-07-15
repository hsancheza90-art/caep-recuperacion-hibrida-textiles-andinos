"""Pruebas del extractor reproducible de embeddings OpenCLIP."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.openclip_baseline.embeddings import (
    EmbeddingExtractionResult,
    OpenCLIPEmbeddingConfig,
    extract_openclip_embeddings,
)


class FakeModel:
    """Modelo determinista para probar extracción sin OpenCLIP real."""

    def __init__(self) -> None:
        self.image_batch_sizes: list[int] = []
        self.text_batch_sizes: list[int] = []

    def encode_image(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        self.image_batch_sizes.append(
            int(images.shape[0])
        )

        means = images.mean(
            dim=(2, 3)
        )

        extra = means.mean(
            dim=1,
            keepdim=True,
        )

        return torch.cat(
            [
                means,
                extra,
            ],
            dim=1,
        )

    def encode_text(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        self.text_batch_sizes.append(
            int(tokens.shape[0])
        )

        return tokens[
            :,
            :4,
        ].float()


class MismatchedFakeModel(FakeModel):
    """Modelo falso con dimensiones incompatibles."""

    def encode_text(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        self.text_batch_sizes.append(
            int(tokens.shape[0])
        )

        return tokens[
            :,
            :3,
        ].float()


def fake_preprocess(
    image: Image.Image,
) -> torch.Tensor:
    """Convierte una imagen RGB en tensor simple."""

    array = np.asarray(
        image.convert("RGB"),
        dtype=np.float32,
    )

    return torch.from_numpy(
        array
    ).permute(
        2,
        0,
        1,
    ) / 255.0


def fake_tokenizer(
    texts: list[str],
) -> torch.Tensor:
    """Tokenizador determinista de cuatro valores."""

    rows = []

    for text in texts:
        encoded = text.encode("utf-8")

        rows.append(
            [
                len(text),
                sum(encoded) % 251,
                len(encoded),
                text.count(" "),
            ]
        )

    return torch.tensor(
        rows,
        dtype=torch.long,
    )


def make_inputs(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crea imágenes y tablas mínimas para las pruebas."""

    image_directory = (
        root
        / "data"
        / "images"
        / "test"
    )

    image_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = [
        (
            "MET:2",
            "MET",
            "first.jpg",
            (255, 0, 0),
        ),
        (
            "CMA:1",
            "CMA",
            "second.jpg",
            (0, 255, 0),
        ),
        (
            "MET:3",
            "MET",
            "third.jpg",
            (0, 0, 255),
        ),
    ]

    corpus_rows = []
    text_rows = []

    for index, (
        item_id,
        museum,
        filename,
        color,
    ) in enumerate(rows):
        image_path = (
            image_directory
            / filename
        )

        Image.new(
            "RGB",
            (4, 4),
            color=color,
        ).save(image_path)

        corpus_rows.append(
            {
                "item_id": item_id,
                "museum": museum,
                "image_local_path": (
                    image_path
                    .relative_to(root)
                    .as_posix()
                ),
            }
        )

        text_rows.append(
            {
                "item_id": item_id,
                "museum": museum,
                "text_visual": (
                    f"Visual description {index}"
                ),
                "text_metadata": (
                    f"Metadata description {index}"
                ),
            }
        )

    return (
        pd.DataFrame(corpus_rows),
        pd.DataFrame(text_rows),
    )


def make_config(
    batch_size: int = 2,
) -> OpenCLIPEmbeddingConfig:
    """Crea configuración de prueba sobre CPU."""

    return OpenCLIPEmbeddingConfig(
        device="cpu",
        precision="fp32",
        batch_size=batch_size,
        normalize=True,
    )


def run_extraction(
    *,
    root: Path,
    corpus: pd.DataFrame,
    text_inputs: pd.DataFrame,
    model: FakeModel | None = None,
    batch_size: int = 2,
) -> EmbeddingExtractionResult:
    """Ejecuta el extractor con dependencias falsas."""

    return extract_openclip_embeddings(
        corpus=corpus,
        text_inputs=text_inputs,
        repository_root=root,
        model=(
            model
            if model is not None
            else FakeModel()
        ),
        preprocess=fake_preprocess,
        tokenizer=fake_tokenizer,
        config=make_config(
            batch_size=batch_size
        ),
    )


def test_extraction_preserves_order_and_shapes(
    tmp_path: Path,
) -> None:
    corpus, text_inputs = make_inputs(
        tmp_path
    )

    result = run_extraction(
        root=tmp_path,
        corpus=corpus,
        text_inputs=text_inputs,
    )

    assert result.item_ids == (
        "MET:2",
        "CMA:1",
        "MET:3",
    )

    assert result.museums == (
        "MET",
        "CMA",
        "MET",
    )

    assert result.image_embeddings.shape == (
        3,
        4,
    )

    assert result.text_visual_embeddings.shape == (
        3,
        4,
    )

    assert result.text_metadata_embeddings.shape == (
        3,
        4,
    )

    assert result.embedding_dimension == 4


def test_extraction_returns_float32_normalized_embeddings(
    tmp_path: Path,
) -> None:
    corpus, text_inputs = make_inputs(
        tmp_path
    )

    result = run_extraction(
        root=tmp_path,
        corpus=corpus,
        text_inputs=text_inputs,
    )

    matrices = [
        result.image_embeddings,
        result.text_visual_embeddings,
        result.text_metadata_embeddings,
    ]

    for matrix in matrices:
        assert matrix.dtype == np.float32

        norms = np.linalg.norm(
            matrix,
            axis=1,
        )

        np.testing.assert_allclose(
            norms,
            np.ones(len(matrix)),
            atol=1e-6,
        )


def test_extraction_uses_requested_batch_size(
    tmp_path: Path,
) -> None:
    corpus, text_inputs = make_inputs(
        tmp_path
    )

    model = FakeModel()

    run_extraction(
        root=tmp_path,
        corpus=corpus,
        text_inputs=text_inputs,
        model=model,
        batch_size=2,
    )

    assert model.image_batch_sizes == [
        2,
        1,
    ]

    assert model.text_batch_sizes == [
        2,
        1,
        2,
        1,
    ]


def test_extraction_is_deterministic(
    tmp_path: Path,
) -> None:
    corpus, text_inputs = make_inputs(
        tmp_path
    )

    first = run_extraction(
        root=tmp_path,
        corpus=corpus,
        text_inputs=text_inputs,
    )

    second = run_extraction(
        root=tmp_path,
        corpus=corpus,
        text_inputs=text_inputs,
    )

    np.testing.assert_array_equal(
        first.image_embeddings,
        second.image_embeddings,
    )

    np.testing.assert_array_equal(
        first.text_visual_embeddings,
        second.text_visual_embeddings,
    )

    np.testing.assert_array_equal(
        first.text_metadata_embeddings,
        second.text_metadata_embeddings,
    )


def test_extraction_rejects_missing_text_record(
    tmp_path: Path,
) -> None:
    corpus, text_inputs = make_inputs(
        tmp_path
    )

    text_inputs = text_inputs.iloc[:-1].copy()

    with pytest.raises(
        ValueError,
        match="cobertura.*textual",
    ):
        run_extraction(
            root=tmp_path,
            corpus=corpus,
            text_inputs=text_inputs,
        )


def test_extraction_rejects_duplicate_item_id(
    tmp_path: Path,
) -> None:
    corpus, text_inputs = make_inputs(
        tmp_path
    )

    duplicated = pd.concat(
        [
            text_inputs,
            text_inputs.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicados",
    ):
        run_extraction(
            root=tmp_path,
            corpus=corpus,
            text_inputs=duplicated,
        )


def test_extraction_rejects_missing_image(
    tmp_path: Path,
) -> None:
    corpus, text_inputs = make_inputs(
        tmp_path
    )

    missing_path = (
        tmp_path
        / corpus.loc[
            0,
            "image_local_path",
        ]
    )

    missing_path.unlink()

    with pytest.raises(
        FileNotFoundError,
        match="imagen",
    ):
        run_extraction(
            root=tmp_path,
            corpus=corpus,
            text_inputs=text_inputs,
        )


def test_extraction_rejects_incompatible_dimensions(
    tmp_path: Path,
) -> None:
    corpus, text_inputs = make_inputs(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="dimensiones.*incompatibles",
    ):
        run_extraction(
            root=tmp_path,
            corpus=corpus,
            text_inputs=text_inputs,
            model=MismatchedFakeModel(),
        )


def test_extraction_does_not_modify_inputs(
    tmp_path: Path,
) -> None:
    corpus, text_inputs = make_inputs(
        tmp_path
    )

    corpus_before = corpus.copy(
        deep=True
    )

    texts_before = text_inputs.copy(
        deep=True
    )

    run_extraction(
        root=tmp_path,
        corpus=corpus,
        text_inputs=text_inputs,
    )

    pd.testing.assert_frame_equal(
        corpus,
        corpus_before,
    )

    pd.testing.assert_frame_equal(
        text_inputs,
        texts_before,
    )


@pytest.mark.parametrize(
    (
        "field",
        "value",
        "message",
    ),
    [
        (
            "batch_size",
            0,
            "batch_size",
        ),
        (
            "device",
            "tpu",
            "device",
        ),
        (
            "precision",
            "int8",
            "precision",
        ),
    ],
)
def test_extraction_rejects_invalid_configuration(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    corpus, text_inputs = make_inputs(
        tmp_path
    )

    values = {
        "device": "cpu",
        "precision": "fp32",
        "batch_size": 2,
        "normalize": True,
    }

    values[field] = value

    config = OpenCLIPEmbeddingConfig(
        **values,
    )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        extract_openclip_embeddings(
            corpus=corpus,
            text_inputs=text_inputs,
            repository_root=tmp_path,
            model=FakeModel(),
            preprocess=fake_preprocess,
            tokenizer=fake_tokenizer,
            config=config,
        )