"""Pruebas de materialización de embeddings OpenCLIP."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.openclip_baseline.embeddings import (
    OpenCLIPEmbeddingConfig,
)
from src.openclip_baseline.materialize_embeddings import (
    DATASET_NAME,
    PIPELINE_VERSION,
    materialize_openclip_embeddings,
)


class FakeModel:
    """Modelo determinista para pruebas sin descargar OpenCLIP."""

    def eval(self) -> FakeModel:
        return self

    def encode_image(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
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
        return tokens[:, :4].float()


def fake_preprocess(
    image: Image.Image,
) -> torch.Tensor:
    """Convierte una imagen RGB en tensor."""

    array = np.asarray(
        image.convert("RGB"),
        dtype=np.float32,
    )

    return (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        / 255.0
    )


def fake_tokenizer(
    texts: list[str],
) -> torch.Tensor:
    """Tokenizador determinista de cuatro dimensiones."""

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


def make_config() -> OpenCLIPEmbeddingConfig:
    """Configuración reproducible de prueba."""

    return OpenCLIPEmbeddingConfig(
        model_name="fake-vit",
        pretrained="fake-weights",
        device="cpu",
        precision="fp32",
        batch_size=2,
        normalize=True,
    )


def make_paths(
    root: Path,
) -> dict[str, Path]:
    """Define las rutas de prueba."""

    return {
        "corpus": (
            root
            / "data"
            / "derived"
            / "paper_corpus_multimodal_v1.csv"
        ),
        "text_inputs": (
            root
            / "data"
            / "derived"
            / "openclip_text_inputs_v1.csv"
        ),
        "output_npz": (
            root
            / "data"
            / "derived"
            / "openclip_embeddings_v1.npz"
        ),
        "output_index": (
            root
            / "data"
            / "derived"
            / "openclip_embeddings_index_v1.csv"
        ),
        "summary": (
            root
            / "outputs"
            / "reports"
            / "openclip_embeddings_summary_v1.csv"
        ),
        "provenance": (
            root
            / "outputs"
            / "reports"
            / "openclip_embeddings_provenance_v1.json"
        ),
    }


def write_sources(
    root: Path,
) -> dict[str, Path]:
    """Crea imágenes y tablas fuente."""

    paths = make_paths(root)

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

    paths["corpus"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        corpus_rows
    ).to_csv(
        paths["corpus"],
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    pd.DataFrame(
        text_rows
    ).to_csv(
        paths["text_inputs"],
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    return paths


def run_materialization(
    root: Path,
    paths: dict[str, Path],
) -> None:
    """Ejecuta la materialización con componentes falsos."""

    materialize_openclip_embeddings(
        corpus_path=paths["corpus"],
        text_inputs_path=paths["text_inputs"],
        output_npz_path=paths["output_npz"],
        output_index_path=paths["output_index"],
        summary_path=paths["summary"],
        provenance_path=paths["provenance"],
        repository_root=root,
        model=FakeModel(),
        preprocess=fake_preprocess,
        tokenizer=fake_tokenizer,
        config=make_config(),
    )


def test_materialization_writes_all_artifacts(
    tmp_path: Path,
) -> None:
    paths = write_sources(tmp_path)

    run_materialization(
        tmp_path,
        paths,
    )

    assert paths["output_npz"].is_file()
    assert paths["output_index"].is_file()
    assert paths["summary"].is_file()
    assert paths["provenance"].is_file()


def test_materialization_preserves_sources(
    tmp_path: Path,
) -> None:
    paths = write_sources(tmp_path)

    corpus_before = paths["corpus"].read_bytes()
    texts_before = paths["text_inputs"].read_bytes()

    run_materialization(
        tmp_path,
        paths,
    )

    assert paths["corpus"].read_bytes() == corpus_before
    assert paths["text_inputs"].read_bytes() == texts_before


def test_npz_contains_ordered_embeddings(
    tmp_path: Path,
) -> None:
    paths = write_sources(tmp_path)

    run_materialization(
        tmp_path,
        paths,
    )

    with np.load(
        paths["output_npz"],
        allow_pickle=False,
    ) as archive:
        assert set(archive.files) == {
            "item_ids",
            "museums",
            "image_embeddings",
            "text_visual_embeddings",
            "text_metadata_embeddings",
        }

        assert archive["item_ids"].tolist() == [
            "MET:2",
            "CMA:1",
            "MET:3",
        ]

        assert archive["museums"].tolist() == [
            "MET",
            "CMA",
            "MET",
        ]

        for name in [
            "image_embeddings",
            "text_visual_embeddings",
            "text_metadata_embeddings",
        ]:
            matrix = archive[name]

            assert matrix.shape == (3, 4)
            assert matrix.dtype == np.float32

            np.testing.assert_allclose(
                np.linalg.norm(
                    matrix,
                    axis=1,
                ),
                np.ones(3),
                atol=1e-6,
            )


def test_index_matches_npz_row_order(
    tmp_path: Path,
) -> None:
    paths = write_sources(tmp_path)

    run_materialization(
        tmp_path,
        paths,
    )

    index = pd.read_csv(
        paths["output_index"],
        keep_default_na=False,
    )

    assert index.columns.tolist() == [
        "row_index",
        "item_id",
        "museum",
    ]

    assert index["row_index"].tolist() == [
        0,
        1,
        2,
    ]

    assert index["item_id"].tolist() == [
        "MET:2",
        "CMA:1",
        "MET:3",
    ]


def test_summary_describes_three_modalities(
    tmp_path: Path,
) -> None:
    paths = write_sources(tmp_path)

    run_materialization(
        tmp_path,
        paths,
    )

    summary = pd.read_csv(
        paths["summary"],
        keep_default_na=False,
    )

    assert summary["modality"].tolist() == [
        "image",
        "text_visual",
        "text_metadata",
    ]

    assert summary["total_records"].tolist() == [
        3,
        3,
        3,
    ]

    assert summary["embedding_dimension"].tolist() == [
        4,
        4,
        4,
    ]

    assert summary["dtype"].tolist() == [
        "float32",
        "float32",
        "float32",
    ]

    assert summary["finite_values"].all()
    assert summary["normalized_l2"].all()

    np.testing.assert_allclose(
        summary["mean_l2_norm"],
        np.ones(3),
        atol=1e-6,
    )


def test_provenance_is_portable_and_complete(
    tmp_path: Path,
) -> None:
    paths = write_sources(tmp_path)

    run_materialization(
        tmp_path,
        paths,
    )

    provenance = json.loads(
        paths["provenance"].read_text(
            encoding="utf-8"
        )
    )

    assert provenance["dataset_name"] == DATASET_NAME
    assert (
        provenance["pipeline_version"]
        == PIPELINE_VERSION
    )

    assert provenance["coverage"] == {
        "corpus_records": 3,
        "text_records": 3,
        "output_records": 3,
    }

    assert provenance["model"] == {
        "model_name": "fake-vit",
        "pretrained": "fake-weights",
        "device": "cpu",
        "precision": "fp32",
        "batch_size": 2,
        "normalize": True,
        "embedding_dimension": 4,
    }

    assert set(provenance["inputs"]) == {
        "corpus",
        "text_inputs",
    }

    assert set(provenance["outputs"]) == {
        "npz",
        "index",
        "summary",
    }

    metadata_entries = [
        *provenance["inputs"].values(),
        *provenance["outputs"].values(),
    ]

    for metadata in metadata_entries:
        assert not Path(
            metadata["path"]
        ).is_absolute()

        assert "\\" not in metadata["path"]
        assert len(metadata["sha256"]) == 64
        assert metadata["size_bytes"] > 0


def test_materialization_is_byte_deterministic(
    tmp_path: Path,
) -> None:
    paths = write_sources(tmp_path)

    run_materialization(
        tmp_path,
        paths,
    )

    before = {
        name: paths[name].read_bytes()
        for name in [
            "output_npz",
            "output_index",
            "summary",
            "provenance",
        ]
    }

    run_materialization(
        tmp_path,
        paths,
    )

    after = {
        name: paths[name].read_bytes()
        for name in [
            "output_npz",
            "output_index",
            "summary",
            "provenance",
        ]
    }

    assert after == before


def test_materialization_rejects_source_overwrite(
    tmp_path: Path,
) -> None:
    paths = write_sources(tmp_path)

    with pytest.raises(
        ValueError,
        match="sobrescribir.*fuente",
    ):
        materialize_openclip_embeddings(
            corpus_path=paths["corpus"],
            text_inputs_path=paths["text_inputs"],
            output_npz_path=paths["corpus"],
            output_index_path=paths["output_index"],
            summary_path=paths["summary"],
            provenance_path=paths["provenance"],
            repository_root=tmp_path,
            model=FakeModel(),
            preprocess=fake_preprocess,
            tokenizer=fake_tokenizer,
            config=make_config(),
        )


def test_materialization_requires_distinct_outputs(
    tmp_path: Path,
) -> None:
    paths = write_sources(tmp_path)

    with pytest.raises(
        ValueError,
        match="rutas de salida.*distintas",
    ):
        materialize_openclip_embeddings(
            corpus_path=paths["corpus"],
            text_inputs_path=paths["text_inputs"],
            output_npz_path=paths["output_npz"],
            output_index_path=paths["output_npz"],
            summary_path=paths["summary"],
            provenance_path=paths["provenance"],
            repository_root=tmp_path,
            model=FakeModel(),
            preprocess=fake_preprocess,
            tokenizer=fake_tokenizer,
            config=make_config(),
        )