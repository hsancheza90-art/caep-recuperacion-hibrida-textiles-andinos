"""Pruebas para la CLI de materialización OpenCLIP."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.openclip_baseline import (
    cli_materialize_embeddings as cli,
)


def test_parser_uses_expected_defaults() -> None:
    """La CLI debe exponer la configuración reproducible por defecto."""

    args = cli.build_parser().parse_args([])

    assert args.model_name == "ViT-B-32"
    assert args.pretrained == "laion2b_s34b_b79k"
    assert args.device == "cuda"
    assert args.precision == "amp_fp16"
    assert args.batch_size == 32
    assert args.normalize is True

    assert (
        args.corpus_path.name
        == "paper_corpus_multimodal_v1.csv"
    )
    assert (
        args.text_inputs_path.name
        == "openclip_text_inputs_v1.csv"
    )
    assert (
        args.output_npz_path.name
        == "openclip_embeddings_v1.npz"
    )


def test_parser_accepts_runtime_overrides() -> None:
    """Los parámetros operativos deben poder sobrescribirse."""

    args = cli.build_parser().parse_args(
        [
            "--device",
            "cpu",
            "--precision",
            "fp32",
            "--batch-size",
            "8",
            "--no-normalize",
        ],
    )

    assert args.device == "cpu"
    assert args.precision == "fp32"
    assert args.batch_size == 8
    assert args.normalize is False


def test_main_rejects_nonpositive_batch_size(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """El tamaño de lote debe ser estrictamente positivo."""

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "--batch-size",
                "0",
            ],
        )

    assert exc_info.value.code == 2

    captured = capsys.readouterr()

    assert (
        "--batch-size debe ser un entero mayor que cero."
        in captured.err
    )


def test_main_rejects_missing_input_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """La CLI debe detenerse cuando falta un archivo de entrada."""

    missing_corpus = tmp_path / "missing_corpus.csv"
    text_inputs = tmp_path / "text_inputs.csv"

    text_inputs.write_text(
        "item_id,text_visual,text_metadata\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "--corpus-path",
                str(missing_corpus),
                "--text-inputs-path",
                str(text_inputs),
            ],
        )

    assert exc_info.value.code == 2

    captured = capsys.readouterr()

    assert "--corpus-path no existe" in captured.err


def test_main_loads_model_and_calls_materializer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La CLI debe transferir configuración y rutas al materializador."""

    corpus_path = tmp_path / "corpus.csv"
    text_inputs_path = tmp_path / "text_inputs.csv"

    corpus_path.write_text(
        "item_id,museum\nitem-1,CMA\n",
        encoding="utf-8",
    )
    text_inputs_path.write_text(
        (
            "item_id,text_visual,text_metadata\n"
            "item-1,texto visual,metadatos\n"
        ),
        encoding="utf-8",
    )

    output_npz_path = tmp_path / "derived" / "embeddings.npz"
    output_index_path = tmp_path / "derived" / "index.csv"
    summary_path = tmp_path / "reports" / "summary.csv"
    provenance_path = tmp_path / "reports" / "provenance.json"

    calls: dict[str, object] = {}

    def fake_load_openclip_components(
        config: cli.OpenCLIPEmbeddingConfig,
    ) -> tuple[str, str, str]:
        calls["load_config"] = config

        return (
            "model",
            "preprocess",
            "tokenizer",
        )

    def fake_materialize_openclip_embeddings(
        **kwargs: object,
    ) -> None:
        calls.update(kwargs)

    monkeypatch.setattr(
        cli,
        "_load_openclip_components",
        fake_load_openclip_components,
    )
    monkeypatch.setattr(
        cli,
        "materialize_openclip_embeddings",
        fake_materialize_openclip_embeddings,
    )

    exit_code = cli.main(
        [
            "--repository-root",
            str(tmp_path),
            "--corpus-path",
            str(corpus_path),
            "--text-inputs-path",
            str(text_inputs_path),
            "--output-npz-path",
            str(output_npz_path),
            "--output-index-path",
            str(output_index_path),
            "--summary-path",
            str(summary_path),
            "--provenance-path",
            str(provenance_path),
            "--device",
            "cpu",
            "--precision",
            "fp32",
            "--batch-size",
            "4",
            "--no-normalize",
        ],
    )

    assert exit_code == 0

    config = calls["config"]

    assert isinstance(
        config,
        cli.OpenCLIPEmbeddingConfig,
    )
    assert config.device == "cpu"
    assert config.precision == "fp32"
    assert config.batch_size == 4
    assert config.normalize is False

    assert calls["model"] == "model"
    assert calls["preprocess"] == "preprocess"
    assert calls["tokenizer"] == "tokenizer"

    assert calls["corpus_path"] == corpus_path.resolve()
    assert (
        calls["text_inputs_path"]
        == text_inputs_path.resolve()
    )
    assert (
        calls["output_npz_path"]
        == output_npz_path.resolve()
    )
    assert (
        calls["output_index_path"]
        == output_index_path.resolve()
    )
    assert calls["summary_path"] == summary_path.resolve()
    assert (
        calls["provenance_path"]
        == provenance_path.resolve()
    )

    assert output_npz_path.parent.is_dir()
    assert summary_path.parent.is_dir()