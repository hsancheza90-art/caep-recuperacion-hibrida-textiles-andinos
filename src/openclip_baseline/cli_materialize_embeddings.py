"""CLI para materializar embeddings OpenCLIP reproducibles."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from src.openclip_baseline.materialize_embeddings import (
    OpenCLIPEmbeddingConfig,
    materialize_openclip_embeddings,
)


def _default_repository_root() -> Path:
    """Obtiene la raíz del repositorio desde este módulo."""

    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos de la CLI."""

    repository_root = _default_repository_root()
    defaults = OpenCLIPEmbeddingConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Extrae y materializa embeddings OpenCLIP de imagen, "
            "texto visual y metadatos."
        ),
    )

    parser.add_argument(
        "--repository-root",
        type=Path,
        default=repository_root,
        help="Raíz del repositorio.",
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=(
            repository_root
            / "data"
            / "derived"
            / "paper_corpus_multimodal_v1.csv"
        ),
        help="Corpus multimodal derivado en formato CSV.",
    )
    parser.add_argument(
        "--text-inputs-path",
        type=Path,
        default=(
            repository_root
            / "data"
            / "derived"
            / "openclip_text_inputs_v1.csv"
        ),
        help="Entradas textuales materializadas para OpenCLIP.",
    )
    parser.add_argument(
        "--output-npz-path",
        type=Path,
        default=(
            repository_root
            / "data"
            / "derived"
            / "openclip_embeddings_v1.npz"
        ),
        help="Archivo NPZ de embeddings.",
    )
    parser.add_argument(
        "--output-index-path",
        type=Path,
        default=(
            repository_root
            / "data"
            / "derived"
            / "openclip_embeddings_index_v1.csv"
        ),
        help="Índice que relaciona filas, objetos y museos.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=(
            repository_root
            / "outputs"
            / "reports"
            / "openclip_embeddings_summary_v1.csv"
        ),
        help="Resumen estadístico de los embeddings.",
    )
    parser.add_argument(
        "--provenance-path",
        type=Path,
        default=(
            repository_root
            / "outputs"
            / "reports"
            / "openclip_embeddings_provenance_v1.json"
        ),
        help="Archivo JSON de procedencia.",
    )
    parser.add_argument(
        "--model-name",
        default=defaults.model_name,
        help="Arquitectura OpenCLIP.",
    )
    parser.add_argument(
        "--pretrained",
        default=defaults.pretrained,
        help="Pesos preentrenados OpenCLIP.",
    )
    parser.add_argument(
        "--device",
        default=defaults.device,
        help="Dispositivo de ejecución, por ejemplo cuda o cpu.",
    )
    parser.add_argument(
        "--precision",
        default=defaults.precision,
        help="Precisión utilizada por OpenCLIP.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=defaults.batch_size,
        help="Número de registros procesados por lote.",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=defaults.normalize,
        help="Normaliza los embeddings a norma L2 unitaria.",
    )

    return parser


def _load_openclip_components(
    config: OpenCLIPEmbeddingConfig,
) -> tuple[Any, Any, Any]:
    """Carga modelo, preprocesamiento y tokenizador OpenCLIP."""

    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError(
            "No se pudo importar open_clip. "
            "Verifica la instalación de open-clip-torch."
        ) from exc

    model, _, preprocess = open_clip.create_model_and_transforms(
        config.model_name,
        pretrained=config.pretrained,
        device=config.device,
        precision=config.precision,
    )

    tokenizer = open_clip.get_tokenizer(
        config.model_name,
    )

    model.eval()

    return model, preprocess, tokenizer


def _require_positive_batch_size(
    parser: argparse.ArgumentParser,
    batch_size: int,
) -> None:
    """Valida que el tamaño de lote sea positivo."""

    if batch_size <= 0:
        parser.error(
            "--batch-size debe ser un entero mayor que cero."
        )


def _require_input_file(
    parser: argparse.ArgumentParser,
    path: Path,
    argument_name: str,
) -> None:
    """Valida la existencia de un archivo de entrada."""

    if not path.is_file():
        parser.error(
            f"{argument_name} no existe o no es un archivo: {path}"
        )


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Ejecuta la materialización de embeddings."""

    parser = build_parser()
    args = parser.parse_args(argv)

    repository_root = args.repository_root.resolve()
    corpus_path = args.corpus_path.resolve()
    text_inputs_path = args.text_inputs_path.resolve()
    output_npz_path = args.output_npz_path.resolve()
    output_index_path = args.output_index_path.resolve()
    summary_path = args.summary_path.resolve()
    provenance_path = args.provenance_path.resolve()

    _require_positive_batch_size(
        parser,
        args.batch_size,
    )
    _require_input_file(
        parser,
        corpus_path,
        "--corpus-path",
    )
    _require_input_file(
        parser,
        text_inputs_path,
        "--text-inputs-path",
    )

    for destination in (
        output_npz_path,
        output_index_path,
        summary_path,
        provenance_path,
    ):
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    config = OpenCLIPEmbeddingConfig(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device,
        precision=args.precision,
        batch_size=args.batch_size,
        normalize=args.normalize,
    )

    print()
    print("MATERIALIZACIÓN DE EMBEDDINGS OPENCLIP")
    print("=" * 80)
    print(f"Modelo:           {config.model_name}")
    print(f"Pesos:            {config.pretrained}")
    print(f"Dispositivo:      {config.device}")
    print(f"Precisión:        {config.precision}")
    print(f"Batch size:       {config.batch_size}")
    print(f"Normalización:    {config.normalize}")
    print(f"Corpus:           {corpus_path}")
    print(f"Entradas texto:   {text_inputs_path}")
    print("=" * 80)

    model, preprocess, tokenizer = (
        _load_openclip_components(config)
    )

    materialize_openclip_embeddings(
        corpus_path=corpus_path,
        text_inputs_path=text_inputs_path,
        output_npz_path=output_npz_path,
        output_index_path=output_index_path,
        summary_path=summary_path,
        provenance_path=provenance_path,
        repository_root=repository_root,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        config=config,
    )

    print()
    print("ARTEFACTOS GENERADOS")
    print("=" * 80)
    print(output_npz_path)
    print(output_index_path)
    print(summary_path)
    print(provenance_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())