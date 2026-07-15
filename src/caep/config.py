from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BUILD_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "paper_corpus_build_v1.yaml"
)


def load_build_config(
    path: Path = BUILD_CONFIG_PATH,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe la configuración de construcción: {path}"
        )

    with path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "La configuración de construcción no es un diccionario válido."
        )

    if "dataset" not in config or "build" not in config:
        raise ValueError(
            "La configuración debe contener las secciones "
            "'dataset' y 'build'."
        )

    return config


def get_processing_timestamp() -> str:
    config = load_build_config()
    timestamp = str(
        config["build"].get("processing_timestamp", "")
    ).strip()

    if not timestamp:
        raise ValueError(
            "build.processing_timestamp no puede estar vacío."
        )

    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"processing_timestamp no es ISO 8601 válido: {timestamp}"
        ) from exc

    return timestamp


def get_dataset_version() -> str:
    config = load_build_config()
    version = str(
        config["dataset"].get("version", "")
    ).strip()

    if not version:
        raise ValueError(
            "dataset.version no puede estar vacío."
        )

    return version