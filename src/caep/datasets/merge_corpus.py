from __future__ import annotations

import pandas as pd

from caep.datasets.build_enriched_corpus import (
    build_enriched_corpus,
    run as run_enriched_corpus,
    validate_enriched_corpus,
)


def build_corpus() -> pd.DataFrame:
    """
    Construye el corpus experimental canónico.

    Esta función se conserva por compatibilidad con código y pruebas
    desarrolladas antes de incorporar el pipeline de recuperación de
    metadata.

    El corpus ya no depende de archivos derivados en data/interim.
    Se reconstruye directamente desde las fuentes curatoriales
    versionadas.
    """
    return build_enriched_corpus()


def validate_combined_corpus(frame: pd.DataFrame) -> None:
    """
    Valida un corpus usando las reglas del corpus enriquecido canónico.
    """
    validate_enriched_corpus(frame)


def run() -> pd.DataFrame:
    """
    Construye y guarda el corpus enriquecido canónico.
    """
    return run_enriched_corpus()


if __name__ == "__main__":
    run()