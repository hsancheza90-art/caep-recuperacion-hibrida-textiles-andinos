from __future__ import annotations

from pathlib import Path

import pandas as pd

from caep.adapters.cma_enriched_adapter import (
    adapt_cma_enriched,
)
from caep.adapters.met_enriched_adapter import (
    adapt_met_enriched,
)
from caep.config import (
    get_processing_timestamp,
    load_build_config,
)
from caep.enrichment.recover_cma_metadata import (
    CANDIDATES_PATH as CMA_CANDIDATES_PATH,
    REVIEWED_PATH as CMA_REVIEWED_PATH,
    recover_cma_metadata,
)
from caep.enrichment.recover_met_metadata import (
    INVENTORY_PATH as MET_INVENTORY_PATH,
    REVIEWED_PATH as MET_REVIEWED_PATH,
    recover_met_metadata,
)


OUTPUT_PARQUET = Path(
    "data/processed/paper_corpus_enriched_v1.parquet"
)
OUTPUT_CSV = OUTPUT_PARQUET.with_suffix(".csv")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo fuente requerido: {path}"
        )

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )


def validate_same_schema(
    met: pd.DataFrame,
    cma: pd.DataFrame,
) -> None:
    if list(met.columns) == list(cma.columns):
        return

    met_only = sorted(set(met.columns) - set(cma.columns))
    cma_only = sorted(set(cma.columns) - set(met.columns))

    raise ValueError(
        "Los esquemas enriquecidos no coinciden.\n"
        f"Solo MET: {met_only}\n"
        f"Solo CMA: {cma_only}"
    )


def validate_enriched_corpus(
    corpus: pd.DataFrame,
) -> None:
    config = load_build_config()
    build_config = config["build"]

    expected_rows = int(build_config["expected_rows"])
    expected_columns = int(
        build_config["expected_columns"]
    )
    expected_counts = {
        str(key): int(value)
        for key, value in build_config[
            "expected_museum_counts"
        ].items()
    }

    if len(corpus) != expected_rows:
        raise ValueError(
            f"Se esperaban {expected_rows} registros, "
            f"pero se obtuvieron {len(corpus)}."
        )

    if len(corpus.columns) != expected_columns:
        raise ValueError(
            f"Se esperaban {expected_columns} columnas, "
            f"pero se obtuvieron {len(corpus.columns)}."
        )

    if not corpus["item_id"].is_unique:
        duplicates = corpus.loc[
            corpus["item_id"].duplicated(keep=False),
            ["item_id", "museum", "source_object_id"],
        ]

        raise ValueError(
            "Se encontraron item_id duplicados:\n"
            f"{duplicates.to_string(index=False)}"
        )

    source_duplicates = corpus.duplicated(
        subset=["museum", "source_object_id"],
        keep=False,
    )

    if source_duplicates.any():
        duplicates = corpus.loc[
            source_duplicates,
            ["item_id", "museum", "source_object_id"],
        ]

        raise ValueError(
            "Se encontraron claves institucionales duplicadas:\n"
            f"{duplicates.to_string(index=False)}"
        )

    observed_counts = (
        corpus["museum"].value_counts().to_dict()
    )

    if observed_counts != expected_counts:
        raise ValueError(
            "Distribución inesperada por museo. "
            f"Esperado: {expected_counts}. "
            f"Observado: {observed_counts}."
        )

    required_complete = [
        "item_id",
        "museum",
        "source_object_id",
        "title",
        "culture",
        "period",
        "object_type",
        "classification",
        "image_url",
        "object_url",
        "license",
        "dataset_split",
        "review_status",
        "source_branch",
        "source_commit",
        "source_file",
        "enrichment_source_file",
        "metadata_recovery_source",
        "metadata_recovery_version",
        "adapter_name",
        "adapter_version",
        "processing_timestamp",
    ]

    for column in required_complete:
        if column not in corpus.columns:
            raise ValueError(
                f"Falta la columna obligatoria: {column}"
            )

        empty_count = (
            corpus[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        if empty_count:
            raise ValueError(
                f"{column} contiene {empty_count} valores vacíos."
            )

    if set(corpus["dataset_split"]) != {"principal"}:
        raise ValueError(
            "El corpus v1 debe incluir únicamente el conjunto principal."
        )

    if set(corpus["review_status"]) != {"aprobado"}:
        raise ValueError(
            "El corpus v1 debe incluir únicamente objetos aprobados."
        )

    timestamps = corpus["processing_timestamp"].unique()

    if len(timestamps) != 1:
        raise ValueError(
            "El corpus contiene múltiples timestamps de procesamiento."
        )

    expected_timestamp = get_processing_timestamp()

    if timestamps[0] != expected_timestamp:
        raise ValueError(
            "El timestamp del corpus no coincide con "
            "la configuración de construcción."
        )


def build_enriched_corpus(
    processing_timestamp: str | None = None,
) -> pd.DataFrame:
    timestamp = (
        processing_timestamp or get_processing_timestamp()
    )

    met_reviewed = load_csv(MET_REVIEWED_PATH)
    met_inventory = load_csv(MET_INVENTORY_PATH)

    cma_reviewed = load_csv(CMA_REVIEWED_PATH)
    cma_candidates = load_csv(CMA_CANDIDATES_PATH)

    met_recovered = recover_met_metadata(
        met_reviewed,
        met_inventory,
    )
    cma_recovered = recover_cma_metadata(
        cma_reviewed,
        cma_candidates,
    )

    met_adapted = adapt_met_enriched(
        met_recovered,
        processing_timestamp=timestamp,
    )
    cma_adapted = adapt_cma_enriched(
        cma_recovered,
        processing_timestamp=timestamp,
    )

    validate_same_schema(
        met_adapted,
        cma_adapted,
    )

    config = load_build_config()
    sort_columns = config["build"]["sort_by"]

    corpus = pd.concat(
        [met_adapted, cma_adapted],
        ignore_index=True,
    ).sort_values(
        sort_columns,
        kind="stable",
    ).reset_index(drop=True)

    validate_enriched_corpus(corpus)

    return corpus


def run() -> pd.DataFrame:
    corpus = build_enriched_corpus()

    OUTPUT_PARQUET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    corpus.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )
    corpus.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Corpus enriquecido: {len(corpus)} registros")
    print(f"Columnas: {len(corpus.columns)}")
    print(
        "Distribución:",
        corpus["museum"].value_counts().to_dict(),
    )
    print(
        "Timestamp:",
        corpus["processing_timestamp"].iloc[0],
    )
    print(f"Parquet: {OUTPUT_PARQUET}")
    print(f"CSV: {OUTPUT_CSV}")

    return corpus


if __name__ == "__main__":
    run()