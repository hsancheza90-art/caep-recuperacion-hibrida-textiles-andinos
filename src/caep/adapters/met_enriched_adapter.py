from __future__ import annotations

from pathlib import Path

import pandas as pd

from caep.config import get_processing_timestamp

from caep.adapters.common import (
    assert_required_columns,
    build_item_id,
    clean_text,
    first_non_empty,
    validate_unique_non_empty,
)


ADAPTER_NAME = "met_enriched_adapter"
ADAPTER_VERSION = "1.0.0"

SOURCE_BRANCH = "mejora/met-flujo-reproducible-es"
SOURCE_COMMIT = "a5f32a929d7c895c4b151e162f734746c226abc4"
SOURCE_FILE = "data/metadata/met_corpus_principal_v2_revisado.csv"
ENRICHMENT_FILE = (
    "data/processed/corpus_met_textiles_andinos_v1_inventario_base.csv"
)

DEFAULT_INPUT_PATH = Path(
    "data/interim/enriched/met_principal_enriched_v1.parquet"
)
DEFAULT_OUTPUT_PATH = Path(
    "data/interim/adapted/met_principal_enriched_adapted_v1.parquet"
)

REQUIRED_COLUMNS = {
    "id_fuente",
    "titulo_original",
    "titulo_es_sugerido",
    "cultura",
    "fecha_objeto",
    "procedencia",
    "tipo_objeto",
    "material",
    "material_normalizado",
    "tecnica",
    "estado_curacion",
    "motivo_curacion",
    "url_objeto",
    "url_imagen",
    "licencia",
    "decision_revision",
    "revisor",
    "fecha_revision",
    "clasificacion_original_recuperada",
    "clasificacion_normalizada_recuperada",
    "pais_recuperado",
    "region_recuperada",
    "metadata_recovery_source",
    "metadata_recovery_version",
}


def adapt_met_enriched(
    frame: pd.DataFrame,
    processing_timestamp: str | None = None,
) -> pd.DataFrame:
    assert_required_columns(
        frame,
        REQUIRED_COLUMNS,
        source_name="MET enriched",
    )

    processing_timestamp = (
        processing_timestamp or get_processing_timestamp()
    )

    output = pd.DataFrame(
        {
            "item_id": frame["id_fuente"].map(
                lambda value: build_item_id("MET", value)
            ),
            "museum": "MET",
            "source_object_id": frame["id_fuente"].map(clean_text),
            "source_accession_number": "",
            "title": frame.apply(
                lambda row: first_non_empty(
                    row["titulo_es_sugerido"],
                    row["titulo_original"],
                ),
                axis=1,
            ),
            "title_original": frame["titulo_original"].map(clean_text),
            "description": "",
            "culture": frame["cultura"].map(clean_text),
            "culture_original": frame["cultura"].map(clean_text),
            "period": frame["fecha_objeto"].map(clean_text),
            "period_original": frame["fecha_objeto"].map(clean_text),
            "object_type": frame["tipo_objeto"].map(clean_text),
            "object_type_original": frame["tipo_objeto"].map(clean_text),
            "material": frame.apply(
                lambda row: first_non_empty(
                    row["material_normalizado"],
                    row["material"],
                ),
                axis=1,
            ),
            "material_original": frame["material"].map(clean_text),
            "technique": frame["tecnica"].map(clean_text),
            "technique_original": frame["tecnica"].map(clean_text),
            "provenance": frame["procedencia"].map(clean_text),
            "classification": frame.apply(
                lambda row: first_non_empty(
                    row["clasificacion_normalizada_recuperada"],
                    row["clasificacion_original_recuperada"],
                ),
                axis=1,
            ),
            "country": frame["pais_recuperado"].map(clean_text),
            "region": frame["region_recuperada"].map(clean_text),
            "image_url": frame["url_imagen"].map(clean_text),
            "object_url": frame["url_objeto"].map(clean_text),
            "license": frame["licencia"].map(clean_text),
            "dataset_split": "principal",
            "review_status": "aprobado",
            "review_decision_original": frame["decision_revision"].map(
                clean_text
            ),
            "review_reason": frame["motivo_curacion"].map(clean_text),
            "reviewer": frame["revisor"].map(clean_text),
            "review_date": frame["fecha_revision"].map(clean_text),
            "source_branch": SOURCE_BRANCH,
            "source_commit": SOURCE_COMMIT,
            "source_file": SOURCE_FILE,
            "enrichment_source_file": ENRICHMENT_FILE,
            "metadata_recovery_source": frame[
                "metadata_recovery_source"
            ].map(clean_text),
            "metadata_recovery_version": frame[
                "metadata_recovery_version"
            ].map(clean_text),
            "adapter_name": ADAPTER_NAME,
            "adapter_version": ADAPTER_VERSION,
            "processing_timestamp": processing_timestamp,
            "image_local_path": "",
            "image_sha256": "",
            "image_phash": "",
        }
    )

    validate_unique_non_empty(output["item_id"], "MET.item_id")
    validate_unique_non_empty(
        output["source_object_id"],
        "MET.source_object_id",
    )

    return output


def run(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    frame = pd.read_parquet(input_path)
    adapted = adapt_met_enriched(frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adapted.to_parquet(output_path, index=False)
    adapted.to_csv(
        output_path.with_suffix(".csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print(f"MET enriquecido adaptado: {len(adapted)} registros")
    print(f"Columnas: {len(adapted.columns)}")

    return adapted


if __name__ == "__main__":
    run()