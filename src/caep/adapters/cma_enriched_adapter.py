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


ADAPTER_NAME = "cma_enriched_adapter"
ADAPTER_VERSION = "1.0.0"

SOURCE_BRANCH = "mejora/cma-flujo-reproducible-es"
SOURCE_COMMIT = "0659cd14769fa8e2cf1bde4b01b2cbb3f0415db6"
SOURCE_FILE = "data/metadata/cma_corpus_principal_revisado.csv"
ENRICHMENT_FILE = "data/metadata/cma_andes_textiles_candidates.csv"

DEFAULT_INPUT_PATH = Path(
    "data/interim/enriched/cma_principal_enriched_v1.parquet"
)
DEFAULT_OUTPUT_PATH = Path(
    "data/interim/adapted/cma_principal_enriched_adapted_v1.parquet"
)

REQUIRED_COLUMNS = {
    "id_objeto",
    "numero_acceso",
    "titulo_original",
    "cultura",
    "periodo",
    "clasificacion_original",
    "nombre_objeto_original",
    "tecnica_original",
    "material_original",
    "descripcion_original",
    "url_objeto",
    "url_imagen",
    "decision_auditoria",
    "motivo_auditoria",
    "tipo_objeto_manual",
    "cultura_manual",
    "periodo_manual",
    "revisor",
    "fecha_revision",
    "departamento_recuperado",
    "licencia_recuperada",
    "metadata_recovery_source",
    "metadata_recovery_version",
}


def adapt_cma_enriched(
    frame: pd.DataFrame,
    processing_timestamp: str | None = None,
) -> pd.DataFrame:
    assert_required_columns(
        frame,
        REQUIRED_COLUMNS,
        source_name="CMA enriched",
    )

    processing_timestamp = (
        processing_timestamp or get_processing_timestamp()
    )

    output = pd.DataFrame(
        {
            "item_id": frame["id_objeto"].map(
                lambda value: build_item_id("CMA", value)
            ),
            "museum": "CMA",
            "source_object_id": frame["id_objeto"].map(clean_text),
            "source_accession_number": frame["numero_acceso"].map(
                clean_text
            ),
            "title": frame["titulo_original"].map(clean_text),
            "title_original": frame["titulo_original"].map(clean_text),
            "description": frame["descripcion_original"].map(clean_text),
            "culture": frame.apply(
                lambda row: first_non_empty(
                    row["cultura_manual"],
                    row["cultura"],
                ),
                axis=1,
            ),
            "culture_original": frame["cultura"].map(clean_text),
            "period": frame.apply(
                lambda row: first_non_empty(
                    row["periodo_manual"],
                    row["periodo"],
                ),
                axis=1,
            ),
            "period_original": frame["periodo"].map(clean_text),
            "object_type": frame.apply(
                lambda row: first_non_empty(
                    row["tipo_objeto_manual"],
                    row["nombre_objeto_original"],
                ),
                axis=1,
            ),
            "object_type_original": frame["nombre_objeto_original"].map(
                clean_text
            ),
            "material": frame["material_original"].map(clean_text),
            "material_original": frame["material_original"].map(
                clean_text
            ),
            "technique": frame["tecnica_original"].map(clean_text),
            "technique_original": frame["tecnica_original"].map(
                clean_text
            ),
            "provenance": "",
            "classification": frame["clasificacion_original"].map(
                clean_text
            ),
            "country": "",
            "region": "",
            "image_url": frame["url_imagen"].map(clean_text),
            "object_url": frame["url_objeto"].map(clean_text),
            "license": frame["licencia_recuperada"].map(clean_text),
            "dataset_split": "principal",
            "review_status": "aprobado",
            "review_decision_original": frame[
                "decision_auditoria"
            ].map(clean_text),
            "review_reason": frame["motivo_auditoria"].map(clean_text),
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

    validate_unique_non_empty(output["item_id"], "CMA.item_id")
    validate_unique_non_empty(
        output["source_object_id"],
        "CMA.source_object_id",
    )

    return output


def run(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    frame = pd.read_parquet(input_path)
    adapted = adapt_cma_enriched(frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adapted.to_parquet(output_path, index=False)
    adapted.to_csv(
        output_path.with_suffix(".csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print(f"CMA enriquecido adaptado: {len(adapted)} registros")
    print(f"Columnas: {len(adapted.columns)}")

    return adapted


if __name__ == "__main__":
    run()