from __future__ import annotations

from pathlib import Path

import pandas as pd

from caep.enrichment.common import (
    assert_complete_match,
    assert_unique_non_empty_key,
    clean_text,
    coalesce_series,
    normalize_key,
)


REVIEWED_PATH = Path(
    "data/source/cma/cma_corpus_principal_revisado.csv"
)
CANDIDATES_PATH = Path(
    "data/source/cma/cma_andes_textiles_candidates.csv"
)

OUTPUT_PARQUET = Path(
    "data/interim/enriched/cma_principal_enriched_v1.parquet"
)
OUTPUT_CSV = OUTPUT_PARQUET.with_suffix(".csv")


def recover_cma_metadata(
    reviewed: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    reviewed = reviewed.copy()
    candidates = candidates.copy()

    reviewed["_join_key"] = normalize_key(
        reviewed["id_objeto"]
    )
    candidates["_join_key"] = normalize_key(
        candidates["source_id"]
    )

    assert_unique_non_empty_key(
        reviewed,
        "_join_key",
        "CMA revisado",
    )
    assert_unique_non_empty_key(
        candidates,
        "_join_key",
        "CMA candidatos",
    )

    candidate_fields = [
        "_join_key",
        "accession_number",
        "title",
        "culture",
        "creation_date",
        "classification",
        "technique",
        "medium",
        "department",
        "description",
        "url",
        "image_url",
        "share_license_status",
        "andean_terms",
        "textile_terms",
        "raw_queries",
    ]

    candidate_subset = candidates[
        candidate_fields
    ].copy()

    candidate_subset = candidate_subset.rename(
        columns={
            column: f"candidate_{column}"
            for column in candidate_subset.columns
            if column != "_join_key"
        }
    )

    merged = reviewed.merge(
        candidate_subset,
        on="_join_key",
        how="left",
        validate="one_to_one",
        indicator="_merge_status",
    )

    assert_complete_match(
        merged,
        indicator_column="_merge_status",
        expected_rows=88,
        source_name="CMA",
    )

    merged["numero_acceso"] = coalesce_series(
        merged["numero_acceso"],
        merged["candidate_accession_number"],
    )
    merged["titulo_original"] = coalesce_series(
        merged["titulo_original"],
        merged["candidate_title"],
    )
    merged["cultura"] = coalesce_series(
        merged["cultura"],
        merged["candidate_culture"],
    )
    merged["periodo"] = coalesce_series(
        merged["periodo"],
        merged["candidate_creation_date"],
    )
    merged["clasificacion_original"] = coalesce_series(
        merged["clasificacion_original"],
        merged["candidate_classification"],
    )
    merged["tecnica_original"] = coalesce_series(
        merged["tecnica_original"],
        merged["candidate_technique"],
    )
    merged["material_original"] = coalesce_series(
        merged["material_original"],
        merged["candidate_medium"],
    )
    merged["descripcion_original"] = coalesce_series(
        merged["descripcion_original"],
        merged["candidate_description"],
    )
    merged["url_objeto"] = coalesce_series(
        merged["url_objeto"],
        merged["candidate_url"],
    )
    merged["url_imagen"] = coalesce_series(
        merged["url_imagen"],
        merged["candidate_image_url"],
    )

    merged["departamento_recuperado"] = merged[
        "candidate_department"
    ].map(clean_text)

    merged["licencia_recuperada"] = merged[
        "candidate_share_license_status"
    ].map(clean_text)

    merged["terminos_andinos_recuperados"] = merged[
        "candidate_andean_terms"
    ].map(clean_text)

    merged["terminos_textiles_recuperados"] = merged[
        "candidate_textile_terms"
    ].map(clean_text)

    merged["consultas_origen_recuperadas"] = merged[
        "candidate_raw_queries"
    ].map(clean_text)

    merged["metadata_recovery_source"] = (
        "cma_andes_textiles_candidates"
    )
    merged["metadata_recovery_version"] = "1.0.0"

    drop_columns = [
        column
        for column in merged.columns
        if column.startswith("candidate_")
    ] + [
        "_join_key",
        "_merge_status",
    ]

    return merged.drop(columns=drop_columns)


def run() -> pd.DataFrame:
    reviewed = pd.read_csv(
        REVIEWED_PATH,
        dtype=str,
        keep_default_na=False,
    )
    candidates = pd.read_csv(
        CANDIDATES_PATH,
        dtype=str,
        keep_default_na=False,
    )

    enriched = recover_cma_metadata(
        reviewed,
        candidates,
    )

    OUTPUT_PARQUET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    enriched.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )
    enriched.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"CMA enriquecido: {len(enriched)} registros")
    print(f"Columnas: {len(enriched.columns)}")
    print(f"Parquet: {OUTPUT_PARQUET}")
    print(f"CSV: {OUTPUT_CSV}")

    return enriched


if __name__ == "__main__":
    run()