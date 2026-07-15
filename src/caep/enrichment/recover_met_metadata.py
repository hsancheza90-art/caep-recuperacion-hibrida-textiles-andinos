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
    "data/source/met/met_corpus_principal_v2_revisado.csv"
)
INVENTORY_PATH = Path(
    "data/source/met/met_inventario_base.csv"
)

OUTPUT_PARQUET = Path(
    "data/interim/enriched/met_principal_enriched_v1.parquet"
)
OUTPUT_CSV = OUTPUT_PARQUET.with_suffix(".csv")


def recover_met_metadata(
    reviewed: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    reviewed = reviewed.copy()
    inventory = inventory.copy()

    reviewed["_join_key"] = normalize_key(
        reviewed["id_fuente"]
    )
    inventory["_join_key"] = normalize_key(
        inventory["id_objeto"]
    )

    assert_unique_non_empty_key(
        reviewed,
        "_join_key",
        "MET revisado",
    )
    assert_unique_non_empty_key(
        inventory,
        "_join_key",
        "MET inventario",
    )

    inventory_fields = [
        "_join_key",
        "titulo_original",
        "titulo_es_sugerido",
        "cultura",
        "fecha_objeto",
        "pais",
        "region",
        "clasificacion_original",
        "clasificacion_normalizada",
        "nombre_objeto_original",
        "material_original",
        "dimensiones",
        "repositorio",
        "url_objeto",
        "url_imagen",
        "url_imagen_miniatura",
        "es_dominio_publico",
        "derechos_reproduccion",
        "etiquetas_originales",
        "fecha_descarga",
    ]

    inventory_subset = inventory[inventory_fields].copy()

    inventory_subset = inventory_subset.rename(
        columns={
            column: f"inventory_{column}"
            for column in inventory_subset.columns
            if column != "_join_key"
        }
    )

    merged = reviewed.merge(
        inventory_subset,
        on="_join_key",
        how="left",
        validate="one_to_one",
        indicator="_merge_status",
    )

    assert_complete_match(
        merged,
        indicator_column="_merge_status",
        expected_rows=127,
        source_name="MET",
    )

    merged["titulo_original"] = coalesce_series(
        merged["titulo_original"],
        merged["inventory_titulo_original"],
    )
    merged["titulo_es_sugerido"] = coalesce_series(
        merged["titulo_es_sugerido"],
        merged["inventory_titulo_es_sugerido"],
    )
    merged["cultura"] = coalesce_series(
        merged["cultura"],
        merged["inventory_cultura"],
    )
    merged["fecha_objeto"] = coalesce_series(
        merged["fecha_objeto"],
        merged["inventory_fecha_objeto"],
    )
    merged["tipo_objeto"] = coalesce_series(
        merged["tipo_objeto"],
        merged["inventory_nombre_objeto_original"],
    )
    merged["material"] = coalesce_series(
        merged["material"],
        merged["inventory_material_original"],
    )
    merged["url_objeto"] = coalesce_series(
        merged["url_objeto"],
        merged["inventory_url_objeto"],
    )
    merged["url_imagen"] = coalesce_series(
        merged["url_imagen"],
        merged["inventory_url_imagen"],
    )

    merged["clasificacion_original_recuperada"] = merged[
        "inventory_clasificacion_original"
    ].map(clean_text)

    merged["clasificacion_normalizada_recuperada"] = merged[
        "inventory_clasificacion_normalizada"
    ].map(clean_text)

    merged["pais_recuperado"] = merged[
        "inventory_pais"
    ].map(clean_text)

    merged["region_recuperada"] = merged[
        "inventory_region"
    ].map(clean_text)

    merged["dimensiones_recuperadas"] = merged[
        "inventory_dimensiones"
    ].map(clean_text)

    merged["etiquetas_originales_recuperadas"] = merged[
        "inventory_etiquetas_originales"
    ].map(clean_text)

    merged["metadata_recovery_source"] = (
        "met_inventario_base"
    )
    merged["metadata_recovery_version"] = "1.0.0"

    drop_columns = [
        column
        for column in merged.columns
        if column.startswith("inventory_")
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
    inventory = pd.read_csv(
        INVENTORY_PATH,
        dtype=str,
        keep_default_na=False,
    )

    enriched = recover_met_metadata(
        reviewed,
        inventory,
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

    print(f"MET enriquecido: {len(enriched)} registros")
    print(f"Columnas: {len(enriched.columns)}")
    print(f"Parquet: {OUTPUT_PARQUET}")
    print(f"CSV: {OUTPUT_CSV}")

    return enriched


if __name__ == "__main__":
    run()