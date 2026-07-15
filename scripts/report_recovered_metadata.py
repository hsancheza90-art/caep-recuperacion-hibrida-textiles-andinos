from __future__ import annotations

from pathlib import Path

import pandas as pd


SOURCES = {
    "MET": Path(
        "data/interim/enriched/met_principal_enriched_v1.parquet"
    ),
    "CMA": Path(
        "data/interim/enriched/cma_principal_enriched_v1.parquet"
    ),
}

FIELDS = {
    "MET": [
        "titulo_original",
        "titulo_es_sugerido",
        "cultura",
        "fecha_objeto",
        "tipo_objeto",
        "material",
        "tecnica",
        "clasificacion_original_recuperada",
        "clasificacion_normalizada_recuperada",
        "pais_recuperado",
        "region_recuperada",
    ],
    "CMA": [
        "numero_acceso",
        "titulo_original",
        "cultura",
        "periodo",
        "clasificacion_original",
        "tecnica_original",
        "material_original",
        "descripcion_original",
        "departamento_recuperado",
        "licencia_recuperada",
    ],
}


def main() -> None:
    rows = []

    for museum, path in SOURCES.items():
        frame = pd.read_parquet(path)

        for field in FIELDS[museum]:
            values = (
                frame[field]
                .fillna("")
                .astype(str)
                .str.strip()
            )

            non_empty = int(values.ne("").sum())
            total = len(frame)

            rows.append(
                {
                    "museum": museum,
                    "field": field,
                    "non_empty": non_empty,
                    "total": total,
                    "coverage": (
                        non_empty / total
                        if total
                        else 0
                    ),
                    "unique_non_empty": int(
                        values[values.ne("")].nunique()
                    ),
                }
            )

    report = pd.DataFrame(rows)

    output = Path(
        "outputs/reports/recovered_metadata_profile_v1.csv"
    )

    report.to_csv(
        output,
        index=False,
        encoding="utf-8-sig",
    )

    print(report.to_string(index=False))
    print(f"\nReporte: {output}")


if __name__ == "__main__":
    main()