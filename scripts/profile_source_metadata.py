from __future__ import annotations

from pathlib import Path

import pandas as pd


SOURCES = {
    "MET": Path(
        "data/source/met/met_corpus_principal_v2_revisado.csv"
    ),
    "CMA": Path(
        "data/source/cma/cma_corpus_principal_revisado.csv"
    ),
}

FIELDS = {
    "MET": [
        "id_fuente",
        "titulo_original",
        "titulo_es_sugerido",
        "cultura",
        "fecha_objeto",
        "procedencia",
        "tipo_objeto",
        "tipo_superficie",
        "material",
        "material_normalizado",
        "tecnica",
        "url_objeto",
        "url_imagen",
    ],
    "CMA": [
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
        "tipo_objeto_manual",
        "cultura_manual",
        "periodo_manual",
        "notas_iconograficas",
        "notas_tecnicas",
        "url_objeto",
        "url_imagen",
    ],
}


def normalize_empty(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
    )


def profile_source(
    museum: str,
    path: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )

    rows: list[dict[str, object]] = []

    for field in FIELDS[museum]:
        if field not in frame.columns:
            rows.append(
                {
                    "museum": museum,
                    "field": field,
                    "available": False,
                    "non_empty": 0,
                    "total": len(frame),
                    "coverage": 0.0,
                    "unique_non_empty": 0,
                }
            )
            continue

        values = normalize_empty(frame[field])
        non_empty_values = values[values.ne("")]

        rows.append(
            {
                "museum": museum,
                "field": field,
                "available": True,
                "non_empty": int(non_empty_values.size),
                "total": len(frame),
                "coverage": (
                    non_empty_values.size / len(frame)
                    if len(frame)
                    else 0.0
                ),
                "unique_non_empty": int(
                    non_empty_values.nunique()
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    reports = [
        profile_source(museum, path)
        for museum, path in SOURCES.items()
    ]

    report = pd.concat(
        reports,
        ignore_index=True,
    )

    output = Path(
        "outputs/reports/source_metadata_profile_v1.csv"
    )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
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