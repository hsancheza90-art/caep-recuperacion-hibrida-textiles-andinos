"""Consolida las decisiones curatoriales del mapeo cultural.

La política conserva tres niveles:

1. include_strict:
   atribución cultural directa y utilizable como ground truth estricto;

2. include_non_strict:
   atribución estilística, compuesta o incierta, utilizable como metadata
   descriptiva, pero no como ground truth cultural estricto;

3. exclude:
   etiqueta sin atribución cultural específica.

El archivo de revisión original no se modifica.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REVIEW_PATH = (
    PROJECT_ROOT
    / "config"
    / "mappings"
    / "culture_mapping_review_v1.csv"
)

CURATED_PATH = (
    PROJECT_ROOT
    / "config"
    / "mappings"
    / "culture_mapping_curated_v1.csv"
)

POLICY_NAME = "culture_curatorial_policy_v1"
POLICY_DATE = "2026-07-15"


REQUIRED_COLUMNS = {
    "museum",
    "source_label",
    "canonical_components",
    "record_count",
    "proposal_decision",
    "review_decision",
    "final_canonical_components",
    "attribution_type",
    "strict_ground_truth_eligible",
    "review_status",
    "review_note",
    "reviewer",
    "review_date",
}


SUPPORTED_ATTRIBUTION_TYPES = {
    "direct",
    "style",
    "composite",
    "uncertain",
    "unattributed",
}


OUTPUT_COLUMNS = [
    "museum",
    "source_label",
    "canonical_components",
    "record_count",
    "proposal_decision",
    "review_decision",
    "final_canonical_components",
    "attribution_type",
    "strict_ground_truth_eligible",
    "review_status",
    "decision_basis",
    "review_note",
    "reviewer",
    "review_date",
]


def clean_text(series: pd.Series) -> pd.Series:
    """Normaliza valores textuales sin alterar su contenido semántico."""

    return (
        series.fillna("")
        .astype(str)
        .str.strip()
    )


def validate_review_table(review: pd.DataFrame) -> None:
    """Valida la tabla preparada antes de consolidarla."""

    missing_columns = REQUIRED_COLUMNS.difference(review.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Faltan columnas requeridas en la revisión: {missing_text}"
        )

    duplicated = review.duplicated(
        subset=["museum", "source_label"],
        keep=False,
    )

    if duplicated.any():
        duplicate_rows = review.loc[
            duplicated,
            ["museum", "source_label"],
        ]

        raise ValueError(
            "Existen pares museum/source_label duplicados:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    attribution_types = set(
        clean_text(review["attribution_type"])
    )

    unknown_types = sorted(
        attribution_types - SUPPORTED_ATTRIBUTION_TYPES
    )

    if unknown_types:
        raise ValueError(
            "Se encontraron tipos de atribución no soportados: "
            f"{unknown_types}"
        )


def resolve_row(row: pd.Series) -> pd.Series:
    """Aplica la política curatorial a una fila."""

    attribution_type = str(row["attribution_type"]).strip()

    components = str(
        row.get("final_canonical_components", "")
        if pd.notna(row.get("final_canonical_components", ""))
        else ""
    ).strip()

    if attribution_type == "direct":
        if not components:
            raise ValueError(
                "Una atribución directa no puede carecer de "
                "componentes canónicos."
            )

        row["review_decision"] = "include_strict"
        row["strict_ground_truth_eligible"] = True
        row["review_status"] = "resolved"
        row["decision_basis"] = "direct_attribution"

        row["review_note"] = (
            "Atribución cultural directa. Se conserva como etiqueta "
            "canónica y puede utilizarse en el ground truth cultural estricto."
        )

    elif attribution_type == "style":
        if not components:
            raise ValueError(
                "Una atribución estilística debe conservar al menos "
                "un componente cultural canónico."
            )

        row["review_decision"] = "include_non_strict"
        row["strict_ground_truth_eligible"] = False
        row["review_status"] = "resolved"
        row["decision_basis"] = "stylistic_attribution"

        row["review_note"] = (
            "La fuente expresa una atribución de estilo. Se conserva como "
            "metadata cultural no estricta, pero no se utiliza como ground "
            "truth cultural inequívoco."
        )

    elif attribution_type == "composite":
        if not components or "|" not in components:
            raise ValueError(
                "Una atribución compuesta debe conservar múltiples "
                "componentes separados por '|'."
            )

        row["review_decision"] = "include_non_strict"
        row["strict_ground_truth_eligible"] = False
        row["review_status"] = "resolved"
        row["decision_basis"] = "composite_attribution"

        row["review_note"] = (
            "La fuente contiene más de una atribución cultural. Se conservan "
            "todos los componentes sin reducirlos automáticamente a una sola "
            "cultura."
        )

    elif attribution_type == "uncertain":
        if not components:
            raise ValueError(
                "Una atribución incierta incluida debe conservar al menos "
                "un componente cultural identificable."
            )

        row["review_decision"] = "include_non_strict"
        row["strict_ground_truth_eligible"] = False
        row["review_status"] = "resolved"
        row["decision_basis"] = "uncertain_attribution"

        row["review_note"] = (
            "La fuente presenta incertidumbre explícita, alternativas o "
            "signos de interrogación. Se conserva para análisis descriptivo, "
            "pero no se utiliza como ground truth cultural estricto."
        )

    elif attribution_type == "unattributed":
        row["review_decision"] = "exclude"
        row["final_canonical_components"] = ""
        row["strict_ground_truth_eligible"] = False
        row["review_status"] = "resolved"
        row["decision_basis"] = "insufficient_cultural_attribution"

        row["review_note"] = (
            "La etiqueta no identifica una cultura específica. El registro "
            "se conserva en el corpus, pero queda excluido de evaluaciones "
            "basadas en identidad cultural."
        )

    else:
        raise ValueError(
            f"Tipo de atribución no soportado: {attribution_type}"
        )

    row["reviewer"] = POLICY_NAME
    row["review_date"] = POLICY_DATE

    return row


def build_curated_mapping(review: pd.DataFrame) -> pd.DataFrame:
    """Construye el mapeo cultural curado y resuelto."""

    validate_review_table(review)

    curated = review.copy()

    curated["museum"] = clean_text(curated["museum"])
    curated["source_label"] = clean_text(curated["source_label"])
    curated["canonical_components"] = clean_text(
        curated["canonical_components"]
    )
    curated["final_canonical_components"] = clean_text(
        curated["final_canonical_components"]
    )
    curated["attribution_type"] = clean_text(
        curated["attribution_type"]
    )

    curated = curated.apply(
        resolve_row,
        axis=1,
    )

    unresolved = curated[
        curated["review_decision"].eq("pending")
        | ~curated["review_status"].eq("resolved")
    ]

    if not unresolved.empty:
        raise ValueError(
            "El mapeo contiene decisiones curatoriales sin resolver."
        )

    curated = curated[OUTPUT_COLUMNS].sort_values(
        by=[
            "museum",
            "source_label",
        ],
        kind="stable",
    )

    return curated.reset_index(drop=True)


def print_summary(curated: pd.DataFrame) -> None:
    """Imprime el resumen de las decisiones consolidadas."""

    summary = (
        curated.groupby(
            [
                "review_decision",
                "strict_ground_truth_eligible",
            ],
            dropna=False,
        )
        .agg(
            source_categories=("source_label", "size"),
            represented_records=("record_count", "sum"),
        )
        .reset_index()
    )

    print("\nMAPEO CULTURAL CURADO")
    print("=" * 100)
    print(summary.to_string(index=False))

    type_summary = (
        curated.groupby(
            [
                "attribution_type",
                "review_decision",
            ],
            dropna=False,
        )
        .agg(
            source_categories=("source_label", "size"),
            represented_records=("record_count", "sum"),
        )
        .reset_index()
    )

    print("\nRESUMEN POR TIPO DE ATRIBUCIÓN")
    print("=" * 100)
    print(type_summary.to_string(index=False))

    total_categories = len(curated)
    total_records = int(curated["record_count"].sum())

    print("\nTOTALES")
    print("=" * 100)
    print(f"Categorías fuente: {total_categories}")
    print(f"Registros representados: {total_records}")
    print(
        "Archivo generado: "
        f"{CURATED_PATH.relative_to(PROJECT_ROOT)}"
    )


def main() -> None:
    """Ejecuta la consolidación del mapeo cultural."""

    if not REVIEW_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró la tabla de revisión: {REVIEW_PATH}"
        )

    review = pd.read_csv(REVIEW_PATH)

    curated = build_curated_mapping(review)

    CURATED_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    curated.to_csv(
        CURATED_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print_summary(curated)


if __name__ == "__main__":
    main()