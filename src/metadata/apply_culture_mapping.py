"""Aplica el mapeo cultural curado al corpus enriquecido.

El corpus base permanece inmutable. El proceso genera un nuevo corpus
derivado con campos culturales normalizados y trazables.

La unión se realiza mediante:

    museum + culture

donde `culture` conserva la etiqueta museográfica original.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paper_corpus_enriched_v1.csv"
)

CURATED_MAPPING_PATH = (
    PROJECT_ROOT
    / "config"
    / "mappings"
    / "culture_mapping_curated_v1.csv"
)

OUTPUT_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paper_corpus_culture_enriched_v1.csv"
)

OUTPUT_PARQUET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paper_corpus_culture_enriched_v1.parquet"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "culture_enrichment_summary_v1.csv"
)

MAPPING_VERSION = "culture_mapping_curated_v1"


REQUIRED_CORPUS_COLUMNS = {
    "item_id",
    "museum",
    "culture",
}


REQUIRED_MAPPING_COLUMNS = {
    "museum",
    "source_label",
    "record_count",
    "review_decision",
    "final_canonical_components",
    "attribution_type",
    "strict_ground_truth_eligible",
    "review_status",
    "decision_basis",
    "review_note",
}


CULTURE_OUTPUT_COLUMNS = [
    "culture_source",
    "culture_canonical",
    "culture_components",
    "culture_component_count",
    "culture_attribution_type",
    "culture_mapping_decision",
    "culture_mapping_basis",
    "culture_strict_ground_truth_eligible",
    "culture_mapping_status",
    "culture_mapping_note",
    "culture_mapping_version",
]


def clean_text(series: pd.Series) -> pd.Series:
    """Limpia espacios y valores nulos sin cambiar el contenido."""

    return (
        series.fillna("")
        .astype(str)
        .str.strip()
    )


def parse_boolean_series(series: pd.Series) -> pd.Series:
    """Convierte una serie textual o booleana en booleanos reales."""

    normalized = clean_text(series).str.lower()

    boolean_map = {
        "true": True,
        "1": True,
        "yes": True,
        "si": True,
        "sí": True,
        "false": False,
        "0": False,
        "no": False,
    }

    parsed = normalized.map(boolean_map)

    if parsed.isna().any():
        invalid_values = sorted(
            normalized[parsed.isna()].unique().tolist()
        )

        raise ValueError(
            "Se encontraron valores booleanos no reconocidos: "
            f"{invalid_values}"
        )

    return parsed.astype(bool)


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    artifact_name: str,
) -> None:
    """Valida la presencia de columnas requeridas."""

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))

        raise ValueError(
            f"Faltan columnas en {artifact_name}: {missing_text}"
        )


def validate_corpus(corpus: pd.DataFrame) -> None:
    """Valida el corpus antes de aplicar el mapeo."""

    validate_required_columns(
        corpus,
        REQUIRED_CORPUS_COLUMNS,
        "el corpus enriquecido",
    )

    if corpus["item_id"].duplicated().any():
        duplicated_ids = (
            corpus.loc[
                corpus["item_id"].duplicated(keep=False),
                "item_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "El corpus contiene item_id duplicados: "
            f"{duplicated_ids[:10]}"
        )

    for column in ["item_id", "museum", "culture"]:
        empty_mask = clean_text(corpus[column]).eq("")

        if empty_mask.any():
            raise ValueError(
                f"El corpus contiene valores vacíos en {column}: "
                f"{int(empty_mask.sum())}"
            )

    existing_derived_columns = set(
        CULTURE_OUTPUT_COLUMNS
    ).intersection(corpus.columns)

    if existing_derived_columns:
        columns_text = ", ".join(
            sorted(existing_derived_columns)
        )

        raise ValueError(
            "El corpus ya contiene columnas culturales derivadas: "
            f"{columns_text}"
        )


def validate_mapping(mapping: pd.DataFrame) -> None:
    """Valida el mapeo cultural curado."""

    validate_required_columns(
        mapping,
        REQUIRED_MAPPING_COLUMNS,
        "el mapeo cultural curado",
    )

    duplicated = mapping.duplicated(
        subset=["museum", "source_label"],
        keep=False,
    )

    if duplicated.any():
        duplicated_rows = mapping.loc[
            duplicated,
            ["museum", "source_label"],
        ]

        raise ValueError(
            "El mapeo contiene claves duplicadas:\n"
            f"{duplicated_rows.to_string(index=False)}"
        )

    decisions = set(
        clean_text(mapping["review_decision"])
    )

    allowed_decisions = {
        "include_strict",
        "include_non_strict",
        "exclude",
    }

    unknown_decisions = decisions - allowed_decisions

    if unknown_decisions:
        raise ValueError(
            "El mapeo contiene decisiones no soportadas: "
            f"{sorted(unknown_decisions)}"
        )

    statuses = set(
        clean_text(mapping["review_status"])
    )

    if statuses != {"resolved"}:
        raise ValueError(
            "Todas las decisiones culturales deben estar resueltas. "
            f"Estados encontrados: {sorted(statuses)}"
        )


def validate_mapping_counts(
    corpus: pd.DataFrame,
    mapping: pd.DataFrame,
) -> None:
    """Comprueba que los conteos del mapeo coincidan con el corpus."""

    actual_counts = (
        corpus.assign(
            museum_key=clean_text(corpus["museum"]),
            culture_key=clean_text(corpus["culture"]),
        )
        .groupby(
            ["museum_key", "culture_key"],
            dropna=False,
        )
        .size()
        .reset_index(name="actual_record_count")
        .rename(
            columns={
                "museum_key": "museum",
                "culture_key": "source_label",
            }
        )
    )

    expected_counts = mapping[
        [
            "museum",
            "source_label",
            "record_count",
        ]
    ].copy()

    expected_counts["museum"] = clean_text(
        expected_counts["museum"]
    )

    expected_counts["source_label"] = clean_text(
        expected_counts["source_label"]
    )

    comparison = expected_counts.merge(
        actual_counts,
        on=["museum", "source_label"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    mismatched = comparison[
        comparison["_merge"].ne("both")
        | comparison["record_count"]
        .fillna(-1)
        .ne(
            comparison["actual_record_count"].fillna(-1)
        )
    ]

    if not mismatched.empty:
        raise ValueError(
            "Los conteos del mapeo no coinciden con el corpus:\n"
            f"{mismatched.to_string(index=False)}"
        )


def derive_canonical_fields(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Deriva cultura canónica y número de componentes."""

    enriched = dataframe.copy()

    components = clean_text(
        enriched["culture_components"]
    )

    component_count = components.apply(
        lambda value: (
            0
            if not value
            else len(
                [
                    component.strip()
                    for component in value.split("|")
                    if component.strip()
                ]
            )
        )
    )

    enriched["culture_component_count"] = (
        component_count.astype(int)
    )

    # culture_canonical solo se completa cuando existe exactamente
    # un componente. Las atribuciones compuestas se conservan únicamente
    # en culture_components para evitar una reducción arbitraria.
    enriched["culture_canonical"] = components.where(
        enriched["culture_component_count"].eq(1),
        "",
    )

    return enriched


def build_culture_enriched_corpus(
    corpus: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Construye el corpus con normalización cultural trazable."""

    validate_corpus(corpus)
    validate_mapping(mapping)
    validate_mapping_counts(corpus, mapping)

    base = corpus.copy()

    base["_original_row_order"] = range(len(base))
    base["museum"] = clean_text(base["museum"])

    # La columna original culture no se modifica.
    base["culture_source"] = clean_text(base["culture"])

    mapping_for_merge = mapping[
        [
            "museum",
            "source_label",
            "final_canonical_components",
            "attribution_type",
            "review_decision",
            "decision_basis",
            "strict_ground_truth_eligible",
            "review_status",
            "review_note",
        ]
    ].copy()

    # Las columnas del mapeo se renombran antes de la unión.
    # Esto evita colisiones con columnas ya existentes en el corpus,
    # como review_status.
    mapping_for_merge = mapping_for_merge.rename(
        columns={
            "final_canonical_components": (
                "culture_components"
            ),
            "attribution_type": (
                "culture_attribution_type"
            ),
            "review_decision": (
                "culture_mapping_decision"
            ),
            "decision_basis": (
                "culture_mapping_basis"
            ),
            "strict_ground_truth_eligible": (
                "culture_strict_ground_truth_eligible"
            ),
            "review_status": (
                "culture_mapping_status"
            ),
            "review_note": (
                "culture_mapping_note"
            ),
        }
    )

    mapping_for_merge["museum"] = clean_text(
        mapping_for_merge["museum"]
    )

    mapping_for_merge["source_label"] = clean_text(
        mapping_for_merge["source_label"]
    )

    mapping_for_merge[
        "culture_strict_ground_truth_eligible"
    ] = parse_boolean_series(
        mapping_for_merge[
            "culture_strict_ground_truth_eligible"
        ]
    )

    enriched = base.merge(
        mapping_for_merge,
        left_on=["museum", "culture_source"],
        right_on=["museum", "source_label"],
        how="left",
        validate="many_to_one",
        sort=False,
        indicator=True,
    )

    unmatched = enriched["_merge"].ne("both")

    if unmatched.any():
        missing_keys = (
            enriched.loc[
                unmatched,
                [
                    "museum",
                    "culture_source",
                ],
            ]
            .drop_duplicates()
            .sort_values(
                ["museum", "culture_source"]
            )
        )

        raise ValueError(
            "Existen etiquetas culturales sin mapeo:\n"
            f"{missing_keys.to_string(index=False)}"
        )

    enriched["culture_components"] = clean_text(
        enriched["culture_components"]
    )

    enriched["culture_mapping_version"] = (
        MAPPING_VERSION
    )

    enriched = derive_canonical_fields(enriched)

    enriched = (
        enriched.sort_values(
            "_original_row_order",
            kind="stable",
        )
        .drop(
            columns=[
                "_original_row_order",
                "source_label",
                "_merge",
            ]
        )
        .reset_index(drop=True)
    )

    # Reubicar las columnas culturales al final y mantener
    # intacto el orden de las columnas originales.
    original_columns = list(corpus.columns)

    enriched = enriched[
        original_columns + CULTURE_OUTPUT_COLUMNS
    ]

    if len(enriched) != len(corpus):
        raise ValueError(
            "La aplicación del mapeo modificó el número "
            "de registros."
        )

    if not enriched["item_id"].equals(
        corpus["item_id"].reset_index(drop=True)
    ):
        raise ValueError(
            "La aplicación del mapeo alteró el orden "
            "de los registros."
        )

    return enriched


def build_summary(
    enriched: pd.DataFrame,
) -> pd.DataFrame:
    """Construye el resumen de cobertura cultural."""

    summary = (
        enriched.groupby(
            [
                "museum",
                "culture_mapping_decision",
                "culture_strict_ground_truth_eligible",
            ],
            dropna=False,
        )
        .agg(
            records=("item_id", "size"),
            source_categories=(
                "culture_source",
                "nunique",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "museum",
                "culture_mapping_decision",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return summary


def print_summary(
    enriched: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Imprime el resumen operativo."""

    print("\nCORPUS CON NORMALIZACIÓN CULTURAL")
    print("=" * 100)
    print(summary.to_string(index=False))

    decision_distribution = (
        enriched[
            "culture_mapping_decision"
        ]
        .value_counts()
        .sort_index()
    )

    print("\nDISTRIBUCIÓN GENERAL")
    print("=" * 100)
    print(decision_distribution.to_string())

    strict_records = int(
        enriched[
            "culture_strict_ground_truth_eligible"
        ].sum()
    )

    print("\nTOTALES")
    print("=" * 100)
    print(f"Registros: {len(enriched)}")
    print(f"Columnas: {len(enriched.columns)}")
    print(
        "Registros elegibles para ground truth "
        f"cultural estricto: {strict_records}"
    )
    print(
        "CSV: "
        f"{OUTPUT_CSV_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Parquet: "
        f"{OUTPUT_PARQUET_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Resumen: "
        f"{SUMMARY_PATH.relative_to(PROJECT_ROOT)}"
    )


def main() -> None:
    """Ejecuta la aplicación del mapeo cultural."""

    if not INPUT_CORPUS_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el corpus enriquecido: "
            f"{INPUT_CORPUS_PATH}"
        )

    if not CURATED_MAPPING_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el mapeo cultural curado: "
            f"{CURATED_MAPPING_PATH}"
        )

    corpus = pd.read_csv(INPUT_CORPUS_PATH)
    mapping = pd.read_csv(CURATED_MAPPING_PATH)

    enriched = build_culture_enriched_corpus(
        corpus,
        mapping,
    )

    summary = build_summary(enriched)

    OUTPUT_CSV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PARQUET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    enriched.to_csv(
        OUTPUT_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )

    enriched.to_parquet(
        OUTPUT_PARQUET_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )

    print_summary(enriched, summary)


if __name__ == "__main__":
    main()