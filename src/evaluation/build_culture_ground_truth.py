"""Construye el ground truth cultural estricto para recuperación.

Se consideran relevantes dos registros cuando:

1. ambos tienen una atribución cultural directa;
2. ambos son elegibles para ground truth estricto;
3. comparten exactamente la misma cultura canónica;
4. no corresponden al mismo objeto.

Este criterio es una señal de relevancia controlada y no una definición
universal de semejanza visual o semántica.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paper_corpus_culture_enriched_v1.csv"
)

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "culture_ground_truth_strict_v1.csv"
)

RELEVANCE_PAIRS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "culture_relevance_pairs_strict_v1.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "culture_ground_truth_summary_v1.csv"
)

GROUND_TRUTH_VERSION = "culture_ground_truth_strict_v1"
GROUND_TRUTH_CRITERION = "same_strict_canonical_culture"


REQUIRED_COLUMNS = {
    "item_id",
    "museum",
    "culture",
    "culture_source",
    "culture_canonical",
    "culture_component_count",
    "culture_attribution_type",
    "culture_mapping_decision",
    "culture_strict_ground_truth_eligible",
}


GROUND_TRUTH_COLUMNS = [
    "item_id",
    "museum",
    "culture_source",
    "culture_canonical",
    "culture_group_size",
    "relevant_item_count",
    "query_eligible",
    "ground_truth_criterion",
    "ground_truth_version",
]


PAIR_COLUMNS = [
    "query_item_id",
    "relevant_item_id",
    "culture_canonical",
    "ground_truth_criterion",
    "ground_truth_version",
]


def clean_text(series: pd.Series) -> pd.Series:
    """Convierte nulos a texto vacío y elimina espacios externos."""

    return (
        series.fillna("")
        .astype(str)
        .str.strip()
    )


def parse_boolean_series(series: pd.Series) -> pd.Series:
    """Convierte valores textuales o booleanos a bool."""

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
        invalid = sorted(
            normalized[parsed.isna()].unique().tolist()
        )

        raise ValueError(
            "Valores booleanos no reconocidos en "
            "culture_strict_ground_truth_eligible: "
            f"{invalid}"
        )

    return parsed.astype(bool)


def validate_corpus(corpus: pd.DataFrame) -> None:
    """Valida las condiciones necesarias para construir el ground truth."""

    missing = REQUIRED_COLUMNS.difference(corpus.columns)

    if missing:
        raise ValueError(
            "Faltan columnas requeridas en el corpus: "
            f"{sorted(missing)}"
        )

    if corpus["item_id"].duplicated().any():
        duplicated = (
            corpus.loc[
                corpus["item_id"].duplicated(keep=False),
                "item_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "El corpus contiene item_id duplicados: "
            f"{duplicated[:10]}"
        )

    eligible = parse_boolean_series(
        corpus["culture_strict_ground_truth_eligible"]
    )

    strict_decision = clean_text(
        corpus["culture_mapping_decision"]
    ).eq("include_strict")

    if not eligible.equals(strict_decision):
        inconsistent = corpus.loc[
            eligible.ne(strict_decision),
            [
                "item_id",
                "culture_mapping_decision",
                "culture_strict_ground_truth_eligible",
            ],
        ]

        raise ValueError(
            "La elegibilidad estricta no coincide con "
            "culture_mapping_decision:\n"
            f"{inconsistent.to_string(index=False)}"
        )

    strict = corpus.loc[eligible].copy()

    if strict.empty:
        raise ValueError(
            "No existen registros culturales estrictos."
        )

    if not clean_text(
        strict["culture_attribution_type"]
    ).eq("direct").all():
        raise ValueError(
            "Todos los registros estrictos deben tener "
            "culture_attribution_type='direct'."
        )

    if not pd.to_numeric(
        strict["culture_component_count"],
        errors="coerce",
    ).eq(1).all():
        raise ValueError(
            "Todos los registros estrictos deben tener "
            "exactamente un componente cultural."
        )

    if clean_text(
        strict["culture_canonical"]
    ).eq("").any():
        raise ValueError(
            "Existen registros estrictos sin cultura canónica."
        )


def build_ground_truth_table(
    corpus: pd.DataFrame,
) -> pd.DataFrame:
    """Construye la tabla de etiquetas culturales estrictas."""

    validate_corpus(corpus)

    working = corpus.copy()

    working["_corpus_order"] = range(len(working))

    eligible = parse_boolean_series(
        working["culture_strict_ground_truth_eligible"]
    )

    strict = working.loc[
        eligible,
        [
            "item_id",
            "museum",
            "culture_source",
            "culture_canonical",
            "_corpus_order",
        ],
    ].copy()

    strict["item_id"] = clean_text(strict["item_id"])
    strict["museum"] = clean_text(strict["museum"])
    strict["culture_source"] = clean_text(
        strict["culture_source"]
    )
    strict["culture_canonical"] = clean_text(
        strict["culture_canonical"]
    )

    strict["culture_group_size"] = (
        strict.groupby(
            "culture_canonical"
        )["item_id"]
        .transform("size")
        .astype(int)
    )

    strict["relevant_item_count"] = (
        strict["culture_group_size"] - 1
    ).astype(int)

    strict["query_eligible"] = (
        strict["relevant_item_count"] > 0
    )

    strict["ground_truth_criterion"] = (
        GROUND_TRUTH_CRITERION
    )

    strict["ground_truth_version"] = (
        GROUND_TRUTH_VERSION
    )

    strict = (
        strict.sort_values(
            "_corpus_order",
            kind="stable",
        )
        .drop(columns="_corpus_order")
        .reset_index(drop=True)
    )

    return strict[GROUND_TRUTH_COLUMNS]


def build_relevance_pairs(
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    """Construye pares dirigidos consulta-relevante."""

    query_side = ground_truth[
        [
            "item_id",
            "culture_canonical",
        ]
    ].copy()

    query_side = query_side.rename(
        columns={
            "item_id": "query_item_id",
        }
    )

    query_side["_query_order"] = range(
        len(query_side)
    )

    relevant_side = ground_truth[
        [
            "item_id",
            "culture_canonical",
        ]
    ].copy()

    relevant_side = relevant_side.rename(
        columns={
            "item_id": "relevant_item_id",
        }
    )

    relevant_side["_relevant_order"] = range(
        len(relevant_side)
    )

    pairs = query_side.merge(
        relevant_side,
        on="culture_canonical",
        how="inner",
        validate="many_to_many",
    )

    pairs = pairs[
        pairs["query_item_id"].ne(
            pairs["relevant_item_id"]
        )
    ].copy()

    pairs["ground_truth_criterion"] = (
        GROUND_TRUTH_CRITERION
    )

    pairs["ground_truth_version"] = (
        GROUND_TRUTH_VERSION
    )

    pairs = (
        pairs.sort_values(
            [
                "_query_order",
                "_relevant_order",
            ],
            kind="stable",
        )
        .drop(
            columns=[
                "_query_order",
                "_relevant_order",
            ]
        )
        .reset_index(drop=True)
    )

    return pairs[PAIR_COLUMNS]


def build_summary(
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    """Resume la cobertura por cultura canónica."""

    summary = (
        ground_truth.groupby(
            "culture_canonical",
            dropna=False,
        )
        .agg(
            records=("item_id", "size"),
            museums=("museum", "nunique"),
            source_labels=("culture_source", "nunique"),
            query_eligible_records=(
                "query_eligible",
                "sum",
            ),
        )
        .reset_index()
    )

    summary["records"] = (
        summary["records"].astype(int)
    )

    summary["query_eligible_records"] = (
        summary["query_eligible_records"].astype(int)
    )

    summary["directed_relevance_pairs"] = (
        summary["records"]
        * (summary["records"] - 1)
    ).astype(int)

    summary = summary.sort_values(
        by=[
            "records",
            "culture_canonical",
        ],
        ascending=[
            False,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)

    return summary


def validate_outputs(
    ground_truth: pd.DataFrame,
    pairs: pd.DataFrame,
) -> None:
    """Valida consistencia entre etiquetas y pares."""

    if ground_truth["item_id"].duplicated().any():
        raise ValueError(
            "El ground truth contiene item_id duplicados."
        )

    if pairs.duplicated(
        subset=[
            "query_item_id",
            "relevant_item_id",
        ]
    ).any():
        raise ValueError(
            "La tabla de relevancia contiene pares duplicados."
        )

    if pairs["query_item_id"].eq(
        pairs["relevant_item_id"]
    ).any():
        raise ValueError(
            "La tabla de relevancia contiene pares autorreferentes."
        )

    expected_pairs = int(
        (
            ground_truth["culture_group_size"]
            * (
                ground_truth["culture_group_size"]
                - 1
            )
        ).groupby(
            ground_truth["culture_canonical"]
        ).first().sum()
    )

    if len(pairs) != expected_pairs:
        raise ValueError(
            "El número de pares de relevancia no coincide "
            "con el esperado: "
            f"{len(pairs)} != {expected_pairs}"
        )


def build_culture_ground_truth(
    corpus: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construye todos los artefactos del ground truth."""

    ground_truth = build_ground_truth_table(corpus)

    pairs = build_relevance_pairs(ground_truth)

    summary = build_summary(ground_truth)

    validate_outputs(
        ground_truth,
        pairs,
    )

    return ground_truth, pairs, summary


def print_summary(
    ground_truth: pd.DataFrame,
    pairs: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Imprime el resumen del ground truth."""

    print("\nGROUND TRUTH CULTURAL ESTRICTO")
    print("=" * 100)
    print(summary.to_string(index=False))

    print("\nTOTALES")
    print("=" * 100)
    print(
        f"Registros estrictos: {len(ground_truth)}"
    )
    print(
        "Consultas con al menos un relevante: "
        f"{int(ground_truth['query_eligible'].sum())}"
    )
    print(
        "Registros sin otro relevante de su cultura: "
        f"{int((~ground_truth['query_eligible']).sum())}"
    )
    print(
        f"Pares dirigidos de relevancia: {len(pairs)}"
    )
    print(
        "Criterio: "
        f"{GROUND_TRUTH_CRITERION}"
    )
    print(
        "Ground truth: "
        f"{GROUND_TRUTH_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Pares: "
        f"{RELEVANCE_PAIRS_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Resumen: "
        f"{SUMMARY_PATH.relative_to(PROJECT_ROOT)}"
    )


def main() -> None:
    """Ejecuta la construcción del ground truth."""

    if not INPUT_CORPUS_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el corpus culturalmente "
            f"enriquecido: {INPUT_CORPUS_PATH}"
        )

    corpus = pd.read_csv(INPUT_CORPUS_PATH)

    ground_truth, pairs, summary = (
        build_culture_ground_truth(corpus)
    )

    GROUND_TRUTH_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RELEVANCE_PAIRS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ground_truth.to_csv(
        GROUND_TRUTH_PATH,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )

    pairs.to_csv(
        RELEVANCE_PAIRS_PATH,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )

    print_summary(
        ground_truth,
        pairs,
        summary,
    )


if __name__ == "__main__":
    main()