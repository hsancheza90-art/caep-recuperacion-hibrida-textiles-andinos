"""Prepara la tabla de revisión curatorial del mapeo cultural.

Este módulo no modifica el corpus y no convierte automáticamente las
propuestas en decisiones curatoriales definitivas. Su finalidad es crear
una tabla trazable para revisión manual.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROPOSALS_PATH = (
    PROJECT_ROOT
    / "config"
    / "mappings"
    / "culture_mapping_proposals_v1.csv"
)

REVIEW_PATH = (
    PROJECT_ROOT
    / "config"
    / "mappings"
    / "culture_mapping_review_v1.csv"
)


REQUIRED_COLUMNS = {
    "museum",
    "source_label",
    "canonical_components",
    "record_count",
    "proposal_decision",
}


PROPOSAL_DEFAULTS: dict[str, dict[str, object]] = {
    "propuesta_aceptable": {
        "review_decision": "include_strict",
        "attribution_type": "direct",
        "strict_ground_truth_eligible": True,
        "review_note": (
            "Propuesta automática clara. Requiere confirmación "
            "curatorial antes de aplicarse al corpus."
        ),
    },
    "excluir_no_atribuida": {
        "review_decision": "exclude",
        "attribution_type": "unattributed",
        "strict_ground_truth_eligible": False,
        "review_note": (
            "La etiqueta no identifica una cultura específica "
            "y no es válida para ground truth cultural."
        ),
    },
    "revisar_estilo": {
        "review_decision": "pending",
        "attribution_type": "style",
        "strict_ground_truth_eligible": False,
        "review_note": (
            "La atribución expresa estilo o afinidad estilística, "
            "no necesariamente pertenencia cultural directa."
        ),
    },
    "revisar_incierta": {
        "review_decision": "pending",
        "attribution_type": "uncertain",
        "strict_ground_truth_eligible": False,
        "review_note": (
            "La fuente contiene incertidumbre explícita, alternativas "
            "o signos de interrogación."
        ),
    },
    "revisar_compuesta": {
        "review_decision": "pending",
        "attribution_type": "composite",
        "strict_ground_truth_eligible": False,
        "review_note": (
            "La etiqueta contiene más de una atribución cultural "
            "y no debe reducirse automáticamente a una sola cultura."
        ),
    },
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
    "review_note",
    "reviewer",
    "review_date",
]


def validate_proposals(proposals: pd.DataFrame) -> None:
    """Valida la estructura mínima de las propuestas."""

    missing_columns = REQUIRED_COLUMNS.difference(proposals.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Faltan columnas requeridas en las propuestas: {missing_text}"
        )

    duplicated = proposals.duplicated(
        subset=["museum", "source_label"],
        keep=False,
    )

    if duplicated.any():
        duplicate_rows = proposals.loc[
            duplicated,
            ["museum", "source_label"],
        ]

        raise ValueError(
            "Existen pares museum/source_label duplicados:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    unknown_decisions = sorted(
        set(proposals["proposal_decision"].dropna())
        - set(PROPOSAL_DEFAULTS)
    )

    if unknown_decisions:
        raise ValueError(
            "Se encontraron decisiones de propuesta desconocidas: "
            f"{unknown_decisions}"
        )


def build_review_table(proposals: pd.DataFrame) -> pd.DataFrame:
    """Construye la tabla inicial para revisión curatorial."""

    validate_proposals(proposals)

    review = proposals.copy()

    defaults = pd.DataFrame(
        [
            PROPOSAL_DEFAULTS[decision]
            for decision in review["proposal_decision"]
        ],
        index=review.index,
    )

    review["review_decision"] = defaults["review_decision"]

    review["final_canonical_components"] = (
        review["canonical_components"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    exclusion_mask = review["review_decision"].eq("exclude")
    review.loc[
        exclusion_mask,
        "final_canonical_components",
    ] = ""

    review["attribution_type"] = defaults["attribution_type"]

    review["strict_ground_truth_eligible"] = (
        defaults["strict_ground_truth_eligible"].astype(bool)
    )

    # Aunque algunas propuestas estén prellenadas, todavía no representan
    # una aprobación curatorial definitiva.
    review["review_status"] = "pending_curatorial_review"

    review["review_note"] = defaults["review_note"]
    review["reviewer"] = ""
    review["review_date"] = ""

    review = review[OUTPUT_COLUMNS].sort_values(
        by=[
            "review_decision",
            "museum",
            "source_label",
        ],
        kind="stable",
    )

    return review.reset_index(drop=True)


def print_summary(review: pd.DataFrame) -> None:
    """Imprime un resumen operativo de la tabla generada."""

    summary = (
        review.groupby(
            [
                "review_decision",
                "attribution_type",
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

    print("\nREVISIÓN CULTURAL PREPARADA")
    print("=" * 100)
    print(summary.to_string(index=False))

    pending = review[
        review["review_decision"].eq("pending")
    ]

    print("\nCASOS PARA REVISIÓN MANUAL")
    print("=" * 100)

    if pending.empty:
        print("No existen casos pendientes.")
    else:
        print(
            pending[
                [
                    "museum",
                    "source_label",
                    "canonical_components",
                    "record_count",
                    "attribution_type",
                ]
            ].to_string(index=False)
        )

    print(f"\nArchivo generado: {REVIEW_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    """Ejecuta la preparación de la tabla de revisión."""

    if not PROPOSALS_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de propuestas: {PROPOSALS_PATH}"
        )

    proposals = pd.read_csv(PROPOSALS_PATH)

    review = build_review_table(proposals)

    REVIEW_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    review.to_csv(
        REVIEW_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print_summary(review)


if __name__ == "__main__":
    main()