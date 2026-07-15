from __future__ import annotations

from pathlib import Path

import pandas as pd


DETAIL_PATH = Path(
    "outputs/reports/category_audit_detail_v1.csv"
)

TEMPLATE_PATH = Path(
    "config/mappings/ground_truth_mapping_v1.csv"
)

SUMMARY_PATH = Path(
    "outputs/reports/category_harmonization_summary_v1.csv"
)

OVERLAP_PATH = Path(
    "outputs/reports/category_exact_overlap_v1.csv"
)

CORE_FIELDS = [
    "classification",
    "object_type",
    "culture",
    "period",
]

TEMPLATE_COLUMNS = [
    "field",
    "category_key",
    "category_label",
    "total_count",
    "met_count",
    "cma_count",
    "current_status",
    "canonical_label",
    "canonical_level",
    "mapping_decision",
    "mapping_rationale",
    "reviewer",
    "review_date",
]


def load_detail() -> pd.DataFrame:
    if not DETAIL_PATH.exists():
        raise FileNotFoundError(
            f"No existe el detalle de auditoría: {DETAIL_PATH}"
        )

    frame = pd.read_csv(
        DETAIL_PATH,
        dtype={
            "field": str,
            "field_role": str,
            "category_key": str,
            "category_label": str,
            "status": str,
        },
        keep_default_na=False,
    )

    required_columns = {
        "field",
        "category_key",
        "category_label",
        "total_count",
        "met_count",
        "cma_count",
        "status",
    }

    missing = required_columns.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas en el detalle: {sorted(missing)}"
        )

    return frame


def build_mapping_template(
    detail: pd.DataFrame,
) -> pd.DataFrame:
    template = (
        detail.loc[
            detail["field"].isin(CORE_FIELDS),
            [
                "field",
                "category_key",
                "category_label",
                "total_count",
                "met_count",
                "cma_count",
                "status",
            ],
        ]
        .rename(
            columns={
                "status": "current_status",
            }
        )
        .copy()
    )

    template["canonical_label"] = ""
    template["canonical_level"] = ""
    template["mapping_decision"] = "pendiente"
    template["mapping_rationale"] = ""
    template["reviewer"] = ""
    template["review_date"] = ""

    template = template[TEMPLATE_COLUMNS]

    field_order = {
        field: index
        for index, field in enumerate(CORE_FIELDS)
    }

    template["_field_order"] = template["field"].map(
        field_order
    )

    template = (
        template
        .sort_values(
            [
                "_field_order",
                "total_count",
                "category_label",
            ],
            ascending=[True, False, True],
            kind="stable",
        )
        .drop(columns="_field_order")
        .reset_index(drop=True)
    )

    return template


def build_summary(
    template: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for field in CORE_FIELDS:
        subset = template.loc[
            template["field"].eq(field)
        ]

        exact_overlap = subset.loc[
            subset["met_count"].gt(0)
            & subset["cma_count"].gt(0)
        ]

        met_only = subset.loc[
            subset["met_count"].gt(0)
            & subset["cma_count"].eq(0)
        ]

        cma_only = subset.loc[
            subset["cma_count"].gt(0)
            & subset["met_count"].eq(0)
        ]

        rows.append(
            {
                "field": field,
                "categories_total": len(subset),
                "exact_overlap_categories": len(exact_overlap),
                "met_only_categories": len(met_only),
                "cma_only_categories": len(cma_only),
                "records_in_exact_overlap": int(
                    exact_overlap["total_count"].sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def print_field_inventory(
    template: pd.DataFrame,
    field: str,
) -> None:
    subset = template.loc[
        template["field"].eq(field)
    ].copy()

    print("\n" + "=" * 110)
    print(field.upper())
    print("=" * 110)

    display_columns = [
        "category_label",
        "total_count",
        "met_count",
        "cma_count",
        "current_status",
    ]

    print(
        subset[display_columns]
        .to_string(index=False)
    )


def main() -> None:
    detail = load_detail()
    template = build_mapping_template(detail)
    summary = build_summary(template)

    exact_overlap = template.loc[
        template["met_count"].gt(0)
        & template["cma_count"].gt(0)
    ].copy()

    TEMPLATE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    template.to_csv(
        TEMPLATE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    exact_overlap.to_csv(
        OVERLAP_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nRESUMEN DE ARMONIZACIÓN")
    print("=" * 110)
    print(summary.to_string(index=False))

    print_field_inventory(
        template,
        "classification",
    )

    print_field_inventory(
        template,
        "object_type",
    )

    print("\nARCHIVOS GENERADOS")
    print("=" * 110)
    print(f"Plantilla: {TEMPLATE_PATH}")
    print(f"Resumen: {SUMMARY_PATH}")
    print(f"Solapamiento exacto: {OVERLAP_PATH}")


if __name__ == "__main__":
    main()