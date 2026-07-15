from __future__ import annotations

from pathlib import Path

from caep.datasets.build_enriched_corpus import (
    build_enriched_corpus,
)
from caep.evaluation.category_audit import (
    audit_categories,
    load_audit_config,
)


SUMMARY_OUTPUT = Path(
    "outputs/reports/category_audit_summary_v1.csv"
)
DETAIL_OUTPUT = Path(
    "outputs/reports/category_audit_detail_v1.csv"
)
MARKDOWN_OUTPUT = Path(
    "outputs/reports/category_audit_report_v1.md"
)


def build_markdown_report(
    summary,
    detail,
) -> str:
    lines = [
        "# Auditoría de categorías para ground truth",
        "",
        "## Propósito",
        "",
        (
            "Evaluar frecuencia, cobertura y posible dependencia "
            "institucional de las categorías curatoriales."
        ),
        "",
        (
            "La normalización aplicada en esta etapa es únicamente "
            "lexical. No se fusionan sinónimos ni categorías "
            "semánticamente próximas."
        ),
        "",
        "## Resumen por campo",
        "",
        (
            "| Campo | Rol | Cobertura | Categorías | "
            "Candidatas globales | Cobertura candidata |"
        ),
        "|---|---|---:|---:|---:|---:|",
    ]

    for row in summary.itertuples(index=False):
        lines.append(
            f"| `{row.field}` | {row.field_role} | "
            f"{row.coverage:.1%} | "
            f"{row.unique_categories} | "
            f"{row.candidate_categories} | "
            f"{row.candidate_record_coverage:.1%} |"
        )

    for field in summary["field"]:
        field_detail = detail.loc[
            detail["field"].eq(field)
        ].head(20)

        lines.extend(
            [
                "",
                f"## Categorías principales: `{field}`",
                "",
                (
                    "| Categoría | Total | MET | CMA | "
                    "Dominancia | Estado |"
                ),
                "|---|---:|---:|---:|---:|---|",
            ]
        )

        for row in field_detail.itertuples(index=False):
            label = str(row.category_label).replace("|", r"\|")

            lines.append(
                f"| {label} | {row.total_count} | "
                f"{row.met_count} | {row.cma_count} | "
                f"{row.museum_dominance:.1%} | "
                f"`{row.status}` |"
            )

    lines.extend(
        [
            "",
            "## Interpretación de estados",
            "",
            "- `candidate_global`: cumple los umbrales preliminares.",
            "- `rare`: soporte total insuficiente.",
            "- `source_specific`: aparece en un solo museo.",
            (
                "- `insufficient_cross_museum_support`: aparece en "
                "ambos museos, pero con soporte mínimo insuficiente."
            ),
            (
                "- `museum_imbalanced`: existe en ambos museos, "
                "pero está excesivamente concentrada en uno."
            ),
            (
                "- `diagnostic_only`: campo incompleto que no se "
                "considera candidato global en esta etapa."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    corpus = build_enriched_corpus()
    config = load_audit_config()

    summary, detail = audit_categories(
        corpus,
        config,
    )

    SUMMARY_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )
    detail.to_csv(
        DETAIL_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    markdown = build_markdown_report(
        summary,
        detail,
    )

    MARKDOWN_OUTPUT.write_text(
        markdown,
        encoding="utf-8",
    )

    print("\nRESUMEN")
    print("=" * 100)
    print(summary.to_string(index=False))

    candidates = detail.loc[
        detail["status"].eq("candidate_global")
    ]

    print("\nCATEGORÍAS CANDIDATAS GLOBALES")
    print("=" * 100)

    if candidates.empty:
        print("No se identificaron categorías candidatas.")
    else:
        print(
            candidates[
                [
                    "field",
                    "category_label",
                    "total_count",
                    "met_count",
                    "cma_count",
                    "museum_dominance",
                ]
            ].to_string(index=False)
        )

    print(f"\nResumen: {SUMMARY_OUTPUT}")
    print(f"Detalle: {DETAIL_OUTPUT}")
    print(f"Informe: {MARKDOWN_OUTPUT}")


if __name__ == "__main__":
    main()