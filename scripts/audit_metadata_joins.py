from __future__ import annotations

from pathlib import Path

import pandas as pd


MET_REVIEWED_PATH = Path(
    "data/source/met/met_corpus_principal_v2_revisado.csv"
)
MET_INVENTORY_PATH = Path(
    "data/source/met/met_inventario_base.csv"
)

CMA_REVIEWED_PATH = Path(
    "data/source/cma/cma_corpus_principal_revisado.csv"
)
CMA_CANDIDATES_PATH = Path(
    "data/source/cma/cma_andes_textiles_candidates.csv"
)

OUTPUT_PATH = Path(
    "outputs/reports/metadata_join_audit_v1.csv"
)


def normalize_key(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
    )


def audit_join(
    museum: str,
    reviewed: pd.DataFrame,
    enrichment: pd.DataFrame,
    reviewed_key: str,
    enrichment_key: str,
) -> list[dict[str, object]]:
    reviewed_keys = normalize_key(reviewed[reviewed_key])
    enrichment_keys = normalize_key(enrichment[enrichment_key])

    reviewed_non_empty = reviewed_keys[reviewed_keys.ne("")]
    enrichment_non_empty = enrichment_keys[enrichment_keys.ne("")]

    reviewed_duplicates = int(reviewed_non_empty.duplicated().sum())
    enrichment_duplicates = int(enrichment_non_empty.duplicated().sum())

    reviewed_set = set(reviewed_non_empty)
    enrichment_set = set(enrichment_non_empty)

    matched = reviewed_set.intersection(enrichment_set)
    missing = reviewed_set.difference(enrichment_set)
    unused = enrichment_set.difference(reviewed_set)

    total_reviewed = len(reviewed)

    return [
        {
            "museum": museum,
            "metric": "reviewed_rows",
            "value": total_reviewed,
        },
        {
            "museum": museum,
            "metric": "enrichment_rows",
            "value": len(enrichment),
        },
        {
            "museum": museum,
            "metric": "reviewed_empty_keys",
            "value": int(reviewed_keys.eq("").sum()),
        },
        {
            "museum": museum,
            "metric": "enrichment_empty_keys",
            "value": int(enrichment_keys.eq("").sum()),
        },
        {
            "museum": museum,
            "metric": "reviewed_duplicate_keys",
            "value": reviewed_duplicates,
        },
        {
            "museum": museum,
            "metric": "enrichment_duplicate_keys",
            "value": enrichment_duplicates,
        },
        {
            "museum": museum,
            "metric": "matched_reviewed_keys",
            "value": len(matched),
        },
        {
            "museum": museum,
            "metric": "missing_reviewed_keys",
            "value": len(missing),
        },
        {
            "museum": museum,
            "metric": "match_rate",
            "value": len(matched) / total_reviewed if total_reviewed else 0,
        },
        {
            "museum": museum,
            "metric": "unused_enrichment_keys",
            "value": len(unused),
        },
    ]


def print_missing_keys(
    museum: str,
    reviewed: pd.DataFrame,
    enrichment: pd.DataFrame,
    reviewed_key: str,
    enrichment_key: str,
) -> None:
    reviewed_keys = set(
        normalize_key(reviewed[reviewed_key]).loc[
            lambda values: values.ne("")
        ]
    )
    enrichment_keys = set(
        normalize_key(enrichment[enrichment_key]).loc[
            lambda values: values.ne("")
        ]
    )

    missing = sorted(reviewed_keys.difference(enrichment_keys))

    print(f"\n{museum} — claves revisadas sin correspondencia:")
    if missing:
        for key in missing[:25]:
            print(f"- {key}")
    else:
        print("- Ninguna")


def main() -> None:
    met_reviewed = pd.read_csv(
        MET_REVIEWED_PATH,
        dtype=str,
        keep_default_na=False,
    )
    met_inventory = pd.read_csv(
        MET_INVENTORY_PATH,
        dtype=str,
        keep_default_na=False,
    )
    cma_reviewed = pd.read_csv(
        CMA_REVIEWED_PATH,
        dtype=str,
        keep_default_na=False,
    )
    cma_candidates = pd.read_csv(
        CMA_CANDIDATES_PATH,
        dtype=str,
        keep_default_na=False,
    )

    rows = []

    rows.extend(
        audit_join(
            museum="MET",
            reviewed=met_reviewed,
            enrichment=met_inventory,
            reviewed_key="id_fuente",
            enrichment_key="id_objeto",
        )
    )

    rows.extend(
        audit_join(
            museum="CMA",
            reviewed=cma_reviewed,
            enrichment=cma_candidates,
            reviewed_key="id_objeto",
            enrichment_key="source_id",
        )
    )

    report = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(report.to_string(index=False))

    print_missing_keys(
        museum="MET",
        reviewed=met_reviewed,
        enrichment=met_inventory,
        reviewed_key="id_fuente",
        enrichment_key="id_objeto",
    )

    print_missing_keys(
        museum="CMA",
        reviewed=cma_reviewed,
        enrichment=cma_candidates,
        reviewed_key="id_objeto",
        enrichment_key="source_id",
    )

    print(f"\nReporte: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()