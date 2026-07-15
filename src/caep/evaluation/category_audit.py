from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "ground_truth_audit_v1.yaml"
)


def clean_raw_label(value: object) -> str:
    """Limpia espacios sin reinterpretar semánticamente una categoría."""
    if value is None:
        return ""

    text = unicodedata.normalize("NFKC", str(value)).strip()

    if not text or text.casefold() in {"nan", "none", "null"}:
        return ""

    return re.sub(r"\s+", " ", text)


def normalize_label(value: object) -> str:
    """
    Construye una clave lexical para auditar categorías.

    Solo normaliza Unicode, espacios y mayúsculas/minúsculas.
    No fusiona sinónimos ni categorías curatoriales.
    """
    return clean_raw_label(value).casefold()


def load_audit_config(
    path: Path = DEFAULT_CONFIG_PATH,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe la configuración de auditoría: {path}"
        )

    with path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or "audit" not in config:
        raise ValueError(
            "La configuración debe contener la sección 'audit'."
        )

    return config


def choose_representative_label(
    values: pd.Series,
) -> str:
    """
    Escoge la forma textual más frecuente.

    En caso de empate, usa orden lexicográfico para mantener
    determinismo.
    """
    counts = values.value_counts()
    maximum = counts.max()

    candidates = sorted(
        counts[counts.eq(maximum)].index.tolist()
    )

    return str(candidates[0])


def classify_category(
    *,
    field_role: str,
    total_count: int,
    museum_counts: dict[str, int],
    thresholds: dict,
) -> str:
    if field_role == "diagnostic":
        return "diagnostic_only"

    if total_count < int(thresholds["min_total_support"]):
        return "rare"

    represented_counts = [
        count
        for count in museum_counts.values()
        if count > 0
    ]

    if len(represented_counts) < int(thresholds["min_museums"]):
        return "source_specific"

    if min(represented_counts) < int(
        thresholds["min_support_per_museum"]
    ):
        return "insufficient_cross_museum_support"

    museum_dominance = max(represented_counts) / total_count

    if museum_dominance > float(
        thresholds["max_museum_dominance"]
    ):
        return "museum_imbalanced"

    return "candidate_global"


def audit_categories(
    corpus: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_config = config["audit"]

    core_fields = list(audit_config["core_fields"])
    diagnostic_fields = list(
        audit_config["diagnostic_fields"]
    )
    expected_museums = list(
        audit_config["expected_museums"]
    )
    thresholds = audit_config["thresholds"]

    fields = core_fields + diagnostic_fields

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for field in fields:
        if field not in corpus.columns:
            raise ValueError(
                f"El corpus no contiene el campo auditado: {field}"
            )

        field_role = (
            "core"
            if field in core_fields
            else "diagnostic"
        )

        working = corpus[["museum", field]].copy()

        working["category_raw"] = working[field].map(
            clean_raw_label
        )
        working["category_key"] = working[field].map(
            normalize_label
        )

        non_empty = working.loc[
            working["category_key"].ne("")
        ].copy()

        representatives = (
            non_empty
            .groupby("category_key")["category_raw"]
            .apply(choose_representative_label)
            .to_dict()
        )

        category_museum_counts = (
            non_empty
            .groupby(["category_key", "museum"])
            .size()
            .unstack(fill_value=0)
        )

        category_status: dict[str, str] = {}

        for category_key, counts_row in (
            category_museum_counts.iterrows()
        ):
            museum_counts = {
                museum: int(counts_row.get(museum, 0))
                for museum in expected_museums
            }

            total_count = sum(museum_counts.values())
            museum_count = sum(
                count > 0
                for count in museum_counts.values()
            )

            dominant_museum = max(
                expected_museums,
                key=lambda museum: (
                    museum_counts[museum],
                    museum,
                ),
            )

            museum_dominance = (
                museum_counts[dominant_museum] / total_count
                if total_count
                else 0.0
            )

            status = classify_category(
                field_role=field_role,
                total_count=total_count,
                museum_counts=museum_counts,
                thresholds=thresholds,
            )

            category_status[category_key] = status

            detail_rows.append(
                {
                    "field": field,
                    "field_role": field_role,
                    "category_key": category_key,
                    "category_label": representatives[
                        category_key
                    ],
                    "total_count": total_count,
                    "met_count": museum_counts.get("MET", 0),
                    "cma_count": museum_counts.get("CMA", 0),
                    "museum_count": museum_count,
                    "dominant_museum": dominant_museum,
                    "museum_dominance": museum_dominance,
                    "status": status,
                }
            )

        candidate_keys = {
            category_key
            for category_key, status
            in category_status.items()
            if status == "candidate_global"
        }

        candidate_records = int(
            non_empty["category_key"]
            .isin(candidate_keys)
            .sum()
        )

        total_rows = len(corpus)
        non_empty_count = len(non_empty)

        summary_rows.append(
            {
                "field": field,
                "field_role": field_role,
                "total_rows": total_rows,
                "non_empty": non_empty_count,
                "missing": total_rows - non_empty_count,
                "coverage": (
                    non_empty_count / total_rows
                    if total_rows
                    else 0.0
                ),
                "unique_categories": int(
                    non_empty["category_key"].nunique()
                ),
                "candidate_categories": len(candidate_keys),
                "candidate_records": candidate_records,
                "candidate_record_coverage": (
                    candidate_records / total_rows
                    if total_rows
                    else 0.0
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    detail = pd.DataFrame(detail_rows)

    status_order = {
        "candidate_global": 0,
        "museum_imbalanced": 1,
        "insufficient_cross_museum_support": 2,
        "source_specific": 3,
        "rare": 4,
        "diagnostic_only": 5,
    }

    if not detail.empty:
        detail["_status_order"] = detail["status"].map(
            status_order
        )

        detail = (
            detail
            .sort_values(
                [
                    "field",
                    "_status_order",
                    "total_count",
                    "category_label",
                ],
                ascending=[True, True, False, True],
                kind="stable",
            )
            .drop(columns="_status_order")
            .reset_index(drop=True)
        )

    return summary, detail