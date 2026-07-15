"""Métricas reproducibles para recuperación de información.

El módulo evalúa rankings completos dentro del universo cultural estricto.
Las métricas implementadas son:

- Precision@K;
- Recall@K;
- nDCG@K con relevancia binaria;
- Reciprocal Rank y MRR.

El evaluador es independiente del modelo que produzca el ranking.
"""

from __future__ import annotations

from collections.abc import Iterable
from math import log2

import pandas as pd


DEFAULT_K_VALUES = (1, 5, 10)
EVALUATION_SCOPE = "strict_culture_universe"


REQUIRED_GROUND_TRUTH_COLUMNS = {
    "item_id",
    "culture_canonical",
    "relevant_item_count",
    "query_eligible",
}


REQUIRED_RANKING_COLUMNS = {
    "query_item_id",
    "candidate_item_id",
    "rank",
}


def clean_text(series: pd.Series) -> pd.Series:
    """Convierte nulos en texto vacío y elimina espacios externos."""

    return (
        series.fillna("")
        .astype(str)
        .str.strip()
    )


def parse_boolean_series(series: pd.Series) -> pd.Series:
    """Convierte representaciones comunes en booleanos reales."""

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


def normalize_k_values(
    k_values: Iterable[int],
    candidate_count: int,
) -> tuple[int, ...]:
    """Valida y normaliza los valores de K."""

    normalized: list[int] = []

    for value in k_values:
        try:
            integer_value = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"El valor de K no es entero: {value}"
            ) from error

        if integer_value != value:
            raise ValueError(
                f"El valor de K debe ser entero: {value}"
            )

        normalized.append(integer_value)

    normalized_tuple = tuple(
        sorted(set(normalized))
    )

    if not normalized_tuple:
        raise ValueError(
            "Debe proporcionarse al menos un valor de K."
        )

    if normalized_tuple[0] <= 0:
        raise ValueError(
            "Todos los valores de K deben ser mayores que cero."
        )

    if normalized_tuple[-1] > candidate_count:
        raise ValueError(
            "K no puede superar el número de candidatos: "
            f"{normalized_tuple[-1]} > {candidate_count}"
        )

    return normalized_tuple


def prepare_ground_truth(
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    """Valida y normaliza el ground truth cultural."""

    missing_columns = (
        REQUIRED_GROUND_TRUTH_COLUMNS
        .difference(ground_truth.columns)
    )

    if missing_columns:
        raise ValueError(
            "Faltan columnas en el ground truth: "
            f"{sorted(missing_columns)}"
        )

    prepared = ground_truth.copy()

    prepared["item_id"] = clean_text(
        prepared["item_id"]
    )

    prepared["culture_canonical"] = clean_text(
        prepared["culture_canonical"]
    )

    if prepared["item_id"].eq("").any():
        raise ValueError(
            "El ground truth contiene item_id vacíos."
        )

    if prepared["culture_canonical"].eq("").any():
        raise ValueError(
            "El ground truth contiene culturas canónicas vacías."
        )

    if prepared["item_id"].duplicated().any():
        duplicated = (
            prepared.loc[
                prepared["item_id"].duplicated(keep=False),
                "item_id",
            ]
            .tolist()
        )

        raise ValueError(
            "El ground truth contiene item_id duplicados: "
            f"{duplicated[:10]}"
        )

    relevant_counts = pd.to_numeric(
        prepared["relevant_item_count"],
        errors="coerce",
    )

    invalid_counts = (
        relevant_counts.isna()
        | relevant_counts.lt(0)
        | relevant_counts.mod(1).ne(0)
    )

    if invalid_counts.any():
        raise ValueError(
            "El ground truth contiene relevant_item_count "
            "inválidos."
        )

    prepared["relevant_item_count"] = (
        relevant_counts.astype(int)
    )

    prepared["query_eligible"] = (
        parse_boolean_series(
            prepared["query_eligible"]
        )
    )

    expected_relevant_counts = (
        prepared.groupby(
            "culture_canonical"
        )["item_id"]
        .transform("size")
        .sub(1)
        .astype(int)
    )

    inconsistent_counts = (
        prepared["relevant_item_count"]
        .ne(expected_relevant_counts)
    )

    if inconsistent_counts.any():
        inconsistent = prepared.loc[
            inconsistent_counts,
            [
                "item_id",
                "culture_canonical",
                "relevant_item_count",
            ],
        ].copy()

        inconsistent["expected_relevant_item_count"] = (
            expected_relevant_counts.loc[
                inconsistent_counts
            ]
        )

        raise ValueError(
            "Los conteos de relevancia no coinciden con "
            "el tamaño de los grupos culturales:\n"
            f"{inconsistent.to_string(index=False)}"
        )

    expected_eligibility = (
        prepared["relevant_item_count"].gt(0)
    )

    if not prepared["query_eligible"].equals(
        expected_eligibility
    ):
        raise ValueError(
            "query_eligible no coincide con "
            "relevant_item_count > 0."
        )

    return prepared.reset_index(drop=True)


def build_relevance_sets(
    ground_truth: pd.DataFrame,
) -> dict[str, set[str]]:
    """Construye el conjunto relevante de cada consulta elegible."""

    prepared = prepare_ground_truth(ground_truth)

    culture_groups = (
        prepared.groupby(
            "culture_canonical",
            sort=False,
        )["item_id"]
        .apply(list)
        .to_dict()
    )

    relevance_sets: dict[str, set[str]] = {}

    eligible_rows = prepared[
        prepared["query_eligible"]
    ]

    for row in eligible_rows.itertuples():
        group_items = set(
            culture_groups[row.culture_canonical]
        )

        group_items.discard(row.item_id)

        relevance_sets[row.item_id] = group_items

    return relevance_sets


def prepare_rankings(
    rankings: pd.DataFrame,
    ground_truth: pd.DataFrame,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[int, ...]]:
    """Valida un ranking completo sobre el universo estricto."""

    missing_columns = (
        REQUIRED_RANKING_COLUMNS
        .difference(rankings.columns)
    )

    if missing_columns:
        raise ValueError(
            "Faltan columnas en el ranking: "
            f"{sorted(missing_columns)}"
        )

    prepared_ground_truth = prepare_ground_truth(
        ground_truth
    )

    prepared_rankings = rankings.copy()

    prepared_rankings["query_item_id"] = clean_text(
        prepared_rankings["query_item_id"]
    )

    prepared_rankings["candidate_item_id"] = clean_text(
        prepared_rankings["candidate_item_id"]
    )

    for column in [
        "query_item_id",
        "candidate_item_id",
    ]:
        if prepared_rankings[column].eq("").any():
            raise ValueError(
                f"El ranking contiene valores vacíos en {column}."
            )

    rank_values = pd.to_numeric(
        prepared_rankings["rank"],
        errors="coerce",
    )

    invalid_ranks = (
        rank_values.isna()
        | rank_values.lt(1)
        | rank_values.mod(1).ne(0)
    )

    if invalid_ranks.any():
        raise ValueError(
            "El ranking contiene posiciones inválidas."
        )

    prepared_rankings["rank"] = (
        rank_values.astype(int)
    )

    duplicated_pairs = prepared_rankings.duplicated(
        subset=[
            "query_item_id",
            "candidate_item_id",
        ],
        keep=False,
    )

    if duplicated_pairs.any():
        duplicates = prepared_rankings.loc[
            duplicated_pairs,
            [
                "query_item_id",
                "candidate_item_id",
            ],
        ]

        raise ValueError(
            "El ranking contiene pares consulta-candidato "
            "duplicados:\n"
            f"{duplicates.to_string(index=False)}"
        )

    duplicated_ranks = prepared_rankings.duplicated(
        subset=[
            "query_item_id",
            "rank",
        ],
        keep=False,
    )

    if duplicated_ranks.any():
        raise ValueError(
            "Una consulta contiene posiciones de ranking "
            "duplicadas."
        )

    self_pairs = prepared_rankings[
        "query_item_id"
    ].eq(
        prepared_rankings["candidate_item_id"]
    )

    if self_pairs.any():
        raise ValueError(
            "El ranking contiene pares autorreferentes."
        )

    strict_universe = set(
        prepared_ground_truth["item_id"]
    )

    eligible_queries = set(
        prepared_ground_truth.loc[
            prepared_ground_truth["query_eligible"],
            "item_id",
        ]
    )

    observed_queries = set(
        prepared_rankings["query_item_id"]
    )

    if observed_queries != eligible_queries:
        missing_queries = sorted(
            eligible_queries - observed_queries
        )

        unexpected_queries = sorted(
            observed_queries - eligible_queries
        )

        raise ValueError(
            "El ranking no cubre exactamente las consultas "
            "elegibles. "
            f"Faltantes: {missing_queries[:10]}. "
            f"Inesperadas: {unexpected_queries[:10]}."
        )

    unknown_candidates = (
        set(prepared_rankings["candidate_item_id"])
        - strict_universe
    )

    if unknown_candidates:
        raise ValueError(
            "El ranking contiene candidatos fuera del universo "
            "cultural estricto: "
            f"{sorted(unknown_candidates)[:10]}"
        )

    candidate_count = len(strict_universe) - 1

    normalized_k_values = normalize_k_values(
        k_values,
        candidate_count,
    )

    expected_ranks = list(
        range(1, candidate_count + 1)
    )

    for query_item_id, group in (
        prepared_rankings.groupby(
            "query_item_id",
            sort=False,
        )
    ):
        query_candidates = set(
            group["candidate_item_id"]
        )

        expected_candidates = (
            strict_universe - {query_item_id}
        )

        if query_candidates != expected_candidates:
            missing_candidates = sorted(
                expected_candidates - query_candidates
            )

            unexpected_candidates = sorted(
                query_candidates - expected_candidates
            )

            raise ValueError(
                "La consulta no contiene el universo completo "
                f"de candidatos: {query_item_id}. "
                f"Faltantes: {missing_candidates[:10]}. "
                f"Inesperados: {unexpected_candidates[:10]}."
            )

        observed_ranks = sorted(
            group["rank"].tolist()
        )

        if observed_ranks != expected_ranks:
            raise ValueError(
                "Las posiciones del ranking no son consecutivas "
                f"para la consulta {query_item_id}."
            )

    prepared_rankings = (
        prepared_rankings.sort_values(
            [
                "query_item_id",
                "rank",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return (
        prepared_rankings,
        prepared_ground_truth,
        normalized_k_values,
    )


def precision_at_k(
    relevance: list[int],
    k: int,
) -> float:
    """Calcula Precision@K para relevancia binaria."""

    return float(
        sum(relevance[:k]) / k
    )


def recall_at_k(
    relevance: list[int],
    relevant_item_count: int,
    k: int,
) -> float:
    """Calcula Recall@K para relevancia binaria."""

    if relevant_item_count <= 0:
        return 0.0

    return float(
        sum(relevance[:k])
        / relevant_item_count
    )


def ndcg_at_k(
    relevance: list[int],
    relevant_item_count: int,
    k: int,
) -> float:
    """Calcula nDCG@K con ganancias binarias."""

    dcg = sum(
        relevance[index - 1]
        / log2(index + 1)
        for index in range(1, k + 1)
    )

    ideal_relevant_count = min(
        relevant_item_count,
        k,
    )

    idcg = sum(
        1.0 / log2(index + 1)
        for index in range(
            1,
            ideal_relevant_count + 1,
        )
    )

    if idcg == 0:
        return 0.0

    return float(dcg / idcg)


def metric_columns(
    k_values: Iterable[int],
) -> list[str]:
    """Devuelve las columnas métricas en orden estable."""

    columns = [
        "reciprocal_rank",
    ]

    for k in k_values:
        columns.extend(
            [
                f"precision_at_{k}",
                f"recall_at_{k}",
                f"ndcg_at_{k}",
            ]
        )

    return columns


def evaluate_rankings(
    rankings: pd.DataFrame,
    ground_truth: pd.DataFrame,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
) -> pd.DataFrame:
    """Calcula métricas por consulta."""

    (
        prepared_rankings,
        prepared_ground_truth,
        normalized_k_values,
    ) = prepare_rankings(
        rankings,
        ground_truth,
        k_values,
    )

    relevance_sets = build_relevance_sets(
        prepared_ground_truth
    )

    query_metadata = (
        prepared_ground_truth.set_index("item_id")[
            [
                "culture_canonical",
                "relevant_item_count",
            ]
        ]
        .to_dict(orient="index")
    )

    ranking_groups = {
        query_item_id: group.sort_values(
            "rank",
            kind="stable",
        )
        for query_item_id, group in (
            prepared_rankings.groupby(
                "query_item_id",
                sort=False,
            )
        )
    }

    eligible_query_order = (
        prepared_ground_truth.loc[
            prepared_ground_truth["query_eligible"],
            "item_id",
        ]
        .tolist()
    )

    rows: list[dict[str, object]] = []

    for query_item_id in eligible_query_order:
        group = ranking_groups[query_item_id]

        ranked_candidates = (
            group["candidate_item_id"].tolist()
        )

        relevant_items = relevance_sets[
            query_item_id
        ]

        relevance = [
            int(candidate in relevant_items)
            for candidate in ranked_candidates
        ]

        relevant_item_count = int(
            query_metadata[
                query_item_id
            ]["relevant_item_count"]
        )

        relevant_positions = [
            position
            for position, value in enumerate(
                relevance,
                start=1,
            )
            if value == 1
        ]

        if not relevant_positions:
            raise ValueError(
                "Una consulta elegible no recuperó ningún "
                "elemento relevante dentro del ranking completo: "
                f"{query_item_id}"
            )

        first_relevant_rank = relevant_positions[0]

        row: dict[str, object] = {
            "query_item_id": query_item_id,
            "culture_canonical": (
                query_metadata[
                    query_item_id
                ]["culture_canonical"]
            ),
            "relevant_item_count": (
                relevant_item_count
            ),
            "first_relevant_rank": (
                first_relevant_rank
            ),
            "reciprocal_rank": (
                1.0 / first_relevant_rank
            ),
            "evaluation_scope": (
                EVALUATION_SCOPE
            ),
        }

        for k in normalized_k_values:
            row[f"precision_at_{k}"] = (
                precision_at_k(
                    relevance,
                    k,
                )
            )

            row[f"recall_at_{k}"] = (
                recall_at_k(
                    relevance,
                    relevant_item_count,
                    k,
                )
            )

            row[f"ndcg_at_{k}"] = (
                ndcg_at_k(
                    relevance,
                    relevant_item_count,
                    k,
                )
            )

        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_metrics(
    per_query_metrics: pd.DataFrame,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
) -> pd.DataFrame:
    """Calcula métricas macro promedio sobre las consultas."""

    normalized_k_values = tuple(
        sorted(set(int(k) for k in k_values))
    )

    columns = metric_columns(
        normalized_k_values
    )

    missing_columns = set(columns).difference(
        per_query_metrics.columns
    )

    if missing_columns:
        raise ValueError(
            "Faltan columnas métricas para agregar: "
            f"{sorted(missing_columns)}"
        )

    row: dict[str, object] = {
        "evaluation_scope": EVALUATION_SCOPE,
        "query_count": len(per_query_metrics),
        "mrr": float(
            per_query_metrics[
                "reciprocal_rank"
            ].mean()
        ),
    }

    for k in normalized_k_values:
        for metric_name in [
            "precision",
            "recall",
            "ndcg",
        ]:
            column = f"{metric_name}_at_{k}"

            row[column] = float(
                per_query_metrics[column].mean()
            )

    return pd.DataFrame([row])


def aggregate_metrics_by_culture(
    per_query_metrics: pd.DataFrame,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
) -> pd.DataFrame:
    """Calcula métricas promedio por cultura canónica."""

    normalized_k_values = tuple(
        sorted(set(int(k) for k in k_values))
    )

    columns = metric_columns(
        normalized_k_values
    )

    required_columns = {
        "culture_canonical",
        *columns,
    }

    missing_columns = required_columns.difference(
        per_query_metrics.columns
    )

    if missing_columns:
        raise ValueError(
            "Faltan columnas para agregar por cultura: "
            f"{sorted(missing_columns)}"
        )

    grouped_means = (
        per_query_metrics.groupby(
            "culture_canonical",
            sort=True,
        )[columns]
        .mean()
        .reset_index()
    )

    query_counts = (
        per_query_metrics.groupby(
            "culture_canonical",
            sort=True,
        )
        .size()
        .rename("query_count")
        .reset_index()
    )

    summary = query_counts.merge(
        grouped_means,
        on="culture_canonical",
        how="inner",
        validate="one_to_one",
    )

    summary = summary.rename(
        columns={
            "reciprocal_rank": "mrr",
        }
    )

    summary.insert(
        1,
        "evaluation_scope",
        EVALUATION_SCOPE,
    )

    return summary