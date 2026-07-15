"""Construye rankings de referencia para la evaluación de recuperación.

Referencias implementadas:

1. oracle_culture:
   coloca primero todos los candidatos culturalmente relevantes;

2. random:
   produce una permutación pseudoaleatoria determinista para cada consulta.

Ambos rankings operan exclusivamente sobre el universo cultural estricto.
"""

from __future__ import annotations

import hashlib
import random
import pandas as pd
from pathlib import Path

from src.evaluation.retrieval_metrics import (
    DEFAULT_K_VALUES,
    aggregate_metrics,
    aggregate_metrics_by_culture,
    evaluate_rankings,
    prepare_ground_truth,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "culture_ground_truth_strict_v1.csv"
)

RANKING_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rankings"
)

EVALUATION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "reference_baselines"
)

ORACLE_RANKING_PATH = (
    RANKING_DIR
    / "oracle_culture_ranking_strict_v1.csv"
)

RANDOM_RANKING_PATH = (
    RANKING_DIR
    / "random_ranking_strict_seed_20260715.csv"
)

ORACLE_PER_QUERY_PATH = (
    EVALUATION_DIR
    / "oracle_culture_per_query_metrics_v1.csv"
)

ORACLE_OVERALL_PATH = (
    EVALUATION_DIR
    / "oracle_culture_overall_metrics_v1.csv"
)

ORACLE_BY_CULTURE_PATH = (
    EVALUATION_DIR
    / "oracle_culture_by_culture_metrics_v1.csv"
)

RANDOM_PER_QUERY_PATH = (
    EVALUATION_DIR
    / "random_per_query_metrics_seed_20260715.csv"
)

RANDOM_OVERALL_PATH = (
    EVALUATION_DIR
    / "random_overall_metrics_seed_20260715.csv"
)

RANDOM_BY_CULTURE_PATH = (
    EVALUATION_DIR
    / "random_by_culture_metrics_seed_20260715.csv"
)

COMBINED_SUMMARY_PATH = (
    EVALUATION_DIR
    / "reference_baselines_summary_v1.csv"
)

REFERENCE_VERSION = "reference_rankings_v1"
ORACLE_METHOD = "oracle_culture"
RANDOM_METHOD = "random"
DEFAULT_RANDOM_SEED = 20260715


RANKING_COLUMNS = [
    "query_item_id",
    "candidate_item_id",
    "rank",
    "ranking_method",
    "generator_seed",
    "reference_version",
]


def stable_query_seed(
    global_seed: int,
    query_item_id: str,
) -> int:
    """Deriva una semilla estable y específica para una consulta.

    No utiliza hash() de Python porque ese valor puede cambiar entre
    procesos. SHA-256 garantiza estabilidad entre ejecuciones y sistemas.
    """

    payload = (
        f"{int(global_seed)}::{query_item_id}"
        .encode("utf-8")
    )

    digest = hashlib.sha256(payload).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )


def prepare_reference_context(
    ground_truth: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    list[str],
    list[str],
    dict[str, str],
]:
    """Prepara universo, consultas y culturas para los rankings."""

    prepared = prepare_ground_truth(
        ground_truth
    )

    universe_order = (
        prepared["item_id"].tolist()
    )

    query_order = (
        prepared.loc[
            prepared["query_eligible"],
            "item_id",
        ]
        .tolist()
    )

    culture_by_item = (
        prepared.set_index("item_id")[
            "culture_canonical"
        ]
        .to_dict()
    )

    return (
        prepared,
        universe_order,
        query_order,
        culture_by_item,
    )


def build_oracle_ranking(
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    """Construye el ranking oracle basado en cultura estricta.

    Dentro de cada bloque —relevante y no relevante— se conserva
    el orden del universo del ground truth como criterio de desempate.
    """

    (
        prepared,
        universe_order,
        query_order,
        culture_by_item,
    ) = prepare_reference_context(
        ground_truth
    )

    relevant_count_by_item = (
        prepared.set_index("item_id")[
            "relevant_item_count"
        ]
        .astype(int)
        .to_dict()
    )

    rows: list[dict[str, object]] = []

    for query_item_id in query_order:
        query_culture = culture_by_item[
            query_item_id
        ]

        relevant_candidates = [
            candidate_item_id
            for candidate_item_id in universe_order
            if (
                candidate_item_id
                != query_item_id
                and culture_by_item[
                    candidate_item_id
                ]
                == query_culture
            )
        ]

        non_relevant_candidates = [
            candidate_item_id
            for candidate_item_id in universe_order
            if (
                candidate_item_id
                != query_item_id
                and culture_by_item[
                    candidate_item_id
                ]
                != query_culture
            )
        ]

        expected_relevant_count = (
            relevant_count_by_item[
                query_item_id
            ]
        )

        if (
            len(relevant_candidates)
            != expected_relevant_count
        ):
            raise ValueError(
                "El oracle no encontró el número esperado "
                "de candidatos relevantes para "
                f"{query_item_id}: "
                f"{len(relevant_candidates)} != "
                f"{expected_relevant_count}"
            )

        candidates = [
            *relevant_candidates,
            *non_relevant_candidates,
        ]

        for rank, candidate_item_id in enumerate(
            candidates,
            start=1,
        ):
            rows.append(
                {
                    "query_item_id": query_item_id,
                    "candidate_item_id": (
                        candidate_item_id
                    ),
                    "rank": rank,
                    "ranking_method": (
                        ORACLE_METHOD
                    ),
                    "generator_seed": -1,
                    "reference_version": (
                        REFERENCE_VERSION
                    ),
                }
            )

    return pd.DataFrame(
        rows,
        columns=RANKING_COLUMNS,
    )


def build_random_ranking(
    ground_truth: pd.DataFrame,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Construye un ranking aleatorio reproducible.

    Cada consulta recibe una semilla derivada de:

        semilla global + item_id de consulta

    Esto evita que cambios en el orden de las consultas alteren
    las permutaciones de consultas ya existentes.
    """

    (
        _,
        universe_order,
        query_order,
        _,
    ) = prepare_reference_context(
        ground_truth
    )

    rows: list[dict[str, object]] = []

    for query_item_id in query_order:
        candidates = [
            candidate_item_id
            for candidate_item_id in universe_order
            if candidate_item_id != query_item_id
        ]

        query_seed = stable_query_seed(
            seed,
            query_item_id,
        )

        query_random = random.Random(
            query_seed
        )

        query_random.shuffle(candidates)

        for rank, candidate_item_id in enumerate(
            candidates,
            start=1,
        ):
            rows.append(
                {
                    "query_item_id": query_item_id,
                    "candidate_item_id": (
                        candidate_item_id
                    ),
                    "rank": rank,
                    "ranking_method": (
                        RANDOM_METHOD
                    ),
                    "generator_seed": int(seed),
                    "reference_version": (
                        REFERENCE_VERSION
                    ),
                }
            )

    return pd.DataFrame(
        rows,
        columns=RANKING_COLUMNS,
    )


def annotate_metrics(
    dataframe: pd.DataFrame,
    ranking_method: str,
    generator_seed: int,
) -> pd.DataFrame:
    """Añade metadata del ranking a un resultado de evaluación."""

    annotated = dataframe.copy()

    annotated.insert(
        0,
        "ranking_method",
        ranking_method,
    )

    annotated.insert(
        1,
        "generator_seed",
        int(generator_seed),
    )

    annotated.insert(
        2,
        "reference_version",
        REFERENCE_VERSION,
    )

    return annotated


def evaluate_reference_ranking(
    ranking: pd.DataFrame,
    ground_truth: pd.DataFrame,
    ranking_method: str,
    generator_seed: int,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Evalúa un ranking y produce tres niveles de resultados."""

    per_query = evaluate_rankings(
        ranking,
        ground_truth,
        k_values=k_values,
    )

    overall = aggregate_metrics(
        per_query,
        k_values=k_values,
    )

    by_culture = (
        aggregate_metrics_by_culture(
            per_query,
            k_values=k_values,
        )
    )

    return (
        annotate_metrics(
            per_query,
            ranking_method,
            generator_seed,
        ),
        annotate_metrics(
            overall,
            ranking_method,
            generator_seed,
        ),
        annotate_metrics(
            by_culture,
            ranking_method,
            generator_seed,
        ),
    )


def build_reference_artifacts(
    ground_truth: pd.DataFrame,
    random_seed: int = DEFAULT_RANDOM_SEED,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> dict[str, pd.DataFrame]:
    """Construye rankings y resultados de ambas referencias."""

    oracle_ranking = build_oracle_ranking(
        ground_truth
    )

    random_ranking = build_random_ranking(
        ground_truth,
        seed=random_seed,
    )

    (
        oracle_per_query,
        oracle_overall,
        oracle_by_culture,
    ) = evaluate_reference_ranking(
        oracle_ranking,
        ground_truth,
        ranking_method=ORACLE_METHOD,
        generator_seed=-1,
        k_values=k_values,
    )

    (
        random_per_query,
        random_overall,
        random_by_culture,
    ) = evaluate_reference_ranking(
        random_ranking,
        ground_truth,
        ranking_method=RANDOM_METHOD,
        generator_seed=random_seed,
        k_values=k_values,
    )

    combined_summary = pd.concat(
        [
            oracle_overall,
            random_overall,
        ],
        ignore_index=True,
    )

    return {
        "oracle_ranking": oracle_ranking,
        "random_ranking": random_ranking,
        "oracle_per_query": oracle_per_query,
        "oracle_overall": oracle_overall,
        "oracle_by_culture": (
            oracle_by_culture
        ),
        "random_per_query": random_per_query,
        "random_overall": random_overall,
        "random_by_culture": (
            random_by_culture
        ),
        "combined_summary": (
            combined_summary
        ),
    }


def write_csv(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    """Escribe un CSV reproducible."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )


def write_reference_artifacts(
    artifacts: dict[str, pd.DataFrame],
) -> None:
    """Guarda todos los artefactos generados."""

    output_paths = {
        "oracle_ranking": (
            ORACLE_RANKING_PATH
        ),
        "random_ranking": (
            RANDOM_RANKING_PATH
        ),
        "oracle_per_query": (
            ORACLE_PER_QUERY_PATH
        ),
        "oracle_overall": (
            ORACLE_OVERALL_PATH
        ),
        "oracle_by_culture": (
            ORACLE_BY_CULTURE_PATH
        ),
        "random_per_query": (
            RANDOM_PER_QUERY_PATH
        ),
        "random_overall": (
            RANDOM_OVERALL_PATH
        ),
        "random_by_culture": (
            RANDOM_BY_CULTURE_PATH
        ),
        "combined_summary": (
            COMBINED_SUMMARY_PATH
        ),
    }

    for artifact_name, output_path in (
        output_paths.items()
    ):
        write_csv(
            artifacts[artifact_name],
            output_path,
        )


def print_summary(
    artifacts: dict[str, pd.DataFrame],
) -> None:
    """Imprime resultados generales."""

    oracle_ranking = artifacts[
        "oracle_ranking"
    ]

    random_ranking = artifacts[
        "random_ranking"
    ]

    combined_summary = artifacts[
        "combined_summary"
    ]

    print("\nRANKINGS DE REFERENCIA")
    print("=" * 110)
    print(
        f"Consultas oracle: "
        f"{oracle_ranking['query_item_id'].nunique()}"
    )
    print(
        f"Consultas aleatorias: "
        f"{random_ranking['query_item_id'].nunique()}"
    )
    print(
        f"Candidatos por consulta: "
        f"{int(oracle_ranking.groupby('query_item_id').size().iloc[0])}"
    )
    print(
        f"Filas por ranking: "
        f"{len(oracle_ranking)}"
    )
    print(
        f"Semilla aleatoria global: "
        f"{DEFAULT_RANDOM_SEED}"
    )

    print("\nMÉTRICAS GENERALES")
    print("=" * 110)
    metric_columns = [
        "mrr",
        "precision_at_1",
        "recall_at_1",
        "ndcg_at_1",
        "precision_at_5",
        "recall_at_5",
        "ndcg_at_5",
        "precision_at_10",
        "recall_at_10",
        "ndcg_at_10",
    ]

    compact_summary = (
        combined_summary[
            [
                "ranking_method",
                *metric_columns,
            ]
        ]
        .set_index("ranking_method")
        .transpose()
        .round(6)
    )

    print(compact_summary.to_string())

    print("\nARTEFACTOS")
    print("=" * 110)
    print(
        ORACLE_RANKING_PATH.relative_to(
            PROJECT_ROOT
        )
    )
    print(
        RANDOM_RANKING_PATH.relative_to(
            PROJECT_ROOT
        )
    )
    print(
        COMBINED_SUMMARY_PATH.relative_to(
            PROJECT_ROOT
        )
    )


def main() -> None:
    """Ejecuta la construcción y evaluación de referencias."""

    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el ground truth cultural: "
            f"{GROUND_TRUTH_PATH}"
        )

    ground_truth = pd.read_csv(
        GROUND_TRUTH_PATH
    )

    artifacts = build_reference_artifacts(
        ground_truth,
        random_seed=DEFAULT_RANDOM_SEED,
        k_values=DEFAULT_K_VALUES,
    )

    write_reference_artifacts(
        artifacts
    )

    print_summary(
        artifacts
    )


if __name__ == "__main__":
    main()