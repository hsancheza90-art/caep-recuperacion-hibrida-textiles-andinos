"""Construye una distribución multisemilla del baseline aleatorio.

El objetivo es sustituir el resultado de una única semilla por una
estimación descriptiva reproducible basada en múltiples permutaciones.

Se reportan:

- resultados generales por ejecución;
- resultados por cultura y ejecución;
- media y desviación estándar;
- percentiles empíricos 2.5 y 97.5;
- expectativas aleatorias teóricas para Precision@K y Recall@K.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from src.evaluation.build_reference_rankings import (
    build_random_ranking,
)
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "reference_baselines"
)

RUNS_PATH = (
    OUTPUT_DIR
    / "random_multiseed_runs_v1.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "random_multiseed_summary_v1.csv"
)

BY_CULTURE_RUNS_PATH = (
    OUTPUT_DIR
    / "random_multiseed_by_culture_runs_v1.csv"
)

BY_CULTURE_SUMMARY_PATH = (
    OUTPUT_DIR
    / "random_multiseed_by_culture_summary_v1.csv"
)

THEORETICAL_PATH = (
    OUTPUT_DIR
    / "random_theoretical_expectations_v1.csv"
)

BASELINE_VERSION = "random_multiseed_baseline_v1"
DEFAULT_BASE_SEED = 20260715
DEFAULT_RUN_COUNT = 100


def metric_columns(
    k_values: Iterable[int],
) -> list[str]:
    """Devuelve las métricas generales en un orden estable."""

    columns = ["mrr"]

    for k in sorted(set(int(value) for value in k_values)):
        columns.extend(
            [
                f"precision_at_{k}",
                f"recall_at_{k}",
                f"ndcg_at_{k}",
            ]
        )

    return columns


def generate_seeds(
    base_seed: int,
    run_count: int,
) -> list[int]:
    """Genera una secuencia determinista de semillas."""

    if run_count < 2:
        raise ValueError(
            "El baseline multisemilla requiere al menos "
            "dos ejecuciones."
        )

    return [
        int(base_seed) + offset
        for offset in range(run_count)
    ]


def summarize_distribution(
    runs: pd.DataFrame,
    metrics: Iterable[str],
) -> pd.DataFrame:
    """Resume la distribución empírica de cada métrica."""

    rows: list[dict[str, object]] = []

    for metric in metrics:
        if metric not in runs.columns:
            raise ValueError(
                f"No existe la métrica requerida: {metric}"
            )

        values = pd.to_numeric(
            runs[metric],
            errors="raise",
        )

        rows.append(
            {
                "baseline_version": BASELINE_VERSION,
                "metric": metric,
                "run_count": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "min": float(values.min()),
                "q025": float(values.quantile(0.025)),
                "median": float(values.median()),
                "q975": float(values.quantile(0.975)),
                "max": float(values.max()),
            }
        )

    return pd.DataFrame(rows)


def summarize_distribution_by_culture(
    runs: pd.DataFrame,
    metrics: Iterable[str],
) -> pd.DataFrame:
    """Resume la distribución de métricas para cada cultura."""

    required = {
        "culture_canonical",
        *metrics,
    }

    missing = required.difference(runs.columns)

    if missing:
        raise ValueError(
            "Faltan columnas para el resumen por cultura: "
            f"{sorted(missing)}"
        )

    rows: list[dict[str, object]] = []

    for culture, group in runs.groupby(
        "culture_canonical",
        sort=True,
    ):
        culture_summary = summarize_distribution(
            group,
            metrics,
        )

        culture_summary.insert(
            1,
            "culture_canonical",
            culture,
        )

        rows.extend(
            culture_summary.to_dict(
                orient="records"
            )
        )

    return pd.DataFrame(rows)


def build_theoretical_expectations(
    ground_truth: pd.DataFrame,
    k_values: Iterable[int],
) -> pd.DataFrame:
    """Calcula expectativas teóricas del ranking aleatorio.

    Para una consulta q:

        E[Precision@K] = R_q / N

        E[Recall@K] = K / N

    donde R_q es el número de relevantes y N es el número total
    de candidatos por consulta.

    La Precision@K macro se obtiene promediando R_q / N sobre
    todas las consultas elegibles.
    """

    prepared = prepare_ground_truth(
        ground_truth
    )

    eligible = prepared[
        prepared["query_eligible"]
    ].copy()

    candidate_count = len(prepared) - 1

    expected_precision = float(
        (
            eligible["relevant_item_count"]
            / candidate_count
        ).mean()
    )

    rows: list[dict[str, object]] = []

    for k in sorted(set(int(value) for value in k_values)):
        if k > candidate_count:
            raise ValueError(
                f"K={k} supera los {candidate_count} candidatos."
            )

        rows.append(
            {
                "baseline_version": BASELINE_VERSION,
                "metric": f"precision_at_{k}",
                "expected_value": expected_precision,
                "candidate_count": candidate_count,
                "query_count": len(eligible),
                "formula": (
                    "macro_mean(relevant_item_count "
                    "/ candidate_count)"
                ),
            }
        )

        rows.append(
            {
                "baseline_version": BASELINE_VERSION,
                "metric": f"recall_at_{k}",
                "expected_value": float(
                    k / candidate_count
                ),
                "candidate_count": candidate_count,
                "query_count": len(eligible),
                "formula": "k / candidate_count",
            }
        )

    return pd.DataFrame(rows)


def build_multiseed_baseline(
    ground_truth: pd.DataFrame,
    seeds: Iterable[int],
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> dict[str, pd.DataFrame]:
    """Ejecuta y resume el baseline para múltiples semillas."""

    seed_list = [int(seed) for seed in seeds]

    if len(seed_list) < 2:
        raise ValueError(
            "Se requieren al menos dos semillas."
        )

    if len(seed_list) != len(set(seed_list)):
        raise ValueError(
            "Las semillas del baseline deben ser únicas."
        )

    overall_runs: list[pd.DataFrame] = []
    culture_runs: list[pd.DataFrame] = []

    for run_index, seed in enumerate(
        seed_list,
        start=1,
    ):
        ranking = build_random_ranking(
            ground_truth,
            seed=seed,
        )

        per_query = evaluate_rankings(
            ranking,
            ground_truth,
            k_values=k_values,
        )

        overall = aggregate_metrics(
            per_query,
            k_values=k_values,
        )

        by_culture = aggregate_metrics_by_culture(
            per_query,
            k_values=k_values,
        )

        for dataframe in [
            overall,
            by_culture,
        ]:
            dataframe.insert(
                0,
                "baseline_version",
                BASELINE_VERSION,
            )

            dataframe.insert(
                1,
                "run_index",
                run_index,
            )

            dataframe.insert(
                2,
                "generator_seed",
                seed,
            )

        overall_runs.append(overall)
        culture_runs.append(by_culture)

    overall_runs_df = pd.concat(
        overall_runs,
        ignore_index=True,
    )

    culture_runs_df = pd.concat(
        culture_runs,
        ignore_index=True,
    )

    metrics = metric_columns(k_values)

    summary = summarize_distribution(
        overall_runs_df,
        metrics,
    )

    by_culture_summary = (
        summarize_distribution_by_culture(
            culture_runs_df,
            metrics,
        )
    )

    theoretical = build_theoretical_expectations(
        ground_truth,
        k_values,
    )

    return {
        "runs": overall_runs_df,
        "summary": summary,
        "by_culture_runs": culture_runs_df,
        "by_culture_summary": by_culture_summary,
        "theoretical": theoretical,
    }


def write_csv(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    """Guarda un CSV con codificación estable."""

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


def write_artifacts(
    artifacts: dict[str, pd.DataFrame],
) -> None:
    """Guarda los artefactos del baseline multisemilla."""

    paths = {
        "runs": RUNS_PATH,
        "summary": SUMMARY_PATH,
        "by_culture_runs": BY_CULTURE_RUNS_PATH,
        "by_culture_summary": BY_CULTURE_SUMMARY_PATH,
        "theoretical": THEORETICAL_PATH,
    }

    for artifact_name, path in paths.items():
        write_csv(
            artifacts[artifact_name],
            path,
        )


def print_summary(
    artifacts: dict[str, pd.DataFrame],
    seeds: list[int],
) -> None:
    """Imprime un resumen compacto."""

    summary = artifacts["summary"]

    compact = (
        summary.set_index("metric")[
            [
                "mean",
                "std",
                "q025",
                "median",
                "q975",
            ]
        ]
        .round(6)
    )

    print("\nBASELINE ALEATORIO MULTISEMILLA")
    print("=" * 90)
    print(f"Ejecuciones: {len(seeds)}")
    print(f"Primera semilla: {seeds[0]}")
    print(f"Última semilla: {seeds[-1]}")

    print("\nDISTRIBUCIÓN GENERAL")
    print("=" * 90)
    print(compact.to_string())

    print("\nEXPECTATIVAS TEÓRICAS")
    print("=" * 90)

    theoretical = artifacts["theoretical"][
        [
            "metric",
            "expected_value",
        ]
    ].copy()

    theoretical["expected_value"] = (
        theoretical["expected_value"].round(6)
    )

    print(theoretical.to_string(index=False))

    print("\nARTEFACTOS")
    print("=" * 90)
    print(RUNS_PATH.relative_to(PROJECT_ROOT))
    print(SUMMARY_PATH.relative_to(PROJECT_ROOT))
    print(
        BY_CULTURE_SUMMARY_PATH.relative_to(
            PROJECT_ROOT
        )
    )
    print(THEORETICAL_PATH.relative_to(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Procesa los argumentos de la línea de comandos."""

    parser = argparse.ArgumentParser(
        description=(
            "Construye el baseline aleatorio "
            "multisemilla."
        )
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUN_COUNT,
        help=(
            "Número de semillas. "
            f"Valor predeterminado: {DEFAULT_RUN_COUNT}."
        ),
    )

    parser.add_argument(
        "--base-seed",
        type=int,
        default=DEFAULT_BASE_SEED,
        help=(
            "Primera semilla de la secuencia. "
            f"Valor predeterminado: {DEFAULT_BASE_SEED}."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Ejecuta el baseline multisemilla."""

    args = parse_args()

    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el ground truth: "
            f"{GROUND_TRUTH_PATH}"
        )

    ground_truth = pd.read_csv(
        GROUND_TRUTH_PATH
    )

    seeds = generate_seeds(
        base_seed=args.base_seed,
        run_count=args.runs,
    )

    artifacts = build_multiseed_baseline(
        ground_truth,
        seeds=seeds,
        k_values=DEFAULT_K_VALUES,
    )

    write_artifacts(artifacts)

    print_summary(
        artifacts,
        seeds,
    )


if __name__ == "__main__":
    main()