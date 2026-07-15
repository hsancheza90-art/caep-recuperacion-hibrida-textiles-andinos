from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pandas as pd


UPSTREAM_REPO = Path(".cache/upstream-textiles")

SOURCES = {
    "met_curacion_principal": {
        "ref": "origin/curacion/corpus-met-textiles-andinos-v1",
        "path": "data/processed/corpus_met_textiles_andinos_v1_principal.csv",
    },
    "cma_curacion_principal": {
        "ref": "origin/curacion/corpus-cma-textiles-andinos-v1",
        "path": "data/metadata/cma_corpus_principal.csv",
    },
}


def git_show(ref: str, path: str) -> bytes:
    """Read a file directly from a Git reference without checking out the branch."""
    result = subprocess.run(
        [
            "git",
            "-C",
            str(UPSTREAM_REPO),
            "show",
            f"{ref}:{path}",
        ],
        check=True,
        capture_output=True,
    )
    return result.stdout


def read_csv_from_git(ref: str, path: str) -> pd.DataFrame:
    """Try common encodings and separators."""
    raw = git_show(ref, path)

    attempts = [
        {"encoding": "utf-8-sig", "sep": ","},
        {"encoding": "utf-8", "sep": ","},
        {"encoding": "latin-1", "sep": ","},
        {"encoding": "utf-8-sig", "sep": ";"},
        {"encoding": "latin-1", "sep": ";"},
    ]

    errors: list[str] = []

    for options in attempts:
        try:
            text = raw.decode(options["encoding"])
            frame = pd.read_csv(
                io.StringIO(text),
                sep=options["sep"],
                dtype=str,
                keep_default_na=False,
            )

            if len(frame.columns) > 1:
                return frame

        except Exception as exc:
            errors.append(f"{options}: {exc}")

    raise RuntimeError(
        f"No se pudo leer {path}.\n" + "\n".join(errors)
    )


def find_candidate_columns(columns: list[str]) -> dict[str, list[str]]:
    patterns = {
        "id": ["id", "object", "accession", "registro"],
        "status": ["estado", "status", "curacion", "revis"],
        "image": ["imagen", "image", "primaryimagesmall", "iiif"],
        "object_url": ["objecturl", "object_url", "url_objeto", "web_url"],
        "title": ["titulo", "title"],
        "description": ["descripcion", "description"],
        "culture": ["cultura", "culture"],
        "period": ["periodo", "period", "date"],
        "material": ["material", "medium"],
        "technique": ["tecnica", "technique"],
    }

    normalized = {column: column.lower().strip() for column in columns}
    matches: dict[str, list[str]] = {}

    for category, tokens in patterns.items():
        matches[category] = [
            original
            for original, lower in normalized.items()
            if any(token in lower for token in tokens)
        ]

    return matches


def print_summary(name: str, frame: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print(name)
    print("=" * 90)
    print(f"Filas: {len(frame)}")
    print(f"Columnas: {len(frame.columns)}")

    print("\nNombres de columnas:")
    for index, column in enumerate(frame.columns, start=1):
        print(f"{index:02d}. {column}")

    print("\nColumnas candidatas:")
    candidates = find_candidate_columns(frame.columns.tolist())
    for category, columns in candidates.items():
        print(f"- {category}: {columns}")

    print("\nPrimer registro, campos no vacíos:")
    if not frame.empty:
        first = frame.iloc[0]
        for column, value in first.items():
            value = str(value).strip()
            if value:
                shortened = value[:160]
                print(f"- {column}: {shortened}")

    status_columns = candidates["status"]
    for column in status_columns:
        print(f"\nValores frecuentes de '{column}':")
        print(frame[column].value_counts(dropna=False).head(15).to_string())

    image_columns = candidates["image"]
    for column in image_columns:
        non_empty = frame[column].astype(str).str.strip().ne("").sum()
        print(
            f"\nCobertura de imagen '{column}': "
            f"{non_empty}/{len(frame)} "
            f"({non_empty / len(frame):.1%})"
            if len(frame)
            else f"\nCobertura de imagen '{column}': dataset vacío"
        )


def main() -> None:
    if not UPSTREAM_REPO.exists():
        raise FileNotFoundError(
            f"No existe el repositorio temporal: {UPSTREAM_REPO}"
        )

    for name, source in SOURCES.items():
        try:
            frame = read_csv_from_git(
                ref=source["ref"],
                path=source["path"],
            )
            print_summary(name, frame)

        except Exception as exc:
            print("\n" + "!" * 90)
            print(f"ERROR EN {name}")
            print(f"{source['path']}")
            print(exc)


if __name__ == "__main__":
    main()