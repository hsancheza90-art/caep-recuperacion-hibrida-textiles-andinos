from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pandas as pd


UPSTREAM_REPO = Path(".cache/upstream-textiles")

SOURCES = {
    "MET_principal_revisado": {
        "ref": "origin/mejora/met-flujo-reproducible-es",
        "path": "data/metadata/met_corpus_principal_v2_revisado.csv",
        "expected_rows": 127,
    },
    "MET_secundario_revisado": {
        "ref": "origin/mejora/met-flujo-reproducible-es",
        "path": "data/metadata/met_corpus_secundario_v2_revisado.csv",
        "expected_rows": 54,
    },
    "MET_descartados_revisado": {
        "ref": "origin/mejora/met-flujo-reproducible-es",
        "path": "data/metadata/met_descartados_v2_revisado.csv",
        "expected_rows": 30,
    },
    "CMA_principal_revisado": {
        "ref": "origin/mejora/cma-flujo-reproducible-es",
        "path": "data/metadata/cma_corpus_principal_revisado.csv",
        "expected_rows": 88,
    },
    "CMA_secundario_revisado": {
        "ref": "origin/mejora/cma-flujo-reproducible-es",
        "path": "data/metadata/cma_corpus_secundario_revisado.csv",
        "expected_rows": 19,
    },
    "CMA_descartados_revisado": {
        "ref": "origin/mejora/cma-flujo-reproducible-es",
        "path": "data/metadata/cma_descartados_revisado.csv",
        "expected_rows": 39,
    },
}


def git_show(ref: str, path: str) -> bytes:
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
    raw = git_show(ref, path)

    attempts = [
        ("utf-8-sig", ","),
        ("utf-8", ","),
        ("latin-1", ","),
        ("utf-8-sig", ";"),
        ("latin-1", ";"),
    ]

    errors: list[str] = []

    for encoding, separator in attempts:
        try:
            text = raw.decode(encoding)
            frame = pd.read_csv(
                io.StringIO(text),
                sep=separator,
                dtype=str,
                keep_default_na=False,
            )

            if len(frame.columns) > 1:
                return frame

        except Exception as exc:
            errors.append(
                f"encoding={encoding}, sep={separator}: {exc}"
            )

    raise RuntimeError(
        f"No se pudo leer {ref}:{path}\n" + "\n".join(errors)
    )


def find_columns(
    frame: pd.DataFrame,
    tokens: tuple[str, ...],
) -> list[str]:
    return [
        column
        for column in frame.columns
        if any(token in column.lower() for token in tokens)
    ]


def print_coverage(frame: pd.DataFrame, column: str) -> None:
    if column not in frame.columns:
        print(f"- {column}: no disponible")
        return

    values = frame[column].astype(str).str.strip()
    valid = values.ne("").sum()
    percentage = valid / len(frame) if len(frame) else 0

    print(
        f"- {column}: {valid}/{len(frame)} "
        f"({percentage:.1%})"
    )


def inspect_source(
    name: str,
    source: dict[str, str | int],
) -> bool:
    frame = read_csv_from_git(
        ref=str(source["ref"]),
        path=str(source["path"]),
    )

    expected_rows = int(source["expected_rows"])
    row_count_ok = len(frame) == expected_rows

    print("\n" + "=" * 88)
    print(name)
    print("=" * 88)
    print(f"Archivo: {source['path']}")
    print(f"Filas observadas: {len(frame)}")
    print(f"Filas esperadas: {expected_rows}")
    print(f"Conteo correcto: {row_count_ok}")
    print(f"Columnas: {len(frame.columns)}")

    print("\nColumnas disponibles:")
    for index, column in enumerate(frame.columns, start=1):
        print(f"{index:02d}. {column}")

    print("\nCobertura de campos de enlace:")
    for column in (
        "url_imagen",
        "imagen_local",
        "url_objeto",
        "object_url",
        "image_url",
    ):
        print_coverage(frame, column)

    status_columns = find_columns(
        frame,
        (
            "estado",
            "status",
            "decision",
            "revision",
            "curacion",
            "destino",
        ),
    )

    print("\nColumnas candidatas de estado:")
    print(status_columns)

    for column in status_columns:
        print(f"\nValores de '{column}':")
        counts = frame[column].value_counts(dropna=False).head(20)
        print(counts.to_string())

    id_columns = find_columns(
        frame,
        (
            "id_objeto",
            "id_registro",
            "objectid",
            "accession",
        ),
    )

    print("\nColumnas candidatas de identificador:")
    print(id_columns)

    for column in id_columns:
        values = frame[column].astype(str).str.strip()
        duplicated = values[values.ne("")].duplicated().sum()
        empty = values.eq("").sum()

        print(
            f"- {column}: vacíos={empty}, "
            f"duplicados_no_vacíos={duplicated}"
        )

    return row_count_ok


def main() -> None:
    all_valid = True

    for name, source in SOURCES.items():
        try:
            valid = inspect_source(name, source)
            all_valid = all_valid and valid
        except Exception as exc:
            all_valid = False
            print("\n" + "!" * 88)
            print(f"ERROR EN {name}")
            print(exc)

    print("\n" + "=" * 88)
    print("RESULTADO GENERAL")
    print("=" * 88)
    print("APROBADO" if all_valid else "REVISAR")


if __name__ == "__main__":
    main()