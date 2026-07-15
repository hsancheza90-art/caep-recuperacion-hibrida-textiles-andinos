from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pandas as pd


REPO = Path(".cache/upstream-textiles")

SOURCES = {
    "MET_curacion": {
        "ref": "origin/curacion/corpus-met-textiles-andinos-v1",
        "path": "data/processed/corpus_met_textiles_andinos_v1_principal.csv",
    },
    "CMA_curacion": {
        "ref": "origin/curacion/corpus-cma-textiles-andinos-v1",
        "path": "data/metadata/cma_corpus_principal.csv",
    },
}


def read_csv_from_git(ref: str, path: str) -> pd.DataFrame:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "show",
            f"{ref}:{path}",
        ],
        check=True,
        capture_output=True,
    )

    raw = result.stdout

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(encoding)
            frame = pd.read_csv(
                io.StringIO(text),
                dtype=str,
                keep_default_na=False,
            )
            if len(frame.columns) > 1:
                return frame
        except Exception:
            continue

    raise RuntimeError(f"No se pudo leer {ref}:{path}")


def report_coverage(name: str, frame: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)
    print(f"Registros: {len(frame)}")

    for column in ("url_imagen", "imagen_local", "url_objeto"):
        if column not in frame.columns:
            print(f"{column}: columna no disponible")
            continue

        values = frame[column].astype(str).str.strip()
        valid = values.ne("").sum()
        percentage = valid / len(frame) if len(frame) else 0

        print(
            f"{column}: {valid}/{len(frame)} "
            f"({percentage:.1%})"
        )


def main() -> None:
    for name, source in SOURCES.items():
        frame = read_csv_from_git(
            source["ref"],
            source["path"],
        )
        report_coverage(name, frame)


if __name__ == "__main__":
    main()