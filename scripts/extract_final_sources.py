from __future__ import annotations

import subprocess
from pathlib import Path


UPSTREAM_REPO = Path(".cache/upstream-textiles")

SOURCES = {
    Path("data/source/met/met_corpus_principal_v2_revisado.csv"): {
        "ref": "origin/mejora/met-flujo-reproducible-es",
        "path": "data/metadata/met_corpus_principal_v2_revisado.csv",
    },
    Path("data/source/cma/cma_corpus_principal_revisado.csv"): {
        "ref": "origin/mejora/cma-flujo-reproducible-es",
        "path": "data/metadata/cma_corpus_principal_revisado.csv",
    },
    Path("data/source/met/met_inventario_base.csv"): {
        "ref": "origin/mejora/met-flujo-reproducible-es",
        "path": (
            "data/processed/"
            "corpus_met_textiles_andinos_v1_inventario_base.csv"
        ),
    },
    Path("data/source/cma/cma_andes_textiles_candidates.csv"): {
        "ref": "origin/mejora/cma-flujo-reproducible-es",
        "path": "data/metadata/cma_andes_textiles_candidates.csv",
    },
}


def extract_file(
    destination: Path,
    ref: str,
    source_path: str,
) -> None:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(UPSTREAM_REPO),
            "show",
            f"{ref}:{source_path}",
        ],
        check=True,
        capture_output=True,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(result.stdout)

    print(
        f"Extraído: {source_path}\n"
        f"Destino: {destination}\n"
        f"Bytes: {destination.stat().st_size}\n"
    )


def main() -> None:
    for destination, source in SOURCES.items():
        extract_file(
            destination=destination,
            ref=source["ref"],
            source_path=source["path"],
        )


if __name__ == "__main__":
    main()