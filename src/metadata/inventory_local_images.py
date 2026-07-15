"""Inventaría imágenes locales y propone correspondencias con el corpus.

La búsqueda recorre el repositorio, excepto directorios técnicos como
.venv y .git. Las propuestas se generan sin modificar el corpus.

Prioridad de correspondencia:

1. nombre del archivo igual al item_id;
2. nombre igual a un identificador de fuente;
3. nombre del archivo igual al nombre extraído de image_url;
4. stem igual al stem extraído de image_url;
5. item_id contenido en el stem, como señal auxiliar.

Las coincidencias ambiguas no se aplican automáticamente.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paper_corpus_culture_enriched_v1.csv"
)

INVENTORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "local_image_inventory_v1.csv"
)

PROPOSALS_PATH = (
    PROJECT_ROOT
    / "config"
    / "mappings"
    / "image_local_path_proposals_v1.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "image_local_path_proposals_summary_v1.csv"
)

INVENTORY_VERSION = "local_image_inventory_v1"
PROPOSAL_VERSION = "image_local_path_proposals_v1"


SUPPORTED_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".tif",
    ".tiff",
    ".bmp",
}


EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
}


SOURCE_ID_COLUMN_CANDIDATES = (
    "source_key",
    "source_id",
    "object_id",
    "accession_number",
    "museum_object_id",
    "api_object_id",
)


INVENTORY_COLUMNS = [
    "relative_image_path",
    "absolute_image_path",
    "filename",
    "stem",
    "suffix",
    "filename_normalized",
    "stem_normalized",
    "size_bytes",
    "image_width",
    "image_height",
    "image_format",
    "image_readable",
    "image_error",
    "sha256",
    "inventory_version",
]


PROPOSAL_COLUMNS = [
    "item_id",
    "museum",
    "image_url",
    "proposal_status",
    "match_score",
    "match_rule",
    "candidate_count",
    "top_candidate_count",
    "candidate_paths",
    "proposed_image_local_path",
    "proposed_filename",
    "proposed_sha256",
    "proposed_image_width",
    "proposed_image_height",
    "proposal_version",
]

PUBLIC_INVENTORY_COLUMNS: tuple[str, ...] = (
    "relative_image_path",
    "filename",
    "stem",
    "suffix",
    "filename_normalized",
    "stem_normalized",
    "size_bytes",
    "image_width",
    "image_height",
    "image_format",
    "image_readable",
    "image_error",
    "sha256",
    "inventory_version",
)


def clean_text_value(value: object) -> str:
    """Convierte nulos en texto vacío y elimina espacios externos."""

    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_key(value: object) -> str:
    """Normaliza una clave para comparaciones deterministas."""

    text = clean_text_value(value).casefold()

    return re.sub(
        r"[^a-z0-9]+",
        "",
        text,
    )


def extract_url_filename(url: object) -> str:
    """Extrae el nombre de archivo de una URL."""

    cleaned = clean_text_value(url)

    if not cleaned:
        return ""

    parsed = urlparse(cleaned)

    return Path(
        unquote(parsed.path)
    ).name.strip()


def discover_image_paths(
    root: Path,
) -> list[Path]:
    """Descubre imágenes dentro del árbol del proyecto."""

    root = root.resolve()
    paths: list[Path] = []

    for current_root, directory_names, filenames in os.walk(root):
        directory_names[:] = sorted(
            directory_name
            for directory_name in directory_names
            if directory_name
            not in EXCLUDED_DIRECTORY_NAMES
        )

        current_path = Path(current_root)

        for filename in sorted(filenames):
            path = current_path / filename

            if (
                path.suffix.casefold()
                in SUPPORTED_IMAGE_SUFFIXES
            ):
                paths.append(path.resolve())

    return sorted(
        paths,
        key=lambda path: str(path).casefold(),
    )


def sha256_file(path: Path) -> str:
    """Calcula el hash SHA-256 de un archivo."""

    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def inspect_image_file(
    path: Path,
    root: Path,
) -> dict[str, object]:
    """Inspecciona una imagen y devuelve metadata reproducible."""

    path = path.resolve()
    root = root.resolve()

    try:
        relative_path = path.relative_to(root)
    except ValueError:
        relative_path = path

    result: dict[str, object] = {
        "relative_image_path": (
            relative_path.as_posix()
        ),
        "absolute_image_path": str(path),
        "filename": path.name,
        "stem": path.stem,
        "suffix": path.suffix.casefold(),
        "filename_normalized": normalize_key(
            path.name
        ),
        "stem_normalized": normalize_key(
            path.stem
        ),
        "size_bytes": 0,
        "image_width": 0,
        "image_height": 0,
        "image_format": "",
        "image_readable": False,
        "image_error": "",
        "sha256": "",
        "inventory_version": INVENTORY_VERSION,
    }

    if not path.exists():
        result["image_error"] = "file_not_found"
        return result

    if not path.is_file():
        result["image_error"] = "not_a_file"
        return result

    result["size_bytes"] = int(
        path.stat().st_size
    )

    try:
        result["sha256"] = sha256_file(
            path
        )

        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format or ""
            image.verify()

        result["image_width"] = int(width)
        result["image_height"] = int(height)
        result["image_format"] = image_format
        result["image_readable"] = True

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        result["image_error"] = (
            f"{type(error).__name__}: {error}"
        )[:300]

    return result


def build_image_inventory(
    image_paths: list[Path],
    root: Path,
) -> pd.DataFrame:
    """Construye el inventario de imágenes locales."""

    rows = [
        inspect_image_file(
            path,
            root,
        )
        for path in image_paths
    ]

    return pd.DataFrame(
        rows,
        columns=INVENTORY_COLUMNS,
    )


def validate_corpus(
    corpus: pd.DataFrame,
) -> None:
    """Valida el corpus antes de generar propuestas."""

    required_columns = {
        "item_id",
        "museum",
    }

    missing = required_columns.difference(
        corpus.columns
    )

    if missing:
        raise ValueError(
            "Faltan columnas requeridas en el corpus: "
            f"{sorted(missing)}"
        )

    if corpus["item_id"].duplicated().any():
        raise ValueError(
            "El corpus contiene item_id duplicados."
        )

    if corpus["item_id"].apply(
        clean_text_value
    ).eq("").any():
        raise ValueError(
            "El corpus contiene item_id vacíos."
        )


def register_candidates(
    candidates: dict[str, dict[str, object]],
    inventory: pd.DataFrame,
    row_indexes: list[int],
    score: int,
    rule: str,
) -> None:
    """Registra candidatos conservando la mejor puntuación."""

    for row_index in row_indexes:
        inventory_row = inventory.loc[
            row_index
        ]

        path = clean_text_value(
            inventory_row[
                "relative_image_path"
            ]
        )

        current = candidates.get(path)

        if current is None:
            candidates[path] = {
                "score": score,
                "rules": {rule},
                "row_index": row_index,
            }

        elif score > int(current["score"]):
            candidates[path] = {
                "score": score,
                "rules": {rule},
                "row_index": row_index,
            }

        elif score == int(current["score"]):
            current_rules = current["rules"]

            if isinstance(
                current_rules,
                set,
            ):
                current_rules.add(rule)


def build_indexes(
    inventory: pd.DataFrame,
) -> tuple[
    dict[str, list[int]],
    dict[str, list[int]],
]:
    """Construye índices por nombre y stem."""

    stem_index: dict[str, list[int]] = {}
    filename_index: dict[str, list[int]] = {}

    readable = inventory[
        inventory["image_readable"].astype(bool)
    ]

    for row_index, row in readable.iterrows():
        stem_key = clean_text_value(
            row["stem_normalized"]
        )

        filename_key = clean_text_value(
            row["filename_normalized"]
        )

        if stem_key:
            stem_index.setdefault(
                stem_key,
                [],
            ).append(row_index)

        if filename_key:
            filename_index.setdefault(
                filename_key,
                [],
            ).append(row_index)

    return stem_index, filename_index


def build_image_match_proposals(
    corpus: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Propone rutas locales para cada registro del corpus."""

    validate_corpus(corpus)

    if inventory.empty:
        empty_inventory = pd.DataFrame(
            columns=INVENTORY_COLUMNS
        )

        inventory = empty_inventory

    stem_index, filename_index = (
        build_indexes(inventory)
    )

    source_id_columns = [
        column
        for column in SOURCE_ID_COLUMN_CANDIDATES
        if column in corpus.columns
    ]

    rows: list[dict[str, object]] = []

    readable_inventory = inventory[
        inventory["image_readable"].astype(bool)
    ]

    for source_row in corpus.itertuples(
        index=False,
        name=None,
    ):
        row = pd.Series(
            source_row,
            index=corpus.columns,
        )

        item_id = clean_text_value(
            row["item_id"]
        )

        museum = clean_text_value(
            row["museum"]
        )

        image_url = clean_text_value(
            row.get("image_url", "")
        )

        candidates: dict[
            str,
            dict[str, object],
        ] = {}

        item_id_key = normalize_key(
            item_id
        )

        register_candidates(
            candidates,
            inventory,
            stem_index.get(
                item_id_key,
                [],
            ),
            score=100,
            rule="exact_item_id_stem",
        )

        for column in source_id_columns:
            source_value = clean_text_value(
                row.get(column, "")
            )

            source_key = normalize_key(
                source_value
            )

            if not source_key:
                continue

            register_candidates(
                candidates,
                inventory,
                stem_index.get(
                    source_key,
                    [],
                ),
                score=95,
                rule=f"exact_{column}_stem",
            )

        url_filename = extract_url_filename(
            image_url
        )

        url_filename_key = normalize_key(
            url_filename
        )

        url_stem_key = normalize_key(
            Path(url_filename).stem
            if url_filename
            else ""
        )

        register_candidates(
            candidates,
            inventory,
            filename_index.get(
                url_filename_key,
                [],
            ),
            score=90,
            rule="exact_url_filename",
        )

        register_candidates(
            candidates,
            inventory,
            stem_index.get(
                url_stem_key,
                [],
            ),
            score=85,
            rule="exact_url_stem",
        )

        # Señal auxiliar para archivos que añaden prefijos
        # o sufijos al item_id. Solo se usa con claves
        # suficientemente largas.
        if (
            not candidates
            and len(item_id_key) >= 6
        ):
            substring_matches = []

            for row_index, image_row in (
                readable_inventory.iterrows()
            ):
                stem_key = clean_text_value(
                    image_row[
                        "stem_normalized"
                    ]
                )

                if item_id_key in stem_key:
                    substring_matches.append(
                        row_index
                    )

            register_candidates(
                candidates,
                inventory,
                substring_matches,
                score=70,
                rule="item_id_substring_in_stem",
            )

        ordered_candidates = sorted(
            candidates.items(),
            key=lambda entry: (
                -int(entry[1]["score"]),
                entry[0].casefold(),
            ),
        )

        if not ordered_candidates:
            rows.append(
                {
                    "item_id": item_id,
                    "museum": museum,
                    "image_url": image_url,
                    "proposal_status": "unmatched",
                    "match_score": 0,
                    "match_rule": "",
                    "candidate_count": 0,
                    "top_candidate_count": 0,
                    "candidate_paths": "",
                    "proposed_image_local_path": "",
                    "proposed_filename": "",
                    "proposed_sha256": "",
                    "proposed_image_width": 0,
                    "proposed_image_height": 0,
                    "proposal_version": (
                        PROPOSAL_VERSION
                    ),
                }
            )

            continue

        top_score = int(
            ordered_candidates[0][1][
                "score"
            ]
        )

        top_candidates = [
            candidate
            for candidate in ordered_candidates
            if int(candidate[1]["score"])
            == top_score
        ]

        top_candidate_count = len(
            top_candidates
        )

        candidate_paths = " | ".join(
            path
            for path, _ in ordered_candidates
        )

        top_rules: set[str] = set()

        for _, candidate_data in top_candidates:
            rules = candidate_data["rules"]

            if isinstance(rules, set):
                top_rules.update(rules)

        match_rule = " | ".join(
            sorted(top_rules)
        )

        if top_candidate_count == 1:
            proposal_status = "matched_unique"

            proposed_path, proposed_data = (
                top_candidates[0]
            )

            proposed_row = inventory.loc[
                int(
                    proposed_data[
                        "row_index"
                    ]
                )
            ]

            proposed_filename = (
                clean_text_value(
                    proposed_row["filename"]
                )
            )

            proposed_sha256 = (
                clean_text_value(
                    proposed_row["sha256"]
                )
            )

            proposed_width = int(
                proposed_row["image_width"]
            )

            proposed_height = int(
                proposed_row["image_height"]
            )

        else:
            proposal_status = "ambiguous"
            proposed_path = ""
            proposed_filename = ""
            proposed_sha256 = ""
            proposed_width = 0
            proposed_height = 0

        rows.append(
            {
                "item_id": item_id,
                "museum": museum,
                "image_url": image_url,
                "proposal_status": (
                    proposal_status
                ),
                "match_score": top_score,
                "match_rule": match_rule,
                "candidate_count": len(
                    ordered_candidates
                ),
                "top_candidate_count": (
                    top_candidate_count
                ),
                "candidate_paths": (
                    candidate_paths
                ),
                "proposed_image_local_path": (
                    proposed_path
                ),
                "proposed_filename": (
                    proposed_filename
                ),
                "proposed_sha256": (
                    proposed_sha256
                ),
                "proposed_image_width": (
                    proposed_width
                ),
                "proposed_image_height": (
                    proposed_height
                ),
                "proposal_version": (
                    PROPOSAL_VERSION
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=PROPOSAL_COLUMNS,
    )


def build_proposal_summary(
    proposals: pd.DataFrame,
) -> pd.DataFrame:
    """Resume las propuestas por museo y estado."""

    grouped = (
        proposals.groupby(
            [
                "museum",
                "proposal_status",
            ],
            dropna=False,
        )
        .size()
        .rename("records")
        .reset_index()
    )

    totals = (
        proposals.groupby(
            "proposal_status",
            dropna=False,
        )
        .size()
        .rename("records")
        .reset_index()
    )

    totals.insert(
        0,
        "museum",
        "ALL",
    )

    summary = pd.concat(
        [
            grouped,
            totals,
        ],
        ignore_index=True,
    )

    return summary.sort_values(
        [
            "museum",
            "proposal_status",
        ],
        kind="stable",
    ).reset_index(drop=True)


def write_csv(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    """Escribe un CSV estable."""

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


def print_summary(
    inventory: pd.DataFrame,
    proposals: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Imprime el resumen de inventario y correspondencias."""

    readable_count = int(
        inventory[
            "image_readable"
        ].astype(bool).sum()
    )

    duplicate_hash_count = int(
        inventory.loc[
            inventory["sha256"].ne(""),
            "sha256",
        ].duplicated(
            keep=False
        ).sum()
    )

    print("\nINVENTARIO LOCAL DE IMÁGENES")
    print("=" * 100)
    print(
        f"Archivos encontrados: {len(inventory)}"
    )
    print(
        f"Imágenes legibles: {readable_count}"
    )
    print(
        "Archivos pertenecientes a grupos con "
        f"hash duplicado: {duplicate_hash_count}"
    )

    print("\nPROPUESTAS DE CORRESPONDENCIA")
    print("=" * 100)
    print(summary.to_string(index=False))

    unique_matches = proposals[
        proposals[
            "proposal_status"
        ].eq("matched_unique")
    ]

    print("\nCOINCIDENCIAS ÚNICAS POR REGLA")
    print("=" * 100)

    if unique_matches.empty:
        print(
            "No se encontraron coincidencias únicas."
        )
    else:
        print(
            unique_matches[
                "match_rule"
            ]
            .value_counts()
            .rename_axis("match_rule")
            .reset_index(name="records")
            .to_string(index=False)
        )

    print("\nARTEFACTOS")
    print("=" * 100)
    print(
        INVENTORY_PATH.relative_to(
            PROJECT_ROOT
        )
    )
    print(
        PROPOSALS_PATH.relative_to(
            PROJECT_ROOT
        )
    )
    print(
        SUMMARY_PATH.relative_to(
            PROJECT_ROOT
        )
    )

def prepare_public_inventory(
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Prepara el inventario portable que será versionado.

    La ruta absoluta puede utilizarse internamente durante la
    ejecución, pero no debe almacenarse en los artefactos públicos,
    porque depende del equipo y del sistema operativo.
    """

    missing_columns = [
        column
        for column in PUBLIC_INVENTORY_COLUMNS
        if column not in inventory.columns
    ]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(
            "El inventario no contiene todas las columnas "
            f"públicas requeridas: {missing}"
        )

    return inventory.loc[
        :,
        list(PUBLIC_INVENTORY_COLUMNS),
    ].copy()

def main() -> None:
    """Ejecuta el inventario y genera las propuestas."""

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el corpus: {CORPUS_PATH}"
        )

    corpus = pd.read_csv(CORPUS_PATH)

    image_paths = discover_image_paths(
        PROJECT_ROOT
    )

    inventory = build_image_inventory(
        image_paths,
        PROJECT_ROOT,
    )

    proposals = build_image_match_proposals(
        corpus,
        inventory,
    )

    summary = build_proposal_summary(
        proposals
    )

    public_inventory = prepare_public_inventory(
        inventory
    )

    public_inventory.to_csv(
        INVENTORY_PATH,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    write_csv(
        proposals,
        PROPOSALS_PATH,
    )

    write_csv(
        summary,
        SUMMARY_PATH,
    )

    print_summary(
        inventory,
        proposals,
        summary,
    )


if __name__ == "__main__":
    main()