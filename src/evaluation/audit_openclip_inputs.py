"""Audita las entradas disponibles para la línea base OpenCLIP.

La auditoría identifica:

- rutas locales de imágenes;
- URLs de origen;
- imágenes existentes y legibles;
- campos textuales utilizables;
- registros listos para evaluación visual y multimodal.

No modifica el corpus ni descarga archivos.
"""

from __future__ import annotations

import sys
from pathlib import Path

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

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "culture_ground_truth_strict_v1.csv"
)

AUDIT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "openclip_input_audit_v1.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "openclip_input_audit_summary_v1.csv"
)

AUDIT_VERSION = "openclip_input_audit_v1"


LOCAL_IMAGE_COLUMN_CANDIDATES = (
    "local_image_path",
    "image_path",
    "downloaded_image_path",
    "image_local_path",
    "local_path",
    "image_file",
    "image_filename",
    "file_path",
)

URL_IMAGE_COLUMN_CANDIDATES = (
    "image_url",
    "primary_image_url",
    "image_url_original",
    "image_url_large",
    "web_image_url",
    "iiif_image_url",
    "iiif_url",
)

TEXT_FIELD_CANDIDATES = (
    "title",
    "object_name",
    "object_type",
    "classification",
    "culture",
    "culture_source",
    "culture_canonical",
    "period",
    "object_date",
    "date",
    "medium",
    "material",
    "materials",
    "technique",
    "description",
)


REQUIRED_CORPUS_COLUMNS = {
    "item_id",
    "museum",
    "culture_mapping_decision",
    "culture_strict_ground_truth_eligible",
}


AUDIT_COLUMNS = [
    "item_id",
    "museum",
    "culture_mapping_decision",
    "culture_strict_ground_truth_eligible",
    "culture_query_eligible",
    "selected_local_image_column",
    "local_image_value",
    "resolved_image_path",
    "local_image_reference_available",
    "local_image_exists",
    "local_image_readable",
    "image_width",
    "image_height",
    "image_format",
    "image_error",
    "selected_image_url_column",
    "image_source_url",
    "text_fields_used",
    "text_input",
    "text_character_count",
    "text_input_available",
    "openclip_image_ready",
    "openclip_text_ready",
    "openclip_multimodal_ready",
    "openclip_image_evaluation_ready",
    "openclip_multimodal_evaluation_ready",
    "audit_version",
]


def clean_text_value(value: object) -> str:
    """Convierte un valor en texto limpio."""

    if pd.isna(value):
        return ""

    return str(value).strip()


def parse_boolean_value(value: object) -> bool:
    """Convierte representaciones habituales en booleanos."""

    normalized = clean_text_value(value).lower()

    boolean_map = {
        "true": True,
        "1": True,
        "yes": True,
        "si": True,
        "sí": True,
        "false": False,
        "0": False,
        "no": False,
        "": False,
    }

    if normalized not in boolean_map:
        raise ValueError(
            f"Valor booleano no reconocido: {value}"
        )

    return boolean_map[normalized]


def detect_local_image_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Detecta posibles columnas con rutas locales."""

    detected: list[str] = []

    for column in LOCAL_IMAGE_COLUMN_CANDIDATES:
        if column in dataframe.columns:
            detected.append(column)

    for column in dataframe.columns:
        normalized = column.lower()

        inferred = (
            "image" in normalized
            and any(
                token in normalized
                for token in (
                    "path",
                    "file",
                    "local",
                    "download",
                )
            )
            and "url" not in normalized
        )

        if inferred and column not in detected:
            detected.append(column)

    return detected


def detect_image_url_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Detecta posibles columnas de URL de imagen."""

    detected: list[str] = []

    for column in URL_IMAGE_COLUMN_CANDIDATES:
        if column in dataframe.columns:
            detected.append(column)

    for column in dataframe.columns:
        normalized = column.lower()

        inferred = (
            (
                "image" in normalized
                and "url" in normalized
            )
            or "iiif" in normalized
        )

        if inferred and column not in detected:
            detected.append(column)

    return detected


def detect_text_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Selecciona los campos textuales disponibles."""

    return [
        column
        for column in TEXT_FIELD_CANDIDATES
        if column in dataframe.columns
    ]


def first_non_empty_value(
    row: pd.Series,
    columns: list[str],
) -> tuple[str, str]:
    """Devuelve la primera columna con un valor no vacío."""

    for column in columns:
        value = clean_text_value(
            row.get(column, "")
        )

        if value:
            return column, value

    return "", ""


def resolve_image_path(
    value: str,
) -> Path | None:
    """Resuelve una ruta absoluta o relativa al proyecto."""

    cleaned = clean_text_value(value)

    if not cleaned:
        return None

    if cleaned.lower().startswith("file://"):
        cleaned = cleaned[7:]

    path = Path(cleaned).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def inspect_image(
    path: Path | None,
) -> dict[str, object]:
    """Comprueba existencia y legibilidad de una imagen."""

    result: dict[str, object] = {
        "local_image_exists": False,
        "local_image_readable": False,
        "image_width": 0,
        "image_height": 0,
        "image_format": "",
        "image_error": "",
    }

    if path is None:
        return result

    if not path.exists():
        result["image_error"] = "file_not_found"
        return result

    if not path.is_file():
        result["image_error"] = "not_a_file"
        return result

    result["local_image_exists"] = True

    try:
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format or ""
            image.verify()

        result.update(
            {
                "local_image_readable": True,
                "image_width": int(width),
                "image_height": int(height),
                "image_format": image_format,
                "image_error": "",
            }
        )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        result["image_error"] = (
            f"{type(error).__name__}: {error}"
        )[:300]

    return result


def build_text_input(
    row: pd.Series,
    text_columns: list[str],
) -> tuple[str, str]:
    """Construye una descripción textual trazable.

    Se eliminan valores repetidos, pero se conserva el nombre
    del campo de origen.
    """

    fragments: list[str] = []
    used_columns: list[str] = []
    observed_values: set[str] = set()

    for column in text_columns:
        value = clean_text_value(
            row.get(column, "")
        )

        if not value:
            continue

        normalized_value = value.casefold()

        if normalized_value in observed_values:
            continue

        observed_values.add(normalized_value)
        used_columns.append(column)

        readable_name = column.replace(
            "_",
            " ",
        )

        fragments.append(
            f"{readable_name}: {value}"
        )

    return (
        ". ".join(fragments),
        " | ".join(used_columns),
    )


def validate_inputs(
    corpus: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> None:
    """Valida los artefactos de entrada."""

    missing = REQUIRED_CORPUS_COLUMNS.difference(
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

    required_ground_truth = {
        "item_id",
        "query_eligible",
    }

    missing_ground_truth = (
        required_ground_truth.difference(
            ground_truth.columns
        )
    )

    if missing_ground_truth:
        raise ValueError(
            "Faltan columnas en el ground truth: "
            f"{sorted(missing_ground_truth)}"
        )

    if ground_truth["item_id"].duplicated().any():
        raise ValueError(
            "El ground truth contiene item_id duplicados."
        )

    unknown_ground_truth_ids = (
        set(ground_truth["item_id"])
        - set(corpus["item_id"])
    )

    if unknown_ground_truth_ids:
        raise ValueError(
            "El ground truth contiene IDs ausentes del corpus: "
            f"{sorted(unknown_ground_truth_ids)[:10]}"
        )


def build_openclip_input_audit(
    corpus: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Construye la auditoría completa de entradas."""

    validate_inputs(
        corpus,
        ground_truth,
    )

    local_image_columns = (
        detect_local_image_columns(corpus)
    )

    image_url_columns = (
        detect_image_url_columns(corpus)
    )

    text_columns = detect_text_columns(
        corpus
    )

    query_eligibility = {
        clean_text_value(row.item_id): (
            parse_boolean_value(
                row.query_eligible
            )
        )
        for row in ground_truth[
            [
                "item_id",
                "query_eligible",
            ]
        ].itertuples(index=False)
    }

    rows: list[dict[str, object]] = []

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

        (
            local_image_column,
            local_image_value,
        ) = first_non_empty_value(
            row,
            local_image_columns,
        )

        resolved_path = resolve_image_path(
            local_image_value
        )

        image_inspection = inspect_image(
            resolved_path
        )

        (
            image_url_column,
            image_source_url,
        ) = first_non_empty_value(
            row,
            image_url_columns,
        )

        text_input, text_fields_used = (
            build_text_input(
                row,
                text_columns,
            )
        )

        strict_eligible = parse_boolean_value(
            row[
                "culture_strict_ground_truth_eligible"
            ]
        )

        culture_query_eligible = (
            query_eligibility.get(
                item_id,
                False,
            )
        )

        image_ready = bool(
            image_inspection[
                "local_image_readable"
            ]
        )

        text_ready = bool(text_input)

        rows.append(
            {
                "item_id": item_id,
                "museum": clean_text_value(
                    row["museum"]
                ),
                "culture_mapping_decision": (
                    clean_text_value(
                        row[
                            "culture_mapping_decision"
                        ]
                    )
                ),
                "culture_strict_ground_truth_eligible": (
                    strict_eligible
                ),
                "culture_query_eligible": (
                    culture_query_eligible
                ),
                "selected_local_image_column": (
                    local_image_column
                ),
                "local_image_value": (
                    local_image_value
                ),
                "resolved_image_path": (
                    str(resolved_path)
                    if resolved_path is not None
                    else ""
                ),
                "local_image_reference_available": (
                    bool(local_image_value)
                ),
                **image_inspection,
                "selected_image_url_column": (
                    image_url_column
                ),
                "image_source_url": (
                    image_source_url
                ),
                "text_fields_used": (
                    text_fields_used
                ),
                "text_input": text_input,
                "text_character_count": len(
                    text_input
                ),
                "text_input_available": (
                    text_ready
                ),
                "openclip_image_ready": (
                    image_ready
                ),
                "openclip_text_ready": (
                    text_ready
                ),
                "openclip_multimodal_ready": (
                    image_ready and text_ready
                ),
                "openclip_image_evaluation_ready": (
                    image_ready
                    and culture_query_eligible
                ),
                "openclip_multimodal_evaluation_ready": (
                    image_ready
                    and text_ready
                    and culture_query_eligible
                ),
                "audit_version": (
                    AUDIT_VERSION
                ),
            }
        )

    audit = pd.DataFrame(
        rows,
        columns=AUDIT_COLUMNS,
    )

    metadata = {
        "local_image_columns": (
            local_image_columns
        ),
        "image_url_columns": (
            image_url_columns
        ),
        "text_columns": text_columns,
    }

    return audit, metadata


def build_summary(
    audit: pd.DataFrame,
) -> pd.DataFrame:
    """Resume disponibilidad por museo y total."""

    def summarize_group(
        group: pd.DataFrame,
        museum: str,
    ) -> dict[str, object]:
        return {
            "museum": museum,
            "records": len(group),
            "strict_records": int(
                group[
                    "culture_strict_ground_truth_eligible"
                ].sum()
            ),
            "query_eligible_records": int(
                group[
                    "culture_query_eligible"
                ].sum()
            ),
            "local_image_references": int(
                group[
                    "local_image_reference_available"
                ].sum()
            ),
            "existing_local_images": int(
                group[
                    "local_image_exists"
                ].sum()
            ),
            "readable_local_images": int(
                group[
                    "local_image_readable"
                ].sum()
            ),
            "text_ready_records": int(
                group[
                    "openclip_text_ready"
                ].sum()
            ),
            "multimodal_ready_records": int(
                group[
                    "openclip_multimodal_ready"
                ].sum()
            ),
            "image_evaluation_ready_queries": int(
                group[
                    "openclip_image_evaluation_ready"
                ].sum()
            ),
            "multimodal_evaluation_ready_queries": int(
                group[
                    "openclip_multimodal_evaluation_ready"
                ].sum()
            ),
            "audit_version": AUDIT_VERSION,
        }

    rows = [
        summarize_group(group, museum)
        for museum, group in audit.groupby(
            "museum",
            sort=True,
        )
    ]

    rows.append(
        summarize_group(
            audit,
            "ALL",
        )
    )

    return pd.DataFrame(rows)


def write_csv(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    """Guarda un CSV reproducible."""

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
    summary: pd.DataFrame,
    metadata: dict[str, list[str]],
) -> None:
    """Imprime el diagnóstico de entradas."""

    print("\nAUDITORÍA DE ENTRADAS OPENCLIP")
    print("=" * 110)

    print(
        "Columnas de imagen local detectadas: "
        + (
            ", ".join(
                metadata["local_image_columns"]
            )
            or "NINGUNA"
        )
    )

    print(
        "Columnas URL detectadas: "
        + (
            ", ".join(
                metadata["image_url_columns"]
            )
            or "NINGUNA"
        )
    )

    print(
        "Campos textuales seleccionados: "
        + (
            ", ".join(
                metadata["text_columns"]
            )
            or "NINGUNO"
        )
    )

    print("\nCOBERTURA")
    print("=" * 110)
    print(summary.to_string(index=False))

    print("\nARTEFACTOS")
    print("=" * 110)
    print(AUDIT_PATH.relative_to(PROJECT_ROOT))
    print(SUMMARY_PATH.relative_to(PROJECT_ROOT))


def main() -> None:
    """Ejecuta la auditoría."""

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el corpus: {CORPUS_PATH}"
        )

    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el ground truth: "
            f"{GROUND_TRUTH_PATH}"
        )

    corpus = pd.read_csv(CORPUS_PATH)
    ground_truth = pd.read_csv(
        GROUND_TRUTH_PATH
    )

    audit, metadata = (
        build_openclip_input_audit(
            corpus,
            ground_truth,
        )
    )

    summary = build_summary(audit)

    write_csv(audit, AUDIT_PATH)
    write_csv(summary, SUMMARY_PATH)

    print_summary(
        summary,
        metadata,
    )


if __name__ == "__main__":
    main()