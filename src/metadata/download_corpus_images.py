"""Descarga y valida las imágenes referenciadas por el corpus.

Características:

- usa únicamente la columna image_url;
- admite solo URLs HTTP y HTTPS;
- aplica tiempo límite y reintentos controlados;
- limita el tamaño máximo descargado;
- valida cada archivo mediante Pillow;
- calcula SHA-256;
- escribe los archivos de forma atómica;
- conserva un manifiesto de éxitos y errores;
- no modifica el corpus de entrada;
- no intenta evadir bloqueos HTTP 403.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
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

IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "images"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "image_download_manifest_v1.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "image_download_summary_v1.csv"
)

DOWNLOAD_VERSION = "image_download_v1"

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_RETRIES = 3
DEFAULT_MAX_BYTES = 25 * 1024 * 1024

USER_AGENT = (
    "UNI-CC-Textiles-Andinos/1.0 "
    "(academic-research; image-corpus-curation)"
)

RETRYABLE_HTTP_STATUSES = {
    408,
    429,
    500,
    502,
    503,
    504,
}

IMAGE_FORMAT_EXTENSIONS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "TIFF": ".tif",
    "BMP": ".bmp",
}

SUPPORTED_LOCAL_SUFFIXES = set(
    IMAGE_FORMAT_EXTENSIONS.values()
) | {
    ".jpeg",
    ".tiff",
}


MANIFEST_COLUMNS = [
    "item_id",
    "museum",
    "image_url",
    "download_status",
    "acquisition_action",
    "failure_code",
    "failure_detail",
    "http_status",
    "attempt_count",
    "final_url",
    "content_type",
    "image_local_path",
    "filename",
    "image_bytes",
    "image_width",
    "image_height",
    "image_format",
    "sha256",
    "download_version",
]


class ImageDownloadError(RuntimeError):
    """Error controlado durante la adquisición de una imagen."""

    def __init__(
        self,
        failure_code: str,
        message: str,
        *,
        http_status: int = 0,
        attempt_count: int = 0,
    ) -> None:
        super().__init__(message)

        self.failure_code = failure_code
        self.http_status = int(http_status)
        self.attempt_count = int(attempt_count)


def clean_text_value(value: object) -> str:
    """Convierte nulos en texto vacío y elimina espacios."""

    if pd.isna(value):
        return ""

    return str(value).strip()


def sanitize_filename_component(value: object) -> str:
    """Convierte una clave en un nombre de archivo seguro."""

    cleaned = clean_text_value(value)

    cleaned = re.sub(
        r'[<>:"/\\|?*\x00-\x1f]+',
        "_",
        cleaned,
    )

    cleaned = re.sub(
        r"\s+",
        "_",
        cleaned,
    )

    # Colapsa secuencias como ___ en un único guion bajo.
    cleaned = re.sub(
        r"_+",
        "_",
        cleaned,
    )

    cleaned = cleaned.strip(
        " ._"
    )

    if not cleaned:
        raise ValueError(
            "No se puede construir un nombre de archivo vacío."
        )

    return cleaned


def validate_http_url(url: str) -> None:
    """Valida que la URL use HTTP o HTTPS."""

    parsed = urlparse(url)

    if parsed.scheme.lower() not in {
        "http",
        "https",
    }:
        raise ImageDownloadError(
            "unsupported_url_scheme",
            (
                "La URL no utiliza HTTP o HTTPS: "
                f"{parsed.scheme or 'sin esquema'}"
            ),
        )

    if not parsed.netloc:
        raise ImageDownloadError(
            "invalid_url",
            "La URL no contiene un host válido.",
        )


def validate_corpus(corpus: pd.DataFrame) -> None:
    """Valida el corpus antes de la descarga."""

    required_columns = {
        "item_id",
        "museum",
        "image_url",
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

    for column in [
        "item_id",
        "museum",
    ]:
        empty = (
            corpus[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
        )

        if empty.any():
            raise ValueError(
                f"El corpus contiene valores vacíos en {column}."
            )


def sha256_bytes(content: bytes) -> str:
    """Calcula el hash SHA-256 de contenido binario."""

    return hashlib.sha256(
        content
    ).hexdigest()


def inspect_image_bytes(
    content: bytes,
) -> dict[str, object]:
    """Valida una imagen y obtiene sus propiedades."""

    if not content:
        raise ImageDownloadError(
            "empty_response",
            "La respuesta no contiene datos.",
        )

    try:
        with Image.open(
            BytesIO(content)
        ) as image:
            width, height = image.size
            image_format = (
                image.format or ""
            ).upper()

            image.verify()

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        raise ImageDownloadError(
            "invalid_image_content",
            (
                f"{type(error).__name__}: "
                f"{error}"
            ),
        ) from error

    if image_format not in IMAGE_FORMAT_EXTENSIONS:
        raise ImageDownloadError(
            "unsupported_image_format",
            (
                "Formato de imagen no soportado: "
                f"{image_format or 'desconocido'}"
            ),
        )

    return {
        "image_width": int(width),
        "image_height": int(height),
        "image_format": image_format,
        "extension": (
            IMAGE_FORMAT_EXTENSIONS[
                image_format
            ]
        ),
        "image_bytes": len(content),
        "sha256": sha256_bytes(content),
    }


def inspect_local_image(
    path: Path,
) -> dict[str, object]:
    """Valida una imagen ya presente en disco."""

    if not path.exists():
        raise ImageDownloadError(
            "local_file_not_found",
            f"No existe el archivo: {path}",
        )

    if not path.is_file():
        raise ImageDownloadError(
            "local_path_not_file",
            f"La ruta no es un archivo: {path}",
        )

    try:
        content = path.read_bytes()

    except OSError as error:
        raise ImageDownloadError(
            "local_file_read_error",
            str(error),
        ) from error

    return inspect_image_bytes(content)


def normalize_content_type(
    value: object,
) -> str:
    """Normaliza el encabezado Content-Type."""

    return (
        clean_text_value(value)
        .split(
            ";",
            maxsplit=1,
        )[0]
        .strip()
        .lower()
    )


def fetch_image_bytes(
    session: requests.Session,
    url: str,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    max_bytes: int = DEFAULT_MAX_BYTES,
    sleep_function=time.sleep,
) -> dict[str, object]:
    """Descarga contenido binario con reintentos controlados."""

    validate_http_url(url)

    if retries < 0:
        raise ValueError(
            "retries no puede ser negativo."
        )

    if max_bytes <= 0:
        raise ValueError(
            "max_bytes debe ser mayor que cero."
        )

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": (
            "image/avif,image/webp,image/apng,"
            "image/svg+xml,image/*,*/*;q=0.8"
        ),
    }

    last_error: ImageDownloadError | None = None

    for attempt in range(
        1,
        retries + 2,
    ):
        response = None

        try:
            response = session.get(
                url,
                headers=headers,
                stream=True,
                timeout=timeout_seconds,
                allow_redirects=True,
            )

            http_status = int(
                response.status_code
            )

            final_url = clean_text_value(
                getattr(
                    response,
                    "url",
                    url,
                )
            ) or url

            content_type = normalize_content_type(
                response.headers.get(
                    "Content-Type",
                    "",
                )
            )

            if (
                http_status
                in RETRYABLE_HTTP_STATUSES
                and attempt <= retries
            ):
                last_error = ImageDownloadError(
                    "retryable_http_error",
                    f"HTTP {http_status}",
                    http_status=http_status,
                    attempt_count=attempt,
                )

                sleep_function(
                    min(
                        2 ** (attempt - 1),
                        8,
                    )
                )

                continue

            if http_status != 200:
                raise ImageDownloadError(
                    "http_error",
                    f"HTTP {http_status}",
                    http_status=http_status,
                    attempt_count=attempt,
                )

            allowed_content_type = (
                not content_type
                or content_type.startswith(
                    "image/"
                )
                or content_type
                == "application/octet-stream"
            )

            if not allowed_content_type:
                raise ImageDownloadError(
                    "invalid_content_type",
                    (
                        "El servidor devolvió "
                        f"{content_type}"
                    ),
                    http_status=http_status,
                    attempt_count=attempt,
                )

            chunks: list[bytes] = []
            total_bytes = 0

            for chunk in response.iter_content(
                chunk_size=64 * 1024
            ):
                if not chunk:
                    continue

                total_bytes += len(chunk)

                if total_bytes > max_bytes:
                    raise ImageDownloadError(
                        "image_too_large",
                        (
                            "La imagen supera el máximo "
                            f"de {max_bytes} bytes."
                        ),
                        http_status=http_status,
                        attempt_count=attempt,
                    )

                chunks.append(chunk)

            content = b"".join(chunks)

            if not content:
                raise ImageDownloadError(
                    "empty_response",
                    "La respuesta no contiene datos.",
                    http_status=http_status,
                    attempt_count=attempt,
                )

            return {
                "content": content,
                "http_status": http_status,
                "attempt_count": attempt,
                "final_url": final_url,
                "content_type": content_type,
            }

        except requests.RequestException as error:
            last_error = ImageDownloadError(
                "network_error",
                (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
                attempt_count=attempt,
            )

            if attempt <= retries:
                sleep_function(
                    min(
                        2 ** (attempt - 1),
                        8,
                    )
                )

                continue

            raise last_error from error

        finally:
            if response is not None:
                close_method = getattr(
                    response,
                    "close",
                    None,
                )

                if callable(close_method):
                    close_method()

    if last_error is not None:
        raise last_error

    raise ImageDownloadError(
        "unknown_download_error",
        "La descarga terminó sin resultado.",
    )


def relative_or_absolute_path(
    path: Path,
) -> str:
    """Devuelve una ruta relativa al proyecto cuando sea posible."""

    resolved = path.resolve()

    try:
        return resolved.relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:
        return resolved.as_posix()


def find_existing_images(
    image_root: Path,
    museum: str,
    item_id: str,
) -> list[Path]:
    """Busca imágenes existentes para un item_id."""

    museum_directory = (
        image_root
        / sanitize_filename_component(
            museum
        )
    )

    if not museum_directory.exists():
        return []

    safe_item_id = (
        sanitize_filename_component(
            item_id
        )
    )

    candidates = [
        path.resolve()
        for path in museum_directory.glob(
            f"{safe_item_id}.*"
        )
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_LOCAL_SUFFIXES
        )
    ]

    return sorted(
        candidates,
        key=lambda path: str(path).casefold(),
    )


def build_output_path(
    image_root: Path,
    museum: str,
    item_id: str,
    extension: str,
) -> Path:
    """Construye la ruta de salida de una imagen."""

    museum_directory = (
        image_root
        / sanitize_filename_component(
            museum
        )
    )

    museum_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        sanitize_filename_component(
            item_id
        )
        + extension
    )

    return museum_directory / filename


def write_bytes_atomically(
    path: Path,
    content: bytes,
) -> None:
    """Escribe un archivo usando sustitución atómica."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        path.name + ".part"
    )

    temporary_path.write_bytes(
        content
    )

    temporary_path.replace(path)


def empty_manifest_row(
    item_id: str,
    museum: str,
    image_url: str,
) -> dict[str, object]:
    """Construye una fila vacía del manifiesto."""

    return {
        "item_id": item_id,
        "museum": museum,
        "image_url": image_url,
        "download_status": "",
        "acquisition_action": "none",
        "failure_code": "",
        "failure_detail": "",
        "http_status": 0,
        "attempt_count": 0,
        "final_url": "",
        "content_type": "",
        "image_local_path": "",
        "filename": "",
        "image_bytes": 0,
        "image_width": 0,
        "image_height": 0,
        "image_format": "",
        "sha256": "",
        "download_version": DOWNLOAD_VERSION,
    }


def process_corpus_row(
    row: pd.Series,
    *,
    image_root: Path,
    session: requests.Session,
    timeout_seconds: int,
    retries: int,
    max_bytes: int,
    sleep_function=time.sleep,
) -> dict[str, object]:
    """Procesa un registro individual del corpus."""

    item_id = clean_text_value(
        row["item_id"]
    )

    museum = clean_text_value(
        row["museum"]
    )

    image_url = clean_text_value(
        row["image_url"]
    )

    result = empty_manifest_row(
        item_id,
        museum,
        image_url,
    )

    existing_images = find_existing_images(
        image_root,
        museum,
        item_id,
    )

    if len(existing_images) > 1:
        result.update(
            {
                "download_status": (
                    "ambiguous_existing"
                ),
                "failure_code": (
                    "multiple_local_images"
                ),
                "failure_detail": " | ".join(
                    relative_or_absolute_path(
                        path
                    )
                    for path in existing_images
                ),
            }
        )

        return result

    if len(existing_images) == 1:
        existing_path = existing_images[0]

        try:
            metadata = inspect_local_image(
                existing_path
            )

            result.update(
                {
                    "download_status": "available",
                    "acquisition_action": "reused",
                    "image_local_path": (
                        relative_or_absolute_path(
                            existing_path
                        )
                    ),
                    "filename": (
                        existing_path.name
                    ),
                    **{
                        key: metadata[key]
                        for key in [
                            "image_bytes",
                            "image_width",
                            "image_height",
                            "image_format",
                            "sha256",
                        ]
                    },
                }
            )

        except ImageDownloadError as error:
            result.update(
                {
                    "download_status": (
                        "invalid_existing"
                    ),
                    "failure_code": (
                        error.failure_code
                    ),
                    "failure_detail": str(error),
                }
            )

        return result

    if not image_url:
        result.update(
            {
                "download_status": "missing_url",
                "failure_code": "missing_image_url",
                "failure_detail": (
                    "El registro no contiene image_url."
                ),
            }
        )

        return result

    try:
        fetched = fetch_image_bytes(
            session,
            image_url,
            timeout_seconds=timeout_seconds,
            retries=retries,
            max_bytes=max_bytes,
            sleep_function=sleep_function,
        )

        content = fetched["content"]

        if not isinstance(
            content,
            bytes,
        ):
            raise ImageDownloadError(
                "invalid_binary_response",
                "La respuesta no es contenido binario.",
            )

        metadata = inspect_image_bytes(
            content
        )

        output_path = build_output_path(
            image_root,
            museum,
            item_id,
            str(metadata["extension"]),
        )

        write_bytes_atomically(
            output_path,
            content,
        )

        # Revalidación desde el archivo escrito.
        disk_metadata = inspect_local_image(
            output_path
        )

        if (
            disk_metadata["sha256"]
            != metadata["sha256"]
        ):
            raise ImageDownloadError(
                "written_hash_mismatch",
                (
                    "El hash del archivo escrito "
                    "no coincide con la descarga."
                ),
            )

        result.update(
            {
                "download_status": "available",
                "acquisition_action": "downloaded",
                "http_status": int(
                    fetched["http_status"]
                ),
                "attempt_count": int(
                    fetched["attempt_count"]
                ),
                "final_url": clean_text_value(
                    fetched["final_url"]
                ),
                "content_type": clean_text_value(
                    fetched["content_type"]
                ),
                "image_local_path": (
                    relative_or_absolute_path(
                        output_path
                    )
                ),
                "filename": output_path.name,
                **{
                    key: disk_metadata[key]
                    for key in [
                        "image_bytes",
                        "image_width",
                        "image_height",
                        "image_format",
                        "sha256",
                    ]
                },
            }
        )

    except ImageDownloadError as error:
        result.update(
            {
                "download_status": "failed",
                "failure_code": error.failure_code,
                "failure_detail": str(error)[:500],
                "http_status": error.http_status,
                "attempt_count": (
                    error.attempt_count
                ),
            }
        )

    return result


def download_corpus_images(
    corpus: pd.DataFrame,
    *,
    image_root: Path = IMAGE_ROOT,
    museum: str | None = None,
    limit: int | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    max_bytes: int = DEFAULT_MAX_BYTES,
    session: requests.Session | None = None,
    sleep_function=time.sleep,
) -> pd.DataFrame:
    """Descarga las imágenes de un subconjunto del corpus."""

    validate_corpus(corpus)

    selected = corpus.copy()

    if museum:
        normalized_museum = (
            clean_text_value(museum)
            .upper()
        )

        selected = selected[
            selected["museum"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
            .eq(normalized_museum)
        ].copy()

    if limit is not None:
        if limit <= 0:
            raise ValueError(
                "limit debe ser mayor que cero."
            )

        selected = selected.head(
            int(limit)
        ).copy()

    owns_session = session is None

    if session is None:
        session = requests.Session()

    rows: list[dict[str, object]] = []

    try:
        for _, row in selected.iterrows():
            rows.append(
                process_corpus_row(
                    row,
                    image_root=image_root,
                    session=session,
                    timeout_seconds=(
                        timeout_seconds
                    ),
                    retries=retries,
                    max_bytes=max_bytes,
                    sleep_function=(
                        sleep_function
                    ),
                )
            )

    finally:
        if owns_session:
            session.close()

    return pd.DataFrame(
        rows,
        columns=MANIFEST_COLUMNS,
    )


def build_download_summary(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Resume el resultado por museo y estado."""

    grouped = (
        manifest.groupby(
            [
                "museum",
                "download_status",
            ],
            dropna=False,
        )
        .size()
        .rename("records")
        .reset_index()
    )

    totals = (
        manifest.groupby(
            "download_status",
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
            "download_status",
        ],
        kind="stable",
    ).reset_index(drop=True)


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
    manifest: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Imprime el resultado de la descarga."""

    print("\nDESCARGA CONTROLADA DE IMÁGENES")
    print("=" * 100)
    print(summary.to_string(index=False))

    available = manifest[
        manifest["download_status"].eq(
            "available"
        )
    ]

    print("\nCOBERTURA")
    print("=" * 100)
    print(
        f"Registros procesados: {len(manifest)}"
    )
    print(
        f"Imágenes disponibles: {len(available)}"
    )
    print(
        "Bytes disponibles: "
        f"{int(available['image_bytes'].sum())}"
    )

    failures = manifest[
        manifest["download_status"].eq(
            "failed"
        )
    ]

    print("\nERRORES POR TIPO")
    print("=" * 100)

    if failures.empty:
        print("No se registraron fallos.")
    else:
        print(
            failures[
                "failure_code"
            ]
            .value_counts()
            .rename_axis("failure_code")
            .reset_index(name="records")
            .to_string(index=False)
        )

    print("\nARTEFACTOS")
    print("=" * 100)
    print(
        IMAGE_ROOT.relative_to(
            PROJECT_ROOT
        )
    )
    print(
        MANIFEST_PATH.relative_to(
            PROJECT_ROOT
        )
    )
    print(
        SUMMARY_PATH.relative_to(
            PROJECT_ROOT
        )
    )


def parse_args() -> argparse.Namespace:
    """Procesa argumentos de línea de comandos."""

    parser = argparse.ArgumentParser(
        description=(
            "Descarga imágenes del corpus de forma "
            "controlada y trazable."
        )
    )

    parser.add_argument(
        "--museum",
        choices=[
            "MET",
            "CMA",
        ],
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
    )

    parser.add_argument(
        "--max-bytes-mb",
        type=int,
        default=25,
    )

    return parser.parse_args()


def main() -> None:
    """Ejecuta la descarga."""

    args = parse_args()

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el corpus: {CORPUS_PATH}"
        )

    corpus = pd.read_csv(
        CORPUS_PATH
    )

    manifest = download_corpus_images(
        corpus,
        image_root=IMAGE_ROOT,
        museum=args.museum,
        limit=args.limit,
        timeout_seconds=args.timeout,
        retries=args.retries,
        max_bytes=(
            args.max_bytes_mb
            * 1024
            * 1024
        ),
    )

    summary = build_download_summary(
        manifest
    )

    write_csv(
        manifest,
        MANIFEST_PATH,
    )

    write_csv(
        summary,
        SUMMARY_PATH,
    )

    print_summary(
        manifest,
        summary,
    )


if __name__ == "__main__":
    main()