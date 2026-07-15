"""Pruebas de la descarga controlada de imágenes."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.metadata.download_corpus_images import (
    MANIFEST_PATH,
    SUMMARY_PATH,
    ImageDownloadError,
    build_download_summary,
    download_corpus_images,
    fetch_image_bytes,
    inspect_image_bytes,
    sanitize_filename_component,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paper_corpus_culture_enriched_v1.csv"
)


def make_image_bytes(
    image_format: str = "PNG",
    size: tuple[int, int] = (20, 10),
) -> bytes:
    buffer = BytesIO()

    Image.new(
        "RGB",
        size,
    ).save(
        buffer,
        format=image_format,
    )

    return buffer.getvalue()


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        content: bytes = b"",
        content_type: str = "image/png",
        url: str = "https://example.org/image.png",
    ) -> None:
        self.status_code = status_code
        self._content = content
        self.headers = {
            "Content-Type": content_type,
        }
        self.url = url
        self.closed = False

    def iter_content(
        self,
        chunk_size: int,
    ):
        del chunk_size

        if self._content:
            yield self._content

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(
        self,
        responses: list[FakeResponse],
    ) -> None:
        self.responses = list(responses)
        self.calls = 0

    def get(self, *args, **kwargs):
        del args, kwargs

        self.calls += 1

        if not self.responses:
            raise RuntimeError(
                "No quedan respuestas simuladas."
            )

        return self.responses.pop(0)

    def close(self) -> None:
        pass


def test_filename_component_is_sanitized() -> None:
    assert sanitize_filename_component(
        "MET:001 / object"
    ) == "MET_001_object"


def test_valid_image_bytes_are_inspected() -> None:
    metadata = inspect_image_bytes(
        make_image_bytes(
            "PNG",
            (32, 16),
        )
    )

    assert metadata["image_width"] == 32
    assert metadata["image_height"] == 16
    assert metadata["image_format"] == "PNG"
    assert metadata["extension"] == ".png"
    assert metadata["sha256"]


def test_corrupt_bytes_are_rejected() -> None:
    with pytest.raises(
        ImageDownloadError,
        match="cannot identify image",
    ):
        inspect_image_bytes(
            b"not an image"
        )


def test_unsupported_url_scheme_is_rejected() -> None:
    session = FakeSession([])

    with pytest.raises(
        ImageDownloadError,
        match="HTTP o HTTPS",
    ):
        fetch_image_bytes(
            session,
            "file:///tmp/image.jpg",
            sleep_function=lambda _: None,
        )


def test_successful_fetch_returns_content() -> None:
    image_content = make_image_bytes()

    session = FakeSession(
        [
            FakeResponse(
                content=image_content,
            )
        ]
    )

    result = fetch_image_bytes(
        session,
        "https://example.org/image.png",
        sleep_function=lambda _: None,
    )

    assert result["content"] == image_content
    assert result["http_status"] == 200
    assert result["attempt_count"] == 1


def test_non_image_content_type_is_rejected() -> None:
    session = FakeSession(
        [
            FakeResponse(
                content=b"<html></html>",
                content_type="text/html",
            )
        ]
    )

    with pytest.raises(
        ImageDownloadError,
        match="text/html",
    ):
        fetch_image_bytes(
            session,
            "https://example.org/image",
            sleep_function=lambda _: None,
        )


def test_image_size_limit_is_enforced() -> None:
    image_content = make_image_bytes()

    session = FakeSession(
        [
            FakeResponse(
                content=image_content,
            )
        ]
    )

    with pytest.raises(
        ImageDownloadError,
        match="supera el máximo",
    ):
        fetch_image_bytes(
            session,
            "https://example.org/image.png",
            max_bytes=10,
            sleep_function=lambda _: None,
        )


def test_missing_url_is_recorded(
    tmp_path: Path,
) -> None:
    corpus = pd.DataFrame(
        [
            {
                "item_id": "MET_001",
                "museum": "MET",
                "image_url": "",
            }
        ]
    )

    manifest = download_corpus_images(
        corpus,
        image_root=tmp_path,
        session=FakeSession([]),
        sleep_function=lambda _: None,
    )

    row = manifest.iloc[0]

    assert row["download_status"] == "missing_url"
    assert row["failure_code"] == "missing_image_url"


def test_successful_download_writes_image(
    tmp_path: Path,
) -> None:
    image_content = make_image_bytes(
        "PNG",
        (40, 30),
    )

    corpus = pd.DataFrame(
        [
            {
                "item_id": "MET_001",
                "museum": "MET",
                "image_url": (
                    "https://example.org/image.png"
                ),
            }
        ]
    )

    manifest = download_corpus_images(
        corpus,
        image_root=tmp_path,
        session=FakeSession(
            [
                FakeResponse(
                    content=image_content,
                )
            ]
        ),
        sleep_function=lambda _: None,
    )

    row = manifest.iloc[0]

    assert row["download_status"] == "available"
    assert row["acquisition_action"] == "downloaded"
    assert row["image_width"] == 40
    assert row["image_height"] == 30

    output_path = (
        tmp_path
        / "MET"
        / "MET_001.png"
    )

    assert output_path.exists()


def test_existing_image_is_reused(
    tmp_path: Path,
) -> None:
    output_path = (
        tmp_path
        / "MET"
        / "MET_001.png"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_bytes(
        make_image_bytes()
    )

    corpus = pd.DataFrame(
        [
            {
                "item_id": "MET_001",
                "museum": "MET",
                "image_url": (
                    "https://example.org/image.png"
                ),
            }
        ]
    )

    manifest = download_corpus_images(
        corpus,
        image_root=tmp_path,
        session=FakeSession([]),
        sleep_function=lambda _: None,
    )

    row = manifest.iloc[0]

    assert row["download_status"] == "available"
    assert row["acquisition_action"] == "reused"
    assert row["sha256"]


def test_http_error_is_recorded(
    tmp_path: Path,
) -> None:
    corpus = pd.DataFrame(
        [
            {
                "item_id": "CMA_001",
                "museum": "CMA",
                "image_url": (
                    "https://example.org/blocked.jpg"
                ),
            }
        ]
    )

    manifest = download_corpus_images(
        corpus,
        image_root=tmp_path,
        session=FakeSession(
            [
                FakeResponse(
                    status_code=403,
                    content=b"",
                )
            ]
        ),
        retries=0,
        sleep_function=lambda _: None,
    )

    row = manifest.iloc[0]

    assert row["download_status"] == "failed"
    assert row["failure_code"] == "http_error"
    assert row["http_status"] == 403


def test_summary_counts_download_statuses() -> None:
    manifest = pd.DataFrame(
        [
            {
                "museum": "MET",
                "download_status": "available",
            },
            {
                "museum": "MET",
                "download_status": "failed",
            },
            {
                "museum": "CMA",
                "download_status": "available",
            },
        ]
    )

    summary = build_download_summary(
        manifest
    )

    all_rows = summary[
        summary["museum"].eq("ALL")
    ]

    observed = all_rows.set_index(
        "download_status"
    )["records"].to_dict()

    assert observed == {
        "available": 2,
        "failed": 1,
    }


def test_download_output_files_exist() -> None:
    assert MANIFEST_PATH.exists()
    assert SUMMARY_PATH.exists()


def test_current_manifest_preserves_full_corpus() -> None:
    corpus = pd.read_csv(CORPUS_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)

    assert len(corpus) == 215
    assert len(manifest) == 215

    assert manifest["item_id"].is_unique

    assert manifest["item_id"].tolist() == (
        corpus["item_id"].tolist()
    )


def test_current_manifest_statuses_are_controlled() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)

    allowed_statuses = {
        "available",
        "failed",
        "missing_url",
        "ambiguous_existing",
        "invalid_existing",
    }

    assert set(
        manifest["download_status"]
    ).issubset(allowed_statuses)