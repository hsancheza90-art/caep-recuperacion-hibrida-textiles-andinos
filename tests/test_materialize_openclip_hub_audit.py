"""Pruebas de materialización de la auditoría de hubs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.openclip_baseline.materialize_hub_audit import (
    materialize_openclip_hub_audit,
)


def _write_input_files(
    tmp_path: Path,
) -> dict[str, Path]:
    """Crea archivos sintéticos compatibles con la auditoría."""

    image_dir = tmp_path / "images"
    image_dir.mkdir()

    items = [
        (
            "item-a",
            "CMA",
            "Paracas",
        ),
        (
            "item-b",
            "CMA",
            "Chancay",
        ),
        (
            "item-c",
            "MET",
            "Paracas",
        ),
        (
            "item-d",
            "MET",
            "Wari",
        ),
    ]

    corpus_records = []

    for item_id, museum, culture in items:
        image_path = (
            image_dir / f"{item_id}.jpg"
        )
        image_path.write_bytes(b"image")

        corpus_records.append(
            {
                "item_id": item_id,
                "museum": museum,
                "title": f"Title {item_id}",
                "title_original": f"Original {item_id}",
                "description": "Description",
                "culture": culture,
                "culture_canonical": culture,
                "period": "1000",
                "object_type": "Textile",
                "material": "Cotton",
                "technique": "Weave",
                "classification": "Textile",
                "image_local_path": str(
                    image_path.relative_to(
                        tmp_path
                    )
                ),
                "image_url": (
                    "https://example.org/image"
                ),
                "object_url": (
                    "https://example.org/object"
                ),
                "image_width": "100",
                "image_height": "100",
                "image_bytes": "5",
                "image_sha256": "hash",
            }
        )

    corpus = pd.DataFrame.from_records(
        corpus_records
    )

    text_inputs = pd.DataFrame(
        {
            "item_id": [
                item[0]
                for item in items
            ],
            "museum": [
                item[1]
                for item in items
            ],
            "text_visual": [
                "visual a",
                "visual b",
                "visual c",
                "visual d",
            ],
            "text_visual_version": [
                "v1",
                "v1",
                "v1",
                "v1",
            ],
            "text_metadata": [
                "metadata a",
                "metadata b",
                "metadata c",
                "metadata d",
            ],
            "text_metadata_version": [
                "v1",
                "v1",
                "v1",
                "v1",
            ],
        }
    )

    museum_by_item = {
        item_id: museum
        for item_id, museum, _ in items
    }

    per_query_records = []

    top1_by_configuration = {
        "config_a": [
            "item-a",
            "item-a",
            "item-a",
            "item-d",
        ],
        "config_b": [
            "item-a",
            "item-a",
            "item-c",
            "item-d",
        ],
    }

    for configuration, top1_items in (
        top1_by_configuration.items()
    ):
        for query_index, (
            item_id,
            museum,
            _,
        ) in enumerate(items):
            top1_item_id = top1_items[
                query_index
            ]

            per_query_records.append(
                {
                    "configuration": configuration,
                    "query_index": query_index,
                    "item_id": item_id,
                    "museum": museum,
                    "matched_rank": query_index + 1,
                    "matched_score": 0.3,
                    "top1_item_id": top1_item_id,
                    "top1_museum": museum_by_item[
                        top1_item_id
                    ],
                    "top1_score": 0.4,
                }
            )

    per_query = pd.DataFrame.from_records(
        per_query_records
    )

    persistent_hubs = pd.DataFrame(
        {
            "candidate_item_id": [
                "item-a",
            ],
            "candidate_museum": [
                "CMA",
            ],
            "configurations_as_hub": [
                2,
            ],
            "configurations": [
                "config_a|config_b",
            ],
            "total_top1_count": [
                5,
            ],
            "total_false_top1_count": [
                3,
            ],
            "max_top1_count": [
                3,
            ],
        }
    )

    candidate_counts = pd.DataFrame(
        [
            {
                "configuration": "config_a",
                "candidate_rank": 1,
                "candidate_item_id": "item-a",
                "candidate_museum": "CMA",
                "top1_count": 3,
                "self_match_count": 1,
                "false_top1_count": 2,
                "top1_share": 0.75,
                "is_top1_hub": True,
            },
            {
                "configuration": "config_b",
                "candidate_rank": 1,
                "candidate_item_id": "item-a",
                "candidate_museum": "CMA",
                "top1_count": 2,
                "self_match_count": 1,
                "false_top1_count": 1,
                "top1_share": 0.50,
                "is_top1_hub": True,
            },
        ]
    )

    paths = {
        "corpus": tmp_path / "corpus.csv",
        "text_inputs": (
            tmp_path / "text_inputs.csv"
        ),
        "per_query": tmp_path / "per_query.csv",
        "persistent_hubs": (
            tmp_path / "persistent_hubs.csv"
        ),
        "candidate_counts": (
            tmp_path / "candidate_counts.csv"
        ),
    }

    corpus.to_csv(
        paths["corpus"],
        index=False,
    )
    text_inputs.to_csv(
        paths["text_inputs"],
        index=False,
    )
    per_query.to_csv(
        paths["per_query"],
        index=False,
    )
    persistent_hubs.to_csv(
        paths["persistent_hubs"],
        index=False,
    )
    candidate_counts.to_csv(
        paths["candidate_counts"],
        index=False,
    )

    return paths


def _materialize(
    tmp_path: Path,
) -> dict[str, Path]:
    """Ejecuta la materialización sintética."""

    inputs = _write_input_files(
        tmp_path
    )

    outputs = {
        "profiles": tmp_path / "profiles.csv",
        "counts": tmp_path / "counts.csv",
        "events": tmp_path / "events.csv",
        "summary": tmp_path / "summary.csv",
        "provenance": (
            tmp_path / "provenance.json"
        ),
    }

    materialize_openclip_hub_audit(
        corpus_path=inputs["corpus"],
        text_inputs_path=inputs[
            "text_inputs"
        ],
        per_query_path=inputs["per_query"],
        persistent_hubs_path=inputs[
            "persistent_hubs"
        ],
        candidate_counts_path=inputs[
            "candidate_counts"
        ],
        hub_profiles_path=outputs[
            "profiles"
        ],
        hub_configuration_counts_path=outputs[
            "counts"
        ],
        attraction_events_path=outputs[
            "events"
        ],
        summary_path=outputs["summary"],
        provenance_path=outputs[
            "provenance"
        ],
        repository_root=tmp_path,
    )

    return outputs


def test_materializer_writes_all_outputs(
    tmp_path: Path,
) -> None:
    """La materialización debe producir cinco artefactos."""

    outputs = _materialize(tmp_path)

    assert all(
        path.is_file()
        for path in outputs.values()
    )


def test_materialized_tables_have_expected_rows(
    tmp_path: Path,
) -> None:
    """Los conteos deben conservar perfiles y eventos."""

    outputs = _materialize(tmp_path)

    profiles = pd.read_csv(
        outputs["profiles"]
    )
    counts = pd.read_csv(
        outputs["counts"]
    )
    events = pd.read_csv(
        outputs["events"]
    )
    summary = pd.read_csv(
        outputs["summary"]
    )

    assert len(profiles) == 1
    assert len(counts) == 2
    assert len(events) == 5
    assert len(summary) == 2

    assert bool(
        profiles.iloc[0]["image_exists"]
    ) is True


def test_provenance_records_audit_coverage(
    tmp_path: Path,
) -> None:
    """La procedencia debe registrar la cobertura completa."""

    outputs = _materialize(tmp_path)

    provenance = json.loads(
        outputs["provenance"].read_text(
            encoding="utf-8"
        )
    )

    coverage = provenance["coverage"]

    assert coverage["corpus_records"] == 4
    assert coverage["persistent_hub_records"] == 1
    assert coverage["hub_profile_rows"] == 1
    assert coverage["attraction_event_rows"] == 5
    assert coverage["available_hub_images"] == 1


def test_materializer_rejects_missing_input(
    tmp_path: Path,
) -> None:
    """Todos los archivos fuente deben existir."""

    with pytest.raises(
        FileNotFoundError,
        match="No existe",
    ):
        materialize_openclip_hub_audit(
            corpus_path=(
                tmp_path / "missing.csv"
            ),
            text_inputs_path=(
                tmp_path / "texts.csv"
            ),
            per_query_path=(
                tmp_path / "per_query.csv"
            ),
            persistent_hubs_path=(
                tmp_path / "hubs.csv"
            ),
            candidate_counts_path=(
                tmp_path / "counts.csv"
            ),
            hub_profiles_path=(
                tmp_path / "profiles.csv"
            ),
            hub_configuration_counts_path=(
                tmp_path / "config_counts.csv"
            ),
            attraction_events_path=(
                tmp_path / "events.csv"
            ),
            summary_path=(
                tmp_path / "summary.csv"
            ),
            provenance_path=(
                tmp_path / "provenance.json"
            ),
            repository_root=tmp_path,
        )