from pathlib import Path

import yaml


SCHEMA_PATH = Path("docs/dataset/paper_dataset_schema_v1.yaml")


def load_schema() -> dict:
    with SCHEMA_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_schema_file_exists() -> None:
    assert SCHEMA_PATH.exists()


def test_schema_version_is_defined() -> None:
    schema = load_schema()
    assert schema["schema"]["version"] == "1.0.0"


def test_primary_key_is_item_id() -> None:
    schema = load_schema()
    assert schema["schema"]["primary_key"] == ["item_id"]


def test_field_names_are_unique() -> None:
    schema = load_schema()
    field_names = [field["name"] for field in schema["fields"]]

    assert len(field_names) == len(set(field_names))


def test_required_core_fields_exist() -> None:
    schema = load_schema()
    field_names = {field["name"] for field in schema["fields"]}

    required_fields = {
        "item_id",
        "museum",
        "source_object_id",
        "title",
        "image_url",
        "object_url",
        "dataset_split",
        "review_status",
        "source_branch",
        "source_commit",
        "source_file",
        "adapter_name",
        "adapter_version",
        "processing_timestamp",
        "enrichment_source_file",
        "metadata_recovery_source",
        "metadata_recovery_version",
    }

    assert required_fields.issubset(field_names)


def test_required_fields_are_not_nullable() -> None:
    schema = load_schema()

    for field in schema["fields"]:
        if field.get("required"):
            assert field.get("nullable") is False, field["name"]