from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_TAXONOMY_PATH = (
    PROJECT_ROOT / "config" / "culture_taxonomy_v1.yaml"
)


def normalize_ascii(value: object) -> str:
    if value is None:
        return ""

    text = unicodedata.normalize("NFKD", str(value))

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    text = text.casefold()
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_culture_taxonomy(
    path: Path = DEFAULT_TAXONOMY_PATH,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe la taxonomía cultural: {path}"
        )

    with path.open("r", encoding="utf-8-sig") as file:
        taxonomy = yaml.safe_load(file)

    if not isinstance(taxonomy, dict):
        raise ValueError(
            "La taxonomía cultural no es válida."
        )

    if "cultures" not in taxonomy:
        raise ValueError(
            "La taxonomía debe contener 'cultures'."
        )

    return taxonomy


def contains_alias(
    normalized_label: str,
    alias: str,
) -> bool:
    normalized_alias = normalize_ascii(alias)

    pattern = (
        r"(?<![a-z0-9])"
        + re.escape(normalized_alias)
        + r"(?![a-z0-9])"
    )

    return bool(
        re.search(pattern, normalized_label)
    )


def detect_culture_components(
    source_label: str,
    taxonomy: dict,
) -> list[str]:
    normalized = normalize_ascii(source_label)
    components: list[str] = []

    for culture in taxonomy["cultures"]:
        canonical_label = str(
            culture["canonical_label"]
        )

        aliases = culture.get("aliases", [])

        if any(
            contains_alias(normalized, alias)
            for alias in aliases
        ):
            components.append(canonical_label)

    return components


def build_signal_pattern(signal: str) -> str:
    """
    Construye una expresión regular segura para detectar señales.

    Las señales formadas por palabras se buscan con límites léxicos.
    Las señales de puntuación, como '?' o '(?)', se buscan de forma
    literal.
    """
    normalized_signal = normalize_ascii(signal)

    if not normalized_signal:
        raise ValueError(
            "La señal normalizada no puede estar vacía."
        )

    escaped = re.escape(normalized_signal).replace(
        r"\ ",
        r"\s+",
    )

    if re.fullmatch(
        r"[a-z0-9]+(?: [a-z0-9]+)*",
        normalized_signal,
    ):
        return (
            rf"(?<![a-z0-9])"
            rf"{escaped}"
            rf"(?![a-z0-9])"
        )

    return escaped


def contains_signal(
    source_label: str,
    signals: list[str],
) -> bool:
    """
    Detecta señales como palabras o puntuación completas.

    Evita falsos positivos como:
    - 'or' dentro de 'Horizon';
    - 'or' dentro de 'north'.
    """
    normalized_label = normalize_ascii(source_label)

    return any(
        re.search(
            build_signal_pattern(signal),
            normalized_label,
        )
        is not None
        for signal in signals
    )


def classify_culture_proposal(
    source_label: str,
    components: list[str],
    taxonomy: dict,
) -> dict[str, object]:
    signals = taxonomy.get("signals", {})

    uncertain = contains_signal(
        source_label,
        signals.get("uncertainty", []),
    )

    style_attribution = contains_signal(
        source_label,
        signals.get("style", []),
    )

    component_count = len(components)
    composite = component_count > 1
    unattributed = component_count == 0

    andean_components = [
        component
        for component in components
        if component != "Spanish"
    ]

    if unattributed:
        decision = "excluir_no_atribuida"
        strict_eligible = False
        rationale = (
            "No contiene una cultura del vocabulario controlado."
        )

    elif uncertain:
        decision = "revisar_incierta"
        strict_eligible = False
        rationale = (
            "La fuente expresa incertidumbre o alternativas."
        )

    elif composite:
        decision = "revisar_compuesta"
        strict_eligible = False
        rationale = (
            "La etiqueta contiene más de un componente cultural."
        )

    elif style_attribution:
        decision = "revisar_estilo"
        strict_eligible = False
        rationale = (
            "La atribución se expresa como estilo y no como "
            "adscripción cultural directa."
        )

    elif len(andean_components) != 1:
        decision = "excluir_no_andina_simple"
        strict_eligible = False
        rationale = (
            "No contiene exactamente una cultura andina simple."
        )

    else:
        decision = "propuesta_aceptable"
        strict_eligible = True
        rationale = (
            "Contiene una atribución cultural explícita, simple "
            "y sin señales de incertidumbre."
        )

    return {
        "canonical_components": " | ".join(components),
        "component_count": component_count,
        "is_composite": composite,
        "is_uncertain": uncertain,
        "is_style_attribution": style_attribution,
        "is_unattributed": unattributed,
        "strict_eligible": strict_eligible,
        "proposal_decision": decision,
        "proposal_rationale": rationale,
    }


def build_culture_proposals(
    inventory: pd.DataFrame,
    taxonomy: dict,
) -> pd.DataFrame:
    required_columns = {
        "field",
        "museum",
        "source_key",
        "source_label",
        "record_count",
        "example_item_ids",
        "example_titles",
    }

    missing = required_columns.difference(
        inventory.columns
    )

    if missing:
        raise ValueError(
            f"Faltan columnas: {sorted(missing)}"
        )

    culture_inventory = inventory.loc[
        inventory["field"].eq("culture")
    ].copy()

    rows: list[dict[str, object]] = []

    for row in culture_inventory.itertuples(
        index=False
    ):
        components = detect_culture_components(
            row.source_label,
            taxonomy,
        )

        classification = classify_culture_proposal(
            row.source_label,
            components,
            taxonomy,
        )

        rows.append(
            {
                "field": "culture",
                "museum": row.museum,
                "source_key": row.source_key,
                "source_label": row.source_label,
                "record_count": int(row.record_count),
                "example_item_ids": row.example_item_ids,
                "example_titles": row.example_titles,
                **classification,
                "review_decision": "pendiente",
                "review_rationale": "",
                "reviewer": "",
                "review_date": "",
            }
        )

    proposals = pd.DataFrame(rows)

    return (
        proposals
        .sort_values(
            [
                "proposal_decision",
                "record_count",
                "museum",
                "source_label",
            ],
            ascending=[True, False, True, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )