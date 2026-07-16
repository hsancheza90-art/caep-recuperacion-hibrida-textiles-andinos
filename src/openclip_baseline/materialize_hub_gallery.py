"""Galería HTML reproducible para auditar hubs OpenCLIP."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from html import escape
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd


DATASET_NAME = "openclip_hub_gallery_v1"
PIPELINE_VERSION = "openclip_hub_gallery_v1"


_PROFILE_REQUIRED_COLUMNS = {
    "candidate_item_id",
    "candidate_museum",
    "title",
    "culture_canonical",
    "object_type",
    "material",
    "image_local_path",
    "configurations_as_hub",
    "total_top1_count",
    "total_false_top1_count",
    "text_visual",
    "text_metadata",
}

_EVENT_REQUIRED_COLUMNS = {
    "configuration",
    "query_index",
    "query_item_id",
    "query_museum",
    "matched_rank",
    "matched_score",
    "top1_score",
    "candidate_item_id",
    "candidate_museum",
    "self_match",
    "cross_museum",
    "same_culture_canonical",
    "same_object_type",
    "query_title",
    "query_culture_canonical",
    "query_object_type",
    "query_image_local_path",
}


@dataclass(frozen=True)
class OpenCLIPHubGalleryConfig:
    """Configuración de la galería visual de hubs."""

    configurations: tuple[str, ...] = (
        "text_visual",
        "text_metadata",
        "text_fused_alpha_0.50",
    )
    max_hubs: int = 16
    max_queries_per_configuration: int = 6
    include_self_matches: bool = False
    title: str = "Auditoría visual de hubs OpenCLIP"

    def __post_init__(self) -> None:
        """Valida la configuración."""

        if not self.configurations:
            raise ValueError(
                "configurations no puede estar vacío."
            )

        if len(set(self.configurations)) != len(
            self.configurations
        ):
            raise ValueError(
                "Las configuraciones deben ser únicas."
            )

        if self.max_hubs <= 0:
            raise ValueError(
                "max_hubs debe ser mayor que cero."
            )

        if self.max_queries_per_configuration <= 0:
            raise ValueError(
                "max_queries_per_configuration debe ser "
                "mayor que cero."
            )

        if not self.title.strip():
            raise ValueError(
                "title no puede estar vacío."
            )


def _require_columns(
    frame: pd.DataFrame,
    *,
    required: set[str],
    frame_name: str,
) -> None:
    """Valida las columnas requeridas."""

    missing = required - set(frame.columns)

    if missing:
        missing_text = ", ".join(
            sorted(missing)
        )

        raise ValueError(
            f"{frame_name} no contiene las columnas "
            f"requeridas: {missing_text}"
        )


def _as_bool(
    value: object,
) -> bool:
    """Convierte valores CSV comunes a booleano."""

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        return bool(value)

    normalized = str(value).strip().lower()

    return normalized in {
        "true",
        "1",
        "yes",
        "si",
        "sí",
    }


def _display_text(
    value: object,
    *,
    default: str = "—",
) -> str:
    """Escapa texto para mostrarlo dentro del HTML."""

    text = str(value).strip()

    if not text:
        text = default

    return escape(
        text,
        quote=True,
    )


def _truncate(
    value: object,
    *,
    max_length: int,
) -> str:
    """Recorta texto largo antes de escapar."""

    text = str(value).strip()

    if len(text) > max_length:
        text = (
            text[: max_length - 1].rstrip()
            + "…"
        )

    return _display_text(text)


def _resolve_local_path(
    value: object,
    *,
    repository_root: Path,
) -> Path:
    """Resuelve una ruta local respecto del repositorio."""

    path = Path(str(value))

    if not path.is_absolute():
        path = repository_root / path

    return path.resolve()


def _image_source(
    value: object,
    *,
    repository_root: Path,
    output_html_path: Path,
) -> str:
    """Construye una ruta relativa desde la galería."""

    image_path = _resolve_local_path(
        value,
        repository_root=repository_root,
    )

    if not image_path.is_file():
        return ""

    relative = os.path.relpath(
        image_path,
        start=output_html_path.parent.resolve(),
    )

    return Path(relative).as_posix()


def _render_image(
    value: object,
    *,
    alt: str,
    repository_root: Path,
    output_html_path: Path,
    css_class: str,
) -> str:
    """Renderiza una imagen local o un marcador vacío."""

    source = _image_source(
        value,
        repository_root=repository_root,
        output_html_path=output_html_path,
    )

    if not source:
        return (
            f'<div class="{css_class} image-missing">'
            "Imagen no disponible"
            "</div>"
        )

    return (
        f'<img class="{css_class}" '
        f'src="{escape(source, quote=True)}" '
        f'alt="{escape(alt, quote=True)}" '
        'loading="lazy">'
    )


def _badge(
    label: str,
    value: bool,
) -> str:
    """Construye una insignia booleana."""

    css_class = (
        "badge positive"
        if value
        else "badge negative"
    )

    status = "sí" if value else "no"

    return (
        f'<span class="{css_class}">'
        f"{escape(label)}: {status}"
        "</span>"
    )


def _render_query_card(
    event: pd.Series,
    *,
    repository_root: Path,
    output_html_path: Path,
) -> str:
    """Renderiza una consulta atraída por un hub."""

    image = _render_image(
        event["query_image_local_path"],
        alt=str(event["query_item_id"]),
        repository_root=repository_root,
        output_html_path=output_html_path,
        css_class="query-image",
    )

    badges = "".join(
        [
            _badge(
                "mismo museo",
                not _as_bool(
                    event["cross_museum"]
                ),
            ),
            _badge(
                "misma cultura",
                _as_bool(
                    event[
                        "same_culture_canonical"
                    ]
                ),
            ),
            _badge(
                "mismo tipo",
                _as_bool(
                    event["same_object_type"]
                ),
            ),
        ]
    )

    return f"""
    <article class="query-card">
        {image}
        <div class="query-content">
            <h5>
                {_display_text(event["query_title"])}
            </h5>
            <p class="identifier">
                {_display_text(event["query_item_id"])}
                · {_display_text(event["query_museum"])}
            </p>
            <p>
                <strong>Cultura:</strong>
                {_display_text(
                    event["query_culture_canonical"]
                )}
            </p>
            <p>
                <strong>Tipo:</strong>
                {_display_text(
                    event["query_object_type"]
                )}
            </p>
            <p>
                <strong>Rango de su imagen correcta:</strong>
                {_display_text(event["matched_rank"])}
            </p>
            <p>
                <strong>Score del hub:</strong>
                {float(event["top1_score"]):.6f}
            </p>
            <p>
                <strong>Score del par correcto:</strong>
                {float(event["matched_score"]):.6f}
            </p>
            <div class="badges">
                {badges}
            </div>
        </div>
    </article>
    """


def _render_configuration_section(
    events: pd.DataFrame,
    *,
    configuration: str,
    repository_root: Path,
    output_html_path: Path,
    config: OpenCLIPHubGalleryConfig,
) -> str:
    """Renderiza las consultas atraídas en una configuración."""

    selected = events.loc[
        events["configuration"].astype(str)
        == configuration
    ].copy()

    if not config.include_self_matches:
        selected = selected.loc[
            ~selected["self_match"].map(
                _as_bool
            )
        ]

    selected = selected.sort_values(
        [
            "top1_score",
            "matched_rank",
            "query_index",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        kind="stable",
    ).head(
        config.max_queries_per_configuration
    )

    if selected.empty:
        query_cards = (
            '<p class="empty-message">'
            "No hay consultas disponibles."
            "</p>"
        )
    else:
        query_cards = "".join(
            _render_query_card(
                event,
                repository_root=repository_root,
                output_html_path=output_html_path,
            )
            for _, event in selected.iterrows()
        )

    total_events = len(
        events.loc[
            events["configuration"].astype(str)
            == configuration
        ]
    )

    return f"""
    <section class="configuration-section">
        <h4>{escape(configuration)}</h4>
        <p class="configuration-count">
            Eventos atraídos: {total_events}
        </p>
        <div class="query-grid">
            {query_cards}
        </div>
    </section>
    """


def _render_hub_card(
    profile: pd.Series,
    attraction_events: pd.DataFrame,
    *,
    repository_root: Path,
    output_html_path: Path,
    config: OpenCLIPHubGalleryConfig,
) -> str:
    """Renderiza un hub y sus consultas atraídas."""

    item_id = str(
        profile["candidate_item_id"]
    )

    events = attraction_events.loc[
        attraction_events[
            "candidate_item_id"
        ].astype(str)
        == item_id
    ].copy()

    hub_image = _render_image(
        profile["image_local_path"],
        alt=item_id,
        repository_root=repository_root,
        output_html_path=output_html_path,
        css_class="hub-image",
    )

    configuration_sections = "".join(
        _render_configuration_section(
            events,
            configuration=configuration,
            repository_root=repository_root,
            output_html_path=output_html_path,
            config=config,
        )
        for configuration in config.configurations
    )

    return f"""
    <article class="hub-card" id="{escape(item_id)}">
        <header class="hub-header">
            <div>
                <h2>{_display_text(profile["title"])}</h2>
                <p class="identifier">
                    {_display_text(item_id)}
                    · {_display_text(
                        profile["candidate_museum"]
                    )}
                </p>
            </div>
            <div class="hub-metrics">
                <span>
                    Configuraciones:
                    {_display_text(
                        profile[
                            "configurations_as_hub"
                        ]
                    )}
                </span>
                <span>
                    Top-1 totales:
                    {_display_text(
                        profile["total_top1_count"]
                    )}
                </span>
                <span>
                    Falsos Top-1:
                    {_display_text(
                        profile[
                            "total_false_top1_count"
                        ]
                    )}
                </span>
            </div>
        </header>

        <div class="hub-overview">
            <div class="hub-image-panel">
                {hub_image}
            </div>

            <div class="hub-metadata">
                <p>
                    <strong>Cultura:</strong>
                    {_display_text(
                        profile["culture_canonical"]
                    )}
                </p>
                <p>
                    <strong>Tipo:</strong>
                    {_display_text(
                        profile["object_type"]
                    )}
                </p>
                <p>
                    <strong>Material:</strong>
                    {_display_text(
                        profile["material"]
                    )}
                </p>

                <details>
                    <summary>Texto visual</summary>
                    <p>
                        {_truncate(
                            profile["text_visual"],
                            max_length=1200,
                        )}
                    </p>
                </details>

                <details>
                    <summary>Texto de metadatos</summary>
                    <p>
                        {_truncate(
                            profile["text_metadata"],
                            max_length=1600,
                        )}
                    </p>
                </details>
            </div>
        </div>

        <div class="configuration-container">
            {configuration_sections}
        </div>
    </article>
    """


def build_openclip_hub_gallery_html(
    *,
    hub_profiles: pd.DataFrame,
    attraction_events: pd.DataFrame,
    repository_root: Path,
    output_html_path: Path,
    config: OpenCLIPHubGalleryConfig,
) -> str:
    """Construye el documento HTML de auditoría visual."""

    _require_columns(
        hub_profiles,
        required=_PROFILE_REQUIRED_COLUMNS,
        frame_name="hub_profiles",
    )
    _require_columns(
        attraction_events,
        required=_EVENT_REQUIRED_COLUMNS,
        frame_name="attraction_events",
    )

    profiles = hub_profiles.sort_values(
        [
            "total_top1_count",
            "total_false_top1_count",
            "candidate_item_id",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        kind="stable",
    ).head(
        config.max_hubs
    )

    cards = "".join(
        _render_hub_card(
            profile,
            attraction_events,
            repository_root=repository_root,
            output_html_path=output_html_path,
            config=config,
        )
        for _, profile in profiles.iterrows()
    )

    return f"""<!doctype html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <meta
        name="viewport"
        content="width=device-width, initial-scale=1"
    >
    <title>{escape(config.title)}</title>
    <style>
        :root {{
            font-family:
                Inter,
                Segoe UI,
                Arial,
                sans-serif;
            color: #202124;
            background: #f5f6f8;
        }}

        body {{
            margin: 0;
            padding: 0;
        }}

        .page-header {{
            padding: 2rem;
            background: #ffffff;
            border-bottom: 1px solid #dfe3e8;
        }}

        .page-header h1 {{
            margin: 0 0 0.5rem;
        }}

        .page-header p {{
            margin: 0;
            color: #5f6368;
        }}

        main {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 1.5rem;
        }}

        .hub-card {{
            margin-bottom: 2rem;
            padding: 1.25rem;
            background: #ffffff;
            border: 1px solid #dfe3e8;
            border-radius: 14px;
            box-shadow:
                0 4px 18px
                rgba(0, 0, 0, 0.06);
        }}

        .hub-header {{
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: flex-start;
            margin-bottom: 1rem;
        }}

        .hub-header h2 {{
            margin: 0;
        }}

        .identifier {{
            color: #5f6368;
            font-family:
                Consolas,
                monospace;
        }}

        .hub-metrics {{
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-end;
            gap: 0.5rem;
        }}

        .hub-metrics span {{
            padding: 0.4rem 0.65rem;
            background: #eef2f7;
            border-radius: 999px;
            font-size: 0.88rem;
        }}

        .hub-overview {{
            display: grid;
            grid-template-columns:
                minmax(260px, 420px)
                1fr;
            gap: 1.25rem;
        }}

        .hub-image,
        .image-missing {{
            width: 100%;
            height: 420px;
            object-fit: contain;
            background: #eceff3;
            border-radius: 10px;
        }}

        .image-missing {{
            display: flex;
            justify-content: center;
            align-items: center;
            color: #777;
        }}

        details {{
            margin-top: 0.75rem;
            padding: 0.75rem;
            background: #f7f8fa;
            border-radius: 8px;
        }}

        details p {{
            white-space: pre-wrap;
            line-height: 1.45;
        }}

        .configuration-container {{
            margin-top: 1.5rem;
        }}

        .configuration-section {{
            margin-top: 1.5rem;
            padding-top: 1rem;
            border-top: 1px solid #e2e6ea;
        }}

        .configuration-section h4 {{
            margin: 0;
        }}

        .configuration-count {{
            color: #5f6368;
        }}

        .query-grid {{
            display: grid;
            grid-template-columns:
                repeat(
                    auto-fit,
                    minmax(260px, 1fr)
                );
            gap: 1rem;
        }}

        .query-card {{
            overflow: hidden;
            border: 1px solid #dfe3e8;
            border-radius: 10px;
            background: #ffffff;
        }}

        .query-image {{
            width: 100%;
            height: 230px;
            object-fit: contain;
            background: #eef0f3;
        }}

        .query-content {{
            padding: 0.85rem;
        }}

        .query-content h5 {{
            margin: 0 0 0.4rem;
            font-size: 1rem;
        }}

        .query-content p {{
            margin: 0.35rem 0;
            font-size: 0.9rem;
        }}

        .badges {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin-top: 0.75rem;
        }}

        .badge {{
            padding: 0.25rem 0.45rem;
            border-radius: 999px;
            font-size: 0.75rem;
        }}

        .positive {{
            background: #e1f3e8;
            color: #176b3a;
        }}

        .negative {{
            background: #fde8e7;
            color: #9c2f2b;
        }}

        .empty-message {{
            color: #777;
            font-style: italic;
        }}

        footer {{
            padding: 1.5rem;
            text-align: center;
            color: #777;
        }}

        @media (max-width: 850px) {{
            .hub-header {{
                flex-direction: column;
            }}

            .hub-metrics {{
                justify-content: flex-start;
            }}

            .hub-overview {{
                grid-template-columns: 1fr;
            }}

            .hub-image,
            .image-missing {{
                height: 330px;
            }}
        }}
    </style>
</head>
<body>
    <header class="page-header">
        <h1>{escape(config.title)}</h1>
        <p>
            Hubs mostrados: {len(profiles)}.
            Las consultas listadas corresponden a
            candidatos recuperados en la posición Top-1.
        </p>
    </header>

    <main>
        {cards}
    </main>

    <footer>
        Pipeline: {PIPELINE_VERSION}
    </footer>
</body>
</html>
"""


def _sha256_file(
    path: Path,
) -> str:
    """Calcula el hash SHA-256 de un archivo."""

    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _relative_path(
    path: Path,
    repository_root: Path,
) -> str:
    """Representa una ruta relativa cuando es posible."""

    resolved = path.resolve()
    root = repository_root.resolve()

    try:
        return resolved.relative_to(
            root
        ).as_posix()
    except ValueError:
        return resolved.as_posix()


def _file_record(
    path: Path,
    repository_root: Path,
) -> dict[str, object]:
    """Construye un registro de trazabilidad."""

    return {
        "path": _relative_path(
            path,
            repository_root,
        ),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _atomic_write_text(
    destination: Path,
    content: str,
) -> None:
    """Escribe texto mediante reemplazo atómico."""

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_name(
        f".{destination.name}.{uuid4().hex}.tmp"
    )

    try:
        temporary.write_text(
            content,
            encoding="utf-8",
            newline="\n",
        )

        os.replace(
            temporary,
            destination,
        )
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(
    payload: dict[str, Any],
    destination: Path,
) -> None:
    """Escribe JSON estable."""

    content = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )

    _atomic_write_text(
        destination,
        f"{content}\n",
    )


def materialize_openclip_hub_gallery(
    *,
    hub_profiles_path: Path,
    attraction_events_path: Path,
    output_html_path: Path,
    provenance_path: Path,
    repository_root: Path,
    config: OpenCLIPHubGalleryConfig,
) -> None:
    """Construye y materializa la galería visual."""

    for path in (
        hub_profiles_path,
        attraction_events_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"No existe el archivo de entrada: {path}"
            )

    hub_profiles = pd.read_csv(
        hub_profiles_path,
        dtype=str,
        keep_default_na=False,
    )

    attraction_events = pd.read_csv(
        attraction_events_path,
        keep_default_na=False,
    )

    html = build_openclip_hub_gallery_html(
        hub_profiles=hub_profiles,
        attraction_events=attraction_events,
        repository_root=repository_root,
        output_html_path=output_html_path,
        config=config,
    )

    _atomic_write_text(
        output_html_path,
        html,
    )

    selected_hubs = min(
        config.max_hubs,
        len(hub_profiles),
    )

    provenance = {
        "dataset_name": DATASET_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "configuration": {
            "configurations": list(
                config.configurations
            ),
            "max_hubs": config.max_hubs,
            "max_queries_per_configuration": (
                config.max_queries_per_configuration
            ),
            "include_self_matches": (
                config.include_self_matches
            ),
            "title": config.title,
        },
        "coverage": {
            "available_hub_profiles": len(
                hub_profiles
            ),
            "selected_hubs": selected_hubs,
            "attraction_event_rows": len(
                attraction_events
            ),
        },
        "inputs": {
            "hub_profiles": _file_record(
                hub_profiles_path,
                repository_root,
            ),
            "attraction_events": _file_record(
                attraction_events_path,
                repository_root,
            ),
        },
        "output": {
            "html": _file_record(
                output_html_path,
                repository_root,
            )
        },
    }

    _atomic_write_json(
        provenance,
        provenance_path,
    )