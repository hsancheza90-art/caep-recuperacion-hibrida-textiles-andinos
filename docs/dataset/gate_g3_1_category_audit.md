# Gate G3.1 — Auditoría de categorías

## Estado

APROBADO CON HALLAZGO CRÍTICO

## Resultados

| Campo | Cobertura | Categorías únicas | Candidatas globales |
|---|---:|---:|---:|
| `culture` | 100 % | 57 | 0 |
| `period` | 100 % | 80 | 0 |
| `object_type` | 100 % | 27 | 0 |
| `classification` | 100 % | 5 | 0 |
| `material` | 59.1 % | 12 | 0 |
| `technique` | 54.4 % | 68 | 0 |

## Interpretación

Las etiquetas curatoriales originales no constituyen un vocabulario global
directamente comparable entre MET y CMA.

La ausencia de categorías candidatas no se interpreta como un fallo del
corpus, sino como evidencia de heterogeneidad terminológica, diferencia de
granularidad y dependencia institucional.

## Decisiones

- No se reducen todavía los umbrales de soporte.
- No se utilizan etiquetas crudas como ground truth global.
- Se construirá un vocabulario controlado y versionado.
- La armonización comenzará por `classification` y `object_type`.
- `material` y `technique` permanecen como variables diagnósticas.
- No se aplicarán equivalencias semánticas automáticas sin revisión.

## Decisión final

Se autoriza el Hito G3.2: armonización controlada de categorías.