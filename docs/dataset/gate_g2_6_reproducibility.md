# Gate G2.6 — Reproducibilidad del corpus enriquecido

## Estado

APROBADO

## Corpus congelado

- Nombre: `paper_corpus_enriched_v1`
- Versión: `1.0.0`
- Registros: 215
- Columnas: 42
- MET: 127
- CMA: 88
- Timestamp de construcción: `2026-07-15T09:44:36Z`

## Artefactos

- `data/processed/paper_corpus_enriched_v1.csv`
- `data/processed/paper_corpus_enriched_v1.parquet`

## Integridad

### CSV canónico

```text
C19556C2DD424B51FF11E0FD0FDD9737659E013A378AB3BD54192BF48127E006
```

### Parquet del entorno experimental

```text
4BF20B745414FDE39039A6B4D5F0BD3018DD92CC84E92643AB678C4F944106A6
```

El CSV se adopta como artefacto canónico portable. El hash Parquet se
documenta para el entorno actual y puede depender de la versión del motor de
serialización.

## Criterios de aprobación
 El corpus se reconstruye sin archivos derivados de data/interim.
 Se obtienen 215 registros.
 Se obtienen 42 columnas.
 La distribución es MET=127 y CMA=88.
 Dos ejecuciones producen el mismo CSV.
 Dos ejecuciones producen el mismo Parquet.
 El timestamp procede de una configuración versionada.
 La suite completa termina con 21 pruebas aprobadas.

### Decisión

El corpus queda congelado como versión experimental v1.0.0, condicionado
únicamente a la confirmación final de la suite automatizada.