# Gate G2.2 — Esquema armonizado

## Estado

APROBADO

## Fuentes previstas

- MET: 127 registros principales revisados.
- CMA: 88 registros principales revisados.
- Total esperado: 215 registros.

## Artefactos

- `docs/dataset/paper_dataset_schema_v1.yaml`
- `docs/dataset/paper_dataset_schema_v1.md`
- `config/controlled_vocabularies.yaml`
- `tests/test_dataset_schema.py`

## Criterios de aprobación

- [x] El esquema YAML es válido.
- [x] La clave primaria es `item_id`.
- [x] No existen nombres de campos duplicados.
- [x] Los campos esenciales están presentes.
- [x] Los campos obligatorios no aceptan nulos.
- [x] Los mapeos MET y CMA están documentados.
- [x] Los vocabularios controlados están definidos.
- [x] Las pruebas automatizadas terminan correctamente.

## Decisiones de diseño

### Identificador armonizado

```text
MET:<id_fuente>
CMA:<id_objeto>
```
## Título principal MET

Se prioriza titulo_es_sugerido. Cuando este campo está vacío, se utiliza
titulo_original. El valor de titulo_original siempre se conserva.

## Título principal CMA

Se utiliza titulo_original tanto como título principal como valor original.

## Evidencia

6 pruebas aprobadas
Python 3.12.10
pytest 9.1.1
Decisión

Se autoriza la implementación de los adaptadores MET y CMA.


# Siguiente hito: adaptadores de fuente