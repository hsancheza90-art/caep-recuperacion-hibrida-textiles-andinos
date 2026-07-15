# Esquema del corpus experimental v1

## 1. Propósito

Este documento define el contrato de datos para el corpus experimental usado
en el artículo sobre recuperación híbrida e interpretable de textiles andinos.

El esquema armoniza registros procedentes del Metropolitan Museum of Art
(MET) y del Cleveland Museum of Art (CMA), preservando la trazabilidad hacia
las fuentes originales.

## 2. Principios

1. No se eliminan ni sobrescriben los archivos curatoriales originales.
2. Cada museo es transformado mediante un adaptador independiente.
3. El dataset armonizado usa un esquema común.
4. Cada registro conserva rama, commit y archivo fuente.
5. El identificador del corpus debe ser único, legible y trazable.
6. Los campos normalizados no sustituyen los valores originales.
7. La primera versión experimental incluye únicamente el corpus principal.

## 3. Identificador principal

La clave primaria es:

```text
item_id = <museum>:<source_object_id>