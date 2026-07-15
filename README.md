# CAEP — Recuperación interpretable de textiles andinos

Proyecto experimental para el artículo:

**Recuperación interpretable de textiles andinos mediante una representación
híbrida basada en modelos visión-lenguaje y descriptores estructurales
combinatorios.**

## Objetivo

Desarrollar y evaluar un método híbrido de recuperación que combine:

1. embeddings visión-lenguaje obtenidos mediante OpenCLIP;
2. un descriptor estructural basado en atributos observables;
3. una función de similitud híbrida e interpretable.

## Fuentes

Los corpus curatoriales de origen proceden del repositorio:

`hsancheza90-art/uni-cc-base-multimodal-textiles-andinos`

Ramas utilizadas:

- `curacion/corpus-met-textiles-andinos-v1`
- `curacion/corpus-cma-textiles-andinos-v1`

Los archivos fuente serán importados con trazabilidad de rama y commit.
Las imágenes no se conservaron en los corpus originales y serán descargadas
desde enlaces institucionales mediante un proceso reproducible.

## Estructura

- `config/`: configuración y registro de fuentes.
- `data/source/`: copias inmutables de los insumos curatoriales.
- `data/interim/`: datos armonizados.
- `data/processed/`: corpus experimental congelado.
- `data/images/`: imágenes descargadas localmente.
- `src/caep/`: código fuente.
- `tests/`: pruebas automatizadas.
- `outputs/`: reportes y figuras.
- `paper/`: manuscrito del artículo.