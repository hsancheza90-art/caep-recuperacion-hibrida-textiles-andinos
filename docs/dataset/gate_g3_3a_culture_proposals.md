# Gate G3.3-A — Propuestas de armonización cultural

## Estado

PENDIENTE DE VALIDACIÓN FINAL

## Correcciones aplicadas

1. La señal `or` se detecta como palabra independiente y no como subcadena.
2. Se agregaron pruebas para evitar coincidencias falsas en `Horizon` y
   `north`.
3. Se incorporaron `Lambayeque/Sicán` y `Chavín` al vocabulario controlado.
4. Las atribuciones de estilo permanecen fuera del subconjunto estricto.
5. Las atribuciones compuestas e inciertas conservan todos sus componentes,
   pero requieren revisión.

## Criterios de aprobación

- [x] `or` no coincide dentro de `Horizon`.
- [x] `or` no coincide dentro de `north`.
- [x] La palabra independiente `or` indica incertidumbre.
- [x] El signo `?` indica incertidumbre.
- [ ] Lambayeque/Sicán se reconoce como atribución simple.
- [ ] Chavín style se reconoce como atribución de estilo.
- [ ] Las 45 pruebas terminan correctamente.
- [ ] Las propuestas fueron regeneradas.

## Decisión

Pendiente.