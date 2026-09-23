# Notas de calidad — limpieza de extracción (Tarea 1, Fase 1)

Complementa `extraction_quality.csv`. Generado a partir de la corrida real de
`build_index.py --stage clean` sobre `ley_32069` y `ds_001_2026_ef`.

## Limpieza aplicada

- **Header repetido de El Peruano** (ej. `"El Peruano / Lunes 24 de junio de 2024 NORMAS
  LEGALES 3"` o su variante `"3 NORMAS LEGALES Lunes 24 de junio de 2024 / El Peruano"`):
  removido en 35/36 páginas de la Ley (la única sin header es la portada, ver abajo) y en
  16/16 páginas del DS.
- **Sello de firma digital** (`"Firmado por: Editora Peru" + "Fecha: dd/mm/yyyy hh:mm"`):
  removido 1 vez por documento (aparece solo en la primera página de cada PDF descargado).
- **Códigos de publicación (OP)** tipo `2300373-1`: se remueven como artefacto porque no son
  texto normativo. Se detectaron 1 en Ley (footer de cierre del documento completo) y 2 en
  el DS (ver limitación de "bleed entre artículos" abajo).
- **Página de portada**: la página 1 de `ley_32069` es la carátula de la "separata especial"
  (título + gráfico, sin contenido normativo real, detectada por la marca "SEPARATA
  ESPECIAL"). Se excluye de Fase 2 (chunking) vía el flag `is_cover_page`, pero se conserva
  en `data/processed/clean/ley_32069.jsonl` para no perder trazabilidad.

## Limitaciones conocidas (aceptadas, no bloqueantes)

1. **Espaciado irregular alrededor de ligaduras tipográficas.** PyMuPDF inserta un espacio
   espurio junto a los glifos de ligadura (`ﬁ`, `ﬂ`, ...) por un problema de métricas de
   fuente. Se detectaron 411 ocurrencias en `ley_32069` (0 en el DS, que no usa esa fuente/
   maquetación). Se corrigió el espacio posterior a la ligadura (ej. `"fi n"` -> `"fin"`,
   `"Efi cientes"` -> `"Eficientes"`), que cubre la mayoría de los casos. El espacio
   *anterior* a la ligadura no es consistente (a veces es un separador real entre palabras,
   a veces no) y se deja sin tocar: en el peor caso, una palabra como "científicos" puede
   quedar partida en dos tokens (`"cientí ficos"`). Impacto estimado: <0.15% de los
   caracteres del documento: no se espera que degrade materialmente Recall@k, pero se deja
   documentado para la Fase 4 (evaluación).
2. **Posible "bleed" de contenido entre artículos vecinos en la misma página física.** El
   DS N.° 001-2026-EF se publicó dentro de la edición diaria "Normas Legales" del
   08/01/2026 junto a otras normas no relacionadas. La última página del PDF descargado
   (impresa como página 48 de esa edición) contiene, después del cierre y firma del DS
   (`"Dado en la Casa de Gobierno..."`), el inicio de un decreto distinto (creación del
   PRONIED, sector Educación) con su propio código OP (`2474593-1`). Ese código se removió
   como artefacto de columna vecina, pero el texto que lo sigue (ajeno al DS) **no** se
   truncó automáticamente para evitar borrar contenido por error. Queda como advertencia
   para revisión manual antes de chunking: si al construir el índice (Fase 2) se detecta
   contenido de otra norma colándose en un chunk, debe recortarse manualmente en
   `data/processed/clean/ds_001_2026_ef.jsonl`, página 16.
3. **Interleaving residual de doble columna.** El diseño en dos columnas de El Peruano
   hace que, en algunas páginas, `get_text(sort=True)` intercale fragmentos cortos de la
   columna vecina dentro de un párrafo (ej. el título "LEY N.° 32069" en medio de la
   definición de "Subcontratación", verificado en retrieval de prueba sobre
   `ley_32069:config_b:p002:c000`). No se intentó resolver con heurísticas de columnas
   (riesgo de introducir errores peores) porque el impacto observado en la calidad de
   retrieval fue bajo: la consulta de prueba "qué es la subcontratación" recuperó ese
   chunk en primer lugar con score 0.785. Queda como limitación aceptada, a revisar en
   la Fase 4 si el Recall@k de preguntas específicas se ve afectado.
