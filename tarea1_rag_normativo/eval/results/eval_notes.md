# Notas de evaluación (Tarea 1, Fase 4)

Generado a partir de `python eval/run_eval.py` sobre `eval_set.json` (21 preguntas:
16 in-domain + 5 out-of-domain) y los índices de `config_a` y `config_b`.
Ninguna métrica de esta fase llama al LLM (solo retrieval).

## Recall@k

| Config | Recall@1 | Recall@3 | Recall@5 |
|---|---|---|---|
| `config_a` (96 tok / 16 overlap) | 0.375 | 0.625 | 0.812 |
| `config_b` (120 tok / 30 overlap) | 0.438 | 0.750 | 0.750 |

`config_b` recupera mejor en Recall@1/@3 (chunks más grandes = más contexto por
fragmento, más fácil que contenga la respuesta exacta); `config_a` recupera
levemente mejor en Recall@5 (fragmentos más finos = más oportunidades de match,
aunque no necesariamente en el top-3). Se mantiene `config_b` como activa
(mejor en las métricas más estrictas, que son las más relevantes porque
`retrieval.top_k=5` ya trae ambos casos dentro del corte).

Recall@1 relativamente bajo (0.375–0.438) es consistente con el tamaño pequeño
del corpus (2 documentos, ~52 páginas) y el ruido de layout documentado en
`extraction_quality_notes.md` (interleaving de columnas). Con Recall@5 en
0.75–0.81, el usuario encuentra la respuesta correcta citada en la mayoría de
los casos entre los 5 fragmentos mostrados en la interfaz (Fase 5).

## Calibración del threshold de abstención

Sweep sobre las similitudes top-1 de las 21 preguntas (`threshold_sweep.csv`),
buscando el punto que separa mejor in-domain (no debería abstenerse) de
out-of-domain (debería abstenerse):

| Threshold | Abstención incorrecta (in-domain) | Abstención correcta (out-of-domain) |
|---|---|---|
| 0.30 – 0.60 | 0% | 40% (2/5) |
| 0.65 | 6% | 40% |
| 0.70 | 19–25% | 40% |
| 0.725 | 25–38% | 60% |
| 0.75+ | 63–88% | 80–100% |

Se eligió **0.60** (config activa `config_b`): es el borde superior de la
meseta donde la abstención incorrecta sobre preguntas reales sigue en 0%, con
el margen más amplio posible antes de que empiece a subir.

### Limitación identificada: preguntas "cercanas al dominio"

El threshold por similitud separa bien las preguntas totalmente ajenas al
dominio (`"capital de Francia"` sim≈0.22–0.25, `"receta de ceviche"`
sim≈0.36–0.38 → correctamente abstenidas en 0.60) pero no separa las
preguntas "cercanas al dominio": tres preguntas sobre el Reglamento excluido,
sobre contratación pública en otra jurisdicción, y sobre beneficios
tributarios obtienen similitudes de 0.70–0.78 — más altas que varias
preguntas in-domain reales (ej. sim=0.62–0.67). Esto ocurre porque comparten
vocabulario con el corpus ("contratación", "Estado", "empresa") aunque la
respuesta correcta no esté ahí.

**Implicación de diseño:** el threshold de similitud actúa como primera línea
de defensa (evita gastar tokens de LLM en preguntas obviamente ajenas), pero
no es suficiente por sí solo para las preguntas cercanas al dominio. Por eso
`scope_note` viaja en todas las respuestas del motor y el `system_prompt`
instruye al LLM a aclarar explícitamente los límites del corpus — la segunda
línea de defensa recae en la generación, no solo en el retrieval. Subir el
threshold para atrapar estos casos (a partir de ~0.70) sacrificaría 19–38% de
las preguntas reales, un peor trade-off para el producto.

## Limitación identificada: Recall@k por página no garantiza el chunk correcto

Al validar el motor en la interfaz, la pregunta *"¿Cuáles son los
procedimientos de selección competitivos regulados en la ley?"* recupera
correctamente la página 17 como top-1 (sim≈0.79, coincide con lo que mide
`run_eval.py`), pero el chunk específico recuperado
(`ley_32069:config_b:p017:c004`) no contiene la enumeración real ("a)
licitación pública... b) concurso público"), presente en los chunks `c000` y
`c003` de esa misma página. La página 17 se dividió en 14 chunks solapados, y
varios repiten la frase "procedimiento de selección competitivo" en contextos
distintos (la definición en el Art. 54.1 y una excepción del Art. 55 que la
reutiliza); el chunk de la excepción concentra más densamente esa frase y por
eso puntúa más alto por similitud coseno, aunque no responda la pregunta. Se
confirmó que el efecto no depende de tildes: la misma consulta con y sin
acentos recupera el mismo chunk con similitud casi idéntica (0.7926 vs. 0.7939).

El LLM, siguiendo el `system_prompt`, respondió que los fragmentos recuperados
no contienen información específica en lugar de generar una respuesta
incorrecta — el diseño de abstención por prompt funciona como red de
seguridad, aunque la respuesta correcta sí estaba disponible en el índice
bajo un chunk distinto.

**Implicación:** Recall@k medido a nivel de página es una métrica más
generosa que la precisión real percibida por el usuario a nivel de chunk. En
corpus normativos donde una misma página repite terminología en artículos
distintos, el overlap alto entre chunks puede generar fragmentos muy
similares que compiten por el mismo top-k sin que el más relevante gane
necesariamente. Mejora identificada para una futura iteración: medir Recall@k
a nivel de `chunk_id` exacto (no solo documento+página) y/o incorporar un
rerank con modelo cross-encoder antes de construir el contexto para el LLM.

## Comparación de embeddings (local vs. OpenAI)

Ejecutada con `eval/run_openai_comparison.py` (costo total inferior a un
centavo de dólar).

| Aspecto | Local (`paraphrase-multilingual-mpnet-base-v2`) | `text-embedding-3-small` (OpenAI) |
|---|---|---|
| Recall@1 (config_a / config_b) | 0.375 / 0.438 | 0.500 / 0.375 |
| Recall@3 (config_a / config_b) | 0.625 / 0.750 | 0.625 / 0.750 |
| Recall@5 (config_a / config_b) | 0.812 / 0.750 | 0.688 / 0.875 |
| Tiempo de indexación (890 / 788 chunks) | 266.6s / 138.1s (CPU local) | 14.6s / 8.9s (API remota) |
| Costo | USD 0.00 | USD 0.002082 (config_a) / USD 0.002298 (config_b) |
| Latencia promedio por consulta | ~0.1–0.3s | 0.777s / 0.476s (round-trip de red) |
| Dimensión | 768 | 1536 |
| Requisito | Local (descarga ~420MB una sola vez) | Remoto (API key + red) |

No hay un proveedor que domine en todas las métricas: OpenAI logra mejor
Recall@5 con `config_b` pero pierde en Recall@1; con `config_a` ocurre lo
inverso. La indexación es 15–18x más rápida con la API (sin competir por CPU
local), a cambio de depender de la red y de un costo marginal por token
(<USD 0.003 por config en este corpus). Dado que el proyecto corre 100% local
por defecto sin costo variable por consulta, se mantiene el modelo local como
proveedor activo (`embeddings.active_provider: "local"` en `config.yaml`);
OpenAI queda documentado como alternativa viable si se requiere reducir el
tiempo de indexación o evitar mantener el modelo localmente.
