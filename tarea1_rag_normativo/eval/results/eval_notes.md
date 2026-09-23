# Notas de evaluación (Tarea 1, Fase 4)

Generado a partir de `python eval/run_eval.py` sobre `eval_set.json` (21 preguntas:
16 in-domain + 5 out-of-domain) y los índices reales de `config_a` y `config_b`.
Ninguna métrica de esta fase llama al LLM (solo retrieval).

## Recall@k

| Config | Recall@1 | Recall@3 | Recall@5 |
|---|---|---|---|
| `config_a` (96 tok / 16 overlap) | 0.375 | 0.625 | 0.812 |
| `config_b` (120 tok / 30 overlap) | 0.438 | 0.750 | 0.750 |

`config_b` recupera mejor en Recall@1/@3 (chunks más grandes = más contexto por
fragmento, más fácil que contenga la respuesta exacta); `config_a` recupera
levemente mejor en Recall@5 (más fragmentos más finos = más oportunidades de que
alguno haga match, aunque no en el top-3). Se mantiene `config_b` como activa
(mejor en las métricas más estrictas, que son las más relevantes porque
`retrieval.top_k=5` ya trae ambos casos dentro del corte).

Recall@1 relativamente bajo (0.375–0.438) es consistente con lo esperado dado el
tamaño pequeño del corpus (2 documentos, ~52 páginas) y el ruido de layout
documentado en `extraction_quality_notes.md` (interleaving de columnas). Con
Recall@5 en 0.75–0.81, el usuario casi siempre encuentra la respuesta correcta
citada en alguno de los 5 fragmentos mostrados en la interfaz (Fase 5).

## Calibración del threshold de abstención

Sweep sobre las similitudes top-1 de las 21 preguntas (`threshold_sweep.csv`),
buscando el punto que separa mejor in-domain (no debería abstenerse) de
out-of-domain (debería abstenerse):

| Threshold | Abstención incorrecta (in-domain) | Abstención correcta (out-of-domain) |
|---|---|---|
| 0.30 – 0.60 | **0%** | 40% (2/5) |
| 0.65 | 6% | 40% |
| 0.70 | 19–25% | 40% |
| 0.725 | 25–38% | 60% |
| 0.75+ | 63–88% | 80–100% |

**Se eligió 0.60** (config activa `config_b`): es el borde superior de la meseta
donde la abstención incorrecta sobre preguntas reales sigue siendo 0%, con el
margen más amplio posible antes de que empiece a subir.

### Hallazgo importante (limitación real, no oculta)

El threshold por similitud **separa bien las preguntas totalmente ajenas al
dominio** (`"capital de Francia"` sim≈0.22–0.25, `"receta de ceviche"`
sim≈0.36–0.38 → correctamente abstenidas en 0.60) **pero no separa las preguntas
"cercanas al dominio"**: las 3 preguntas `out_domain_near` (sobre el Reglamento
excluido, sobre contratación pública en Chile, sobre beneficios tributarios)
obtienen similitudes de 0.70–0.78 — **más altas que varias preguntas in-domain
reales** (ej. q08 sim=0.62–0.67, q09 sim=0.64). Esto ocurre porque comparten
vocabulario con el corpus ("contratación", "Estado", "empresa") aunque la
respuesta correcta no esté ahí.

**Implicación de diseño:** el threshold de similitud es una primera línea de
defensa (atrapa preguntas obviamente ajenas antes de gastar tokens de LLM), pero
**no es suficiente por sí solo** para las preguntas "trampa" cercanas al dominio.
Por eso `scope_note` viaja en *todas* las respuestas del motor (Fase 3) y el
`system_prompt` (`config.yaml`) instruye al LLM a aclarar explícitamente los
límites del corpus — la segunda línea de defensa recae en la generación, no solo
en el retrieval. Subir el threshold para atrapar estos 3 casos (a partir de
~0.70) sacrificaría 19–38% de las preguntas reales, un peor trade-off para el
producto (preferible no abstenerse una pregunta real, apoyándose en el LLM para
aclarar el alcance, que negarle la respuesta a un usuario legítimo).

## Hallazgo de prueba en vivo: Recall@k por página no garantiza el chunk correcto

Detectado al probar la app real (no solo el eval automatizado): la pregunta
*"¿Cuáles son los procedimientos de selección competitivos regulados en la ley?"*
recupera correctamente la **página** 17 como top-1 (sim≈0.79, coincide con lo que
mide `run_eval.py`), pero el **chunk** específico recuperado
(`ley_32069:config_b:p017:c004`) no contiene la enumeración real ("a) licitación
pública... b) concurso público", que sí está en los chunks `c000` y `c003` de esa
misma página). El motivo: la página 17 se partió en 14 chunks solapados, y varios
repiten la frase "procedimiento de selección competitivo" en contextos distintos
(la definición real en el Art. 54.1, y una excepción del Art. 55 que la reutiliza);
el chunk de la excepción anota más densamente esa frase y por eso puntúa más alto
por similitud coseno, aunque no responda la pregunta.

Verificado que no es un problema de tildes (se probó la misma consulta con y sin
acentos: ambas recuperan exactamente el mismo top-1 chunk con similitud casi
idéntica, 0.7926 vs 0.7939).

**Consecuencia observada:** el LLM, siguiendo el `system_prompt` ("si los
fragmentos no contienen la respuesta, indícalo en vez de inventar"), respondió
correctamente que *"los fragmentos recuperados no contienen información
específica..."* en vez de alucinar una respuesta — el diseño de abstención por
prompt funcionó como red de seguridad, aunque la respuesta ideal (con el chunk
correcto) sí estaba disponible en el índice.

**Implicación para el reporte de "hallazgos y limitaciones" del video:** Recall@k
tal como se mide aquí es a nivel de **página**, no de **chunk** — es una métrica
más generosa que la precisión real percibida por el usuario. Con corpus normativos
donde una misma página repite terminología en artículos distintos, el overlap alto
entre chunks (30 tokens en `config_b`) puede generar "casi duplicados" que
compiten por el mismo top-k sin que el correcto necesariamente gane. Mejora futura
no implementada por alcance de tiempo: Recall@k a nivel de chunk_id exacto (no solo
doc+página) en el eval set, y/o rerank con un modelo cross-encoder antes de pasar
contexto al LLM.

## Comparación de embeddings (local vs. OpenAI)

Completada con `eval/run_openai_comparison.py` (API key real, costo real
< 1 centavo de dólar en total).

| Aspecto | Local (`paraphrase-multilingual-mpnet-base-v2`) | `text-embedding-3-small` (OpenAI) |
|---|---|---|
| Recall@1 (config_a / config_b) | 0.375 / 0.438 | 0.500 / 0.375 |
| Recall@3 (config_a / config_b) | 0.625 / 0.750 | 0.625 / 0.750 |
| Recall@5 (config_a / config_b) | 0.812 / 0.750 | 0.688 / **0.875** |
| Tiempo de indexación (890 / 788 chunks) | 266.6s / 138.1s (CPU local, 1ª corrida) | **14.6s / 8.9s** (API remota) |
| Costo real | USD 0.00 | USD 0.002082 (config_a) / USD 0.002298 (config_b) |
| Latencia promedio por consulta | ~0.1–0.3s (modelo ya cargado en memoria) | 0.777s / 0.476s (round-trip de red) |
| Dimensión | 768 | 1536 |
| Requisito | Local (descarga ~420MB una sola vez) | Remoto (API key + red) |

**No hay un ganador universal:** OpenAI gana en Recall@5 para `config_b` (0.875 vs.
0.750) pero pierde en Recall@1 (0.375 vs. 0.438); para `config_a` es al revés
(gana Recall@1, pierde Recall@5). La indexación es ~15-18x más rápida con la API
(sin competir por CPU local), a cambio de depender de red y de un costo por token
(aunque marginal en este corpus: <$0.003 por config). Dado que el proyecto corre
100% local por defecto sin costo variable por consulta, **se mantiene el modelo
local como proveedor activo** (`embeddings.active_provider: "local"` en
`config.yaml`); OpenAI queda documentado como alternativa viable si se necesita
reducir tiempo de build o evitar mantener el modelo localmente.
