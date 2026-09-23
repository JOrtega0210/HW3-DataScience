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

## Comparación de embeddings (local vs. OpenAI)

| Aspecto | Local (`paraphrase-multilingual-mpnet-base-v2`) | `text-embedding-3-small` (OpenAI) |
|---|---|---|
| Recall@1 / @3 / @5 (config_b) | 0.438 / 0.750 / 0.750 | **pendiente** — requiere `OPENAI_EMBEDDINGS_API_KEY` |
| Tiempo de indexación (890 chunks, 1ª corrida) | 266.6s (config_a) / 138.1s (config_b), CPU | pendiente |
| Costo | USD 0.00 | pendiente (según tarifa vigente de OpenAI) |
| Latencia promedio por consulta | ~0.1–0.3s (tras cache de modelo en memoria) | pendiente |
| Dimensión | 768 | 1536 |

La mitad local de la comparación ya está completa y documentada; la mitad OpenAI
queda pendiente hasta contar con la API key (ver README, sección Credenciales).
