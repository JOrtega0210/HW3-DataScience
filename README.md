# HW_03 — Normative RAG and Public Procurement Radar

Proyecto integrador (curso Data Science, D2CML) que aplica Retrieval-Augmented
Generation (RAG) al dominio de contrataciones públicas en Perú, a través de dos
productos conectados.

- **Tarea 1 — RAG Normativo** (`tarea1_rag_normativo/`): asistente que responde
  preguntas sobre normas de contrataciones públicas citando documento y página,
  absteniéndose cuando la respuesta no está en el corpus indexado.
- **Tarea 2 — RAG Radar** (`tarea2_radar/`): dashboard que analiza datos abiertos
  de contrataciones estatales (portal OECE), combinando filtros estructurados con
  búsqueda semántica sobre descripciones de procesos, y reutilizando el motor de
  embeddings de la Tarea 1.

## Video de presentación

[Enlace pendiente de agregar tras la grabación] (≤12 minutos).

## Estructura del repositorio

```
├── README.md
├── requirements.txt
├── .env.example
├── tarea1_rag_normativo/
│   ├── config.yaml         # paths, thresholds, prompts, modelos
│   ├── build_index.py      # pipeline offline de indexación
│   ├── app.py               # interfaz Streamlit
│   ├── src/                 # motor RAG (sin dependencias de UI)
│   ├── eval/                # conjunto de evaluación y resultados
│   ├── data/{raw,processed}/
│   └── logs/
├── tarea2_radar/
│   ├── config.yaml
│   ├── app.py
│   ├── src/                 # acquisition / validation / index / metrics
│   ├── eval/
│   ├── data/{raw,processed,outputs}/
│   └── logs/
└── docs/
    └── pipeline.md          # diagramas offline/online de ambas tareas
```

## Setup (Windows)

```powershell
# 1. Clonar el repositorio
git clone https://github.com/JOrtega0210/HW3-DataScience.git
cd HW3-DataScience

# 2. Crear y activar entorno virtual (en la raíz del repo)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
copy .env.example .env
# Completar .env con las claves necesarias (ver sección "Credenciales")
```

## Credenciales

El pipeline offline de ambas tareas (extracción, limpieza, chunking, embeddings
locales, retrieval, cálculo de Recall@k) no requiere ninguna API key de pago.

| Variable | Uso |
|---|---|
| `GEMINI_API_KEY` | Generación de la respuesta final del LLM en ambas tareas (proveedor activo, `gemini-3.5-flash-lite`) |
| `OPENAI_API_KEY` | Proveedor alternativo de LLM (soportado en el código, no activo por defecto) |
| `OPENAI_EMBEDDINGS_API_KEY` | Comparación de embeddings local vs. `text-embedding-3-small` (Fase 4, Tarea 1) |
| `ANTHROPIC_API_KEY` | Proveedor alternativo de LLM adicional (soportado en el código) |
| `OECE_API_KEY` | Reservada para autenticación futura del portal OECE (hoy no la requiere) |

El motor de embeddings usa el modelo **local** por defecto en ambas tareas
(`embeddings.active_provider: "local"`); OpenAI se usa únicamente para la
comparación de la Fase 4 de la Tarea 1, documentada en esa sección.

## Cómo ejecutar — Tarea 1 (RAG Normativo)

### Fuentes oficiales

El corpus normativo consiste en la Ley N.° 32069 (Ley General de
Contrataciones Públicas) y el Decreto Supremo N.° 001-2026-EF, descargados
desde `busquedas.elperuano.pe` y versionados en `tarea1_rag_normativo/data/raw/`.
Para volver a descargarlos desde cero:

1. Abrir en el navegador la URL indicada en `config.yaml` → `documents[].url`.
2. Hacer clic en el botón **PDF** del visor.
3. Guardar el archivo en `data/raw/` con el nombre indicado en `raw_filename`.

### Pipeline offline

```powershell
cd tarea1_rag_normativo
python build_index.py --stage extract   # extracción + reporte de source check
python build_index.py --stage clean     # limpieza de headers + reporte de calidad
python build_index.py --stage chunk     # chunking (2 configuraciones)
python build_index.py --stage embed     # embeddings locales + índice FAISS
```

`--stage embed` genera, por cada configuración de chunking, un índice FAISS
(`IndexFlatIP` sobre embeddings normalizados L2, equivalente a similitud
coseno) en `data/processed/index/<config_name>/` (no versionado en git, se
regenera con el comando anterior). El proceso es idempotente: los embeddings
ya calculados se identifican por `chunk_id` y no se recalculan en corridas
posteriores.

| Config | Chunks totales | Tiempo de indexación inicial | Tiempo en corridas posteriores |
|---|---|---|---|
| `config_a` | 890 | 266.6 s | 14.0 s |
| `config_b` | 788 | 138.1 s | 0.0 s |

### Motor RAG (Fase 3)

`src/engine.py` expone una única función pública, `answer_question(query, config)`,
sin dependencias de UI. Encapsula retrieval, abstención por threshold de
similitud, citación de documento/página, nota de alcance del corpus,
generación con LLM (Gemini `gemini-3.5-flash-lite`) y logging de costo. Los
errores de API se devuelven como dato estructurado (`llm_error`) en lugar de
interrumpir la ejecución.

Validación con tres preguntas de control:

| Pregunta | Resultado |
|---|---|
| "¿Qué es la subcontratación...?" (dentro del dominio) | No se abstiene. Respuesta correcta citando Ley N.° 32069, página 2. Costo: ~USD 0.0002, 944/95 tokens de entrada/salida |
| "¿Cuál es la capital de Francia?" (fuera de dominio) | Se abstiene (similitud 0.232 < threshold 0.60). Costo USD 0.00, sin llamada al LLM |
| "¿Qué dice el reglamento sobre el procedimiento de selección?" | El retrieval no distingue que la pregunta se refiere al Reglamento (fuera de alcance) y recupera contenido de la Ley, pero el LLM reconoce el límite gracias a la nota de alcance inyectada en el prompt, y responde: *"No puedo proporcionar información específica sobre el reglamento... no forma parte del corpus indexado"* |

La tercera pregunta ilustra el diseño de dos líneas de defensa: el retrieval
por similitud no siempre distingue el alcance exacto del corpus, pero la nota
de alcance (`scope_note`) inyectada en cada respuesta permite que la
generación lo corrija.

El log de costos con llamadas reales se guarda en `logs/costs.csv` (modelo,
tokens de entrada/salida, costo en USD y latencia por consulta).

**Manejo de versiones normativas:** cada fragmento citado incluye su `version`
(`"original"` para la Ley, `"modificatoria"` para el Decreto Supremo). El DS
001-2026-EF modifica el Reglamento de la Ley 32069, no el texto de la Ley
indexado, por lo que no existen artículos duplicados en conflicto dentro del
corpus; esta distinción se documenta explícitamente en la nota de alcance.

### Evaluación y calibración de threshold (Fase 4)

Conjunto de evaluación en `eval/eval_set.json`: 21 preguntas (16 dentro del
dominio, 5 fuera de dominio), incluyendo 4 sobre artículos modificados por el
Decreto Supremo y 5 redactadas en tono de pequeño empresario.

```powershell
python eval/run_eval.py   # Recall@k sin llamar al LLM
```

**Recall@k** (`eval/results/recall_summary.csv`):

| Config | Recall@1 | Recall@3 | Recall@5 |
|---|---|---|---|
| `config_a` (96 tok) | 0.375 | 0.625 | 0.812 |
| `config_b` (120 tok, activa) | 0.438 | 0.750 | 0.750 |

**Calibración de threshold** (`eval/results/threshold_sweep.csv`): 0.60,
punto que mantiene 0% de abstención incorrecta sobre las 16 preguntas dentro
del dominio, con el mayor margen posible antes de que la tasa suba.

**Limitación conocida:** el threshold por similitud separa bien preguntas
totalmente ajenas al dominio (similitud ≈0.22 para "capital de Francia") pero
no distingue tres preguntas "cercanas al dominio" (sobre el Reglamento
excluido, sobre otra jurisdicción, sobre materia tributaria), cuya similitud
(0.70–0.78) supera la de varias preguntas legítimas. Subir el threshold para
capturarlas sacrificaría 19–38% de las preguntas reales. Por diseño, la nota
de alcance viaja en todas las respuestas y el `system_prompt` instruye al LLM
a aclarar el alcance del corpus — el retrieval es la primera línea de
defensa, la generación es la segunda. Detalle en `eval/results/eval_notes.md`.

**Comparación de embeddings — local vs. OpenAI:**

```powershell
python eval/run_openai_comparison.py
```

| Aspecto | Local (768 dim) | OpenAI `text-embedding-3-small` (1536 dim) |
|---|---|---|
| Recall@1 / @3 / @5 (`config_b`) | 0.438 / 0.750 / 0.750 | 0.375 / 0.750 / 0.875 |
| Recall@1 / @3 / @5 (`config_a`) | 0.375 / 0.625 / 0.812 | 0.500 / 0.625 / 0.688 |
| Tiempo de indexación | 266.6 s / 138.1 s (CPU local) | 14.6 s / 8.9 s (API) |
| Costo | USD 0.00 | USD 0.0021 / USD 0.0023 |
| Latencia por consulta | ~0.1–0.3 s | 0.48–0.78 s |

No hay un proveedor que domine en todas las métricas: OpenAI logra mejor
Recall@5 con `config_b` y mejor Recall@1 con `config_a`; el modelo local es
mejor en los casos restantes. Se mantiene el modelo local como proveedor
activo del motor porque el proyecto no incurre en costo variable por
consulta. Justificación completa en `eval/results/eval_notes.md`.

### Extracción y limpieza (Fase 1)

**Source check** (`data/processed/reports/source_check.csv`):

| Documento | Páginas | Caracteres totales | Prom. car/página | Mín. car/página | Máx. car/página | Páginas sin texto |
|---|---|---|---|---|---|---|
| Ley N.° 32069 | 36 | 316,158 | 8,782.2 | 948 | 13,087 | 0 |
| DS N.° 001-2026-EF | 16 | 134,755 | 8,422.2 | 8,031 | 9,144 | 0 |

El orden de lectura se preserva con `page.get_text("text", sort=True)` de
PyMuPDF, que ordena los bloques de texto por posición antes de concatenarlos.

**Limpieza** (`data/processed/reports/extraction_quality.csv`): cada página
trae pegado el encabezado repetido de la edición de El Peruano, removido en
esta etapa junto con el sello de firma digital y las ligaduras tipográficas
rotas.

| Documento | Páginas | Header removido | Sello firma removido | Códigos OP removidos | Ligaduras normalizadas | Portada excluida |
|---|---|---|---|---|---|---|
| Ley N.° 32069 | 36 | 35/36 (1 es portada) | 1 | 1 | 411 | página 1 |
| DS N.° 001-2026-EF | 16 | 16/16 | 1 | 2 | 0 | ninguna |

Limitaciones conocidas, documentadas en `extraction_quality_notes.md`:
espaciado irregular alrededor de ligaduras tipográficas en menos del 0.15% de
los caracteres, y contenido de una norma vecina que se filtra en la última
página del Decreto Supremo (dos normas comparten una misma página física en
la edición impresa) — se deja marcado para revisión manual en lugar de
recortarse automáticamente.

### Chunking (Fase 2)

El modelo de embeddings local
(`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`) tiene un
límite de `max_seq_length = 128` tokens, verificado en su
`sentence_bert_config.json`. Los tamaños de chunk se definieron por debajo de
ese límite (96 y 120 tokens) para evitar truncamiento silencioso al generar
los embeddings; `build_index.py --stage chunk` valida esta condición en cada
corrida.

El chunking se realiza por página (nunca cruza el límite de una página), con
IDs deterministas (`"{doc_id}:{config_name}:p{página}:c{índice}"`) que
garantizan resultados idénticos entre ejecuciones.

| Config | chunk_size / overlap (tokens) | Chunks Ley 32069 | Chunks DS 001-2026-EF | Prom. tokens/chunk |
|---|---|---|---|---|
| `config_a` | 96 / 16 | 584 | 306 | ~93.8 |
| `config_b` | 120 / 30 | 516 | 272 | ~117.0 |

Comparación completa en `data/processed/reports/chunking_comparison.csv`. Se
mantiene `config_b` como configuración activa por defecto (más contexto por
chunk); ambas se comparan en la Fase 4 con Recall@k.

### Interfaz Streamlit (Fase 5)

```powershell
streamlit run app.py
```

Carga el índice ya construido sin reconstruirlo al iniciar. Tres pestañas:
**Consulta** (pregunta → respuesta, costo, latencia y fragmentos citados con
página/versión/similitud), **Calidad de datos** (reportes de la Fase 1) y
**Evaluación** (Recall@k, comparación de embeddings y sweep de threshold de
la Fase 4).

Validado en navegador: la pregunta *"Tengo una empresa pequeña, ¿hasta cuánto
me pueden multar...?"* devuelve la respuesta correcta citando el Art. 89.3,
página 27, con costo real de USD 0.000185; la pregunta *"¿Cuál es la capital
de Francia?"* se abstiene con costo USD 0.00 y 0.51 s de latencia.

## Cómo ejecutar — Tarea 2 (RAG Radar)

### Adquisición de datos (Fase 1)

```powershell
cd tarea2_radar
python acquire_data.py --stage bulk     # descarga masiva: 3 meses de 2026
python acquire_data.py --stage recent   # API de actualizaciones recientes (últimos 7 días)
python acquire_data.py --stage combine  # une ambos en una fila por proceso (por ocid)
python acquire_data.py --stage report   # resumen de tiempo/requests/tamaños
# o los cuatro pasos juntos:
python acquire_data.py --stage all
```

Fuente: portal OECE (`contratacionesabiertas.oece.gob.pe`), estándar OCDS.

- **Descarga masiva** (`GET /api/v1/file/{source}/{type}/{year}/{month}`): sin
  autenticación. Se descargaron 3 meses reales (2026-06, 2026-07, 2026-08,
  ~30 MB comprimidos en total).
- **API de actualizaciones recientes** (`GET /api/v1/records?startDate=...&endDate=...`):
  filtra por fecha de convocatoria en el servidor, usada para traer los
  últimos 7 días sin depender de la publicación del próximo archivo mensual.
- **Modelo OCDS:** cada `record` del `recordPackage` ya es la vista compilada
  (`compiledRelease`) de todas las `releases` (eventos de planificación,
  convocatoria, adjudicación) de un mismo `ocid`, por lo que una fila por
  `ocid` representa un proceso de contratación completo.

**Consideración de la API:** el parámetro `size` no es respetado por el
servidor (siempre pagina de 20 resultados); la paginación se implementó
siguiendo el enlace `links.next` que devuelve cada página, en lugar de
calcular el offset localmente, para garantizar que no se pierdan resultados.

**Resultados de una corrida representativa:**

| Fuente | Procesos | Requests | Datos transferidos | Tiempo |
|---|---|---|---|---|
| Descarga masiva (3 meses) | 20,441 | 6 | 29.7 MB | 33.8 s |
| API de actualizaciones recientes (7 días) | 1,309 | 67 (+66 desde caché en la 2ª corrida) | 6.0 MB | 41.5 s |
| **Combinado (una fila por `ocid`)** | **21,750** | | | |

> La ventana de "últimos 7 días" es relativa a la fecha de ejecución, por lo
> que estos conteos varían ligeramente entre corridas. Los conteos vigentes
> se encuentran en `data/outputs/acquisition_summary.csv` y
> `data/processed/processes_validated.csv`.

Ambos mecanismos son re-ejecutables sin duplicar información: la descarga
masiva omite meses ya extraídos (`data/raw/extracted/`) y la API cachea cada
página consultada (`data/raw/api_cache/`), de modo que una segunda corrida no
genera requests adicionales. El log de tiempo/requests/tamaños se guarda en
`logs/acquisition_log.csv`.

Los archivos crudos (`data/raw/extracted/`, `data/raw/api_cache/`, ~290 MB) no
se versionan en git; se regeneran con el comando de adquisición. El dataset
combinado intermedio tampoco se versiona porque queda superado por el
dataset validado de la Fase 2.

### Validación de calidad y normalización territorial (Fase 2)

```powershell
python validate_data.py
```

Seis reglas de calidad sobre el dataset combinado de la Fase 1:

| Regla | Casos | Acción tomada |
|---|---|---|
| Registros repetidos por el mismo proceso | 884 (827 grupos) | se conserva el registro más completo por grupo; auditoría en `duplicates_removed.csv` |
| Proceso sin monto | 0 | — |
| Proceso con monto cero | 3,419 | se conserva (válido en catálogos y convenios); documentado para no inflar promedios sin advertencia |
| Proceso sin descripción | 0 | — |
| Ubicación que no corresponde a un departamento | 0 | `buyer.address.department` del estándar OCDS peruano ya viene normalizado por una extensión propia del OECE |
| Inconsistencia de encoding o acentos | 0 | — |

**Registros repetidos:** se detectaron grupos donde el mismo comprador, la
misma nomenclatura y la misma fecha de convocatoria aparecen bajo distintos
`ocid` — por ejemplo, dos registros con monto idéntico y mismo título, uno
sin postores registrados aún y otro con 25, correspondientes a una
reemisión del identificador interno del sistema de origen para el mismo
proceso real. Se conserva el registro con más campos poblados (heurística de
completitud) y nunca se combinan ni promedian los valores.

La resolución de duplicados es determinística: el desempate entre candidatos
con igual nivel de completitud usa el propio `ocid` como criterio final, de
modo que dos ejecuciones sobre el mismo conjunto de datos producen
exactamente el mismo resultado (verificado comparando la salida de dos
corridas consecutivas byte a byte).

**Normalización territorial:** sobre las filas reales del dataset aparecen
exactamente 25 valores únicos de departamento, coincidentes con los 25
departamentos oficiales, sin variantes de acento/mayúsculas ni provincias
mezcladas — el campo ya viene curado por la extensión OCDS del OECE. La
función `normalize_department()` (`src/validation.py`) implementa de todas
formas comparación insensible a tildes y mayúsculas contra la lista oficial
de `config.yaml`, con reporte de ubicaciones no localizables en
`unmatched_locations.csv` para datasets menos curados.

Dataset final: `data/processed/processes_validated.jsonl` / `.csv`, con
columnas adicionales `buyer_department` (normalizado) e `is_encoding_issue`.

### RAG híbrido (Fase 3)

```powershell
python build_hybrid_index.py     # embebe título + descripción de cada proceso
python eval/run_hybrid_eval.py   # Recall@k sin LLM, sobre 12 preguntas conocidas
```

Reutiliza el modelo de embeddings local de la Tarea 1, con un embedding por
proceso (la descripción cabe en ~50–70 tokens, sin necesidad de chunking). El
build es idempotente y reanudable con checkpointing incremental cada 250
procesos, de modo que una interrupción no obliga a reprocesar el índice
completo.

`src/hybrid_engine.py` expone `answer_question(query, config, filters)`: los
filtros estructurados (`departamento`, `categoria`, `monto_min/max`,
`fecha_desde/hasta`) se aplican siempre sobre metadata antes de tocar los
embeddings — nunca se convierten en texto para la búsqueda semántica.

**Recall@k** (`eval/results/hybrid_retrieval_detail.csv`, 12 preguntas con
proceso relevante conocido): Recall@1=0.250, Recall@3=0.333, Recall@5=0.417 —
sensiblemente más bajo que en la Tarea 1. Dos limitaciones identificadas:

1. **El lenguaje burocrático repetitivo dificulta la discriminación fina del
   embedding a esta escala.** Ejemplo: una pregunta sobre "mejoramiento de
   agua potable en San Sebastián de Choropampa" (Cajamarca) recupera un
   proceso distinto, de otra localidad del mismo departamento, con mayor
   similitud — ambas descripciones son casi idénticas salvo el nombre del
   centro poblado, y el modelo no discrimina bien nombres de localidades
   pequeñas y poco frecuentes. El filtro de departamento reduce el universo
   de búsqueda (de 20,870 a 874 candidatos en este caso) pero no resuelve la
   discriminación entre localidades dentro del mismo departamento.
2. **A esta escala, ningún threshold único separa limpiamente preguntas
   dentro y fuera de dominio.** Una pregunta sin relación con el corpus (ej.
   sobre gastronomía) puede obtener mayor similitud que preguntas legítimas.
   El threshold se recalibró de 0.60 (heredado de la Tarea 1) a 0.45,
   priorizando no abstenerse de preguntas reales; la defensa principal ante
   preguntas fuera de dominio recae en el LLM, que cita únicamente por
   `ocid` y se niega a responder cuando los procesos recuperados no son
   pertinentes. Detalle completo en `eval/results/hybrid_eval_notes.md`.

El motor soporta Gemini, OpenAI y Anthropic como proveedores de LLM
(`llm.provider` en `config.yaml`); el proveedor activo en ambas tareas es
Gemini (`gemini-3.5-flash-lite`, USD 0.30 / 1M tokens de entrada, USD 2.50 /
1M de salida).

Validación funcional tras incluir la descripción del proceso en el contexto
enviado al LLM (además del título):

| Pregunta | Resultado |
|---|---|
| "Tengo una empresa pequeña, ¿hay agua potable en Cajamarca?" | Respuesta correcta citando el `ocid`, monto y entidad reales |
| "¿Cómo se prepara un ceviche peruano?" | El retrieval no se abstiene (ver limitación de separabilidad arriba), pero el LLM se niega a responder, explicando que los procesos recuperados corresponden a insumos agropecuarios y no a gastronomía — la segunda línea de defensa opera correctamente aun cuando el retrieval no discrimina |

### Dashboard Streamlit (Fase 4)

```powershell
streamlit run app.py
```

Lee únicamente archivos precomputados de las fases anteriores — no descarga
ni reconstruye nada al iniciar. Seis pestañas, con un panel de KPIs siempre
visible (procesos, monto total, departamentos, share monopostor):

1. **Mapa** — choropleth por departamento (procesos o monto).
2. **Pregunta (RAG híbrido)** — filtros exactos de departamento, categoría y
   monto, combinados con la pregunta en lenguaje natural; threshold
   ajustable desde el sidebar.
3. **Tabla** — ordenable, con descarga en CSV según los filtros aplicados.
4. **Distribución** — monto por categoría, procesos por mes, top
   departamentos por monto.
5. **Indicador de riesgo** — con la advertencia de interpretación siempre
   visible.
6. **Calidad de datos** — reportes de las fases de adquisición y validación.

El sidebar incluye filtros de departamento, categoría, rango de monto, rango
de fecha y threshold de similitud; una selección sin resultados se maneja
mostrando un aviso en lugar de una falla.

Validado en navegador: al aplicar el filtro `departamento=Cajamarca` en la
pestaña de pregunta, la consulta sobre agua potable rural en Choropampa
reproduce el mismo comportamiento documentado en la evaluación offline (874
candidatos tras el filtro, mismo proceso de mayor similitud), confirmando
consistencia entre el dashboard y la evaluación.

### Indicador de riesgo — adjudicaciones monopostor (Fase 5)

```powershell
python compute_risk_indicator.py
```

Calcula el share de procesos adjudicados con exactamente un postor, por
departamento y por proveedor (`src/risk_indicator.py`). El dashboard muestra
siempre la advertencia: un solo postor no es evidencia de irregularidad — puede
reflejar un mercado con poca oferta o una contratación muy especializada; es
una señal estadística para priorizar revisión, no una acusación.

Top 5 departamentos por share:

| Departamento | Share monopostor | Procesos (monopostor / adjudicados) |
|---|---|---|
| Tumbes | 36.7% | 44 / 120 |
| Lima | 30.2% | 1,175 / 3,886 |
| Amazonas | 16.2% | 35 / 216 |
| Arequipa | 14.2% | 78 / 548 |
| La Libertad | 10.3% | 50 / 486 |

El ranking de proveedores solo considera entidades con al menos 5 procesos
adjudicados (`risk_indicator.min_processes_per_entity` en `config.yaml`), ya
que con menos casos un share de 100% no es estadísticamente representativo.
Los proveedores que no corresponden a un patrón de persona jurídica (sin
"S.A.C.", "S.R.L.", "CONSORCIO", etc. en el nombre) se anonimizan como
*"(persona natural #N — nombre no publicado)"*, para no exponer identidades
individuales en un indicador de riesgo agregado.

## Pipelines

Ver [`docs/pipeline.md`](docs/pipeline.md) para los diagramas offline/online
de ambas tareas.

## Resultados

**Tarea 1 — RAG Normativo**

| Métrica | Resultado |
|---|---|
| Source check | Ley 32069: 36 pág. / 316,158 caracteres · DS 001-2026-EF: 16 pág. / 134,755 caracteres · 0 páginas sin texto extraíble |
| Chunking | `config_a` (96 tok): 890 chunks · `config_b` (120 tok, activa): 788 chunks |
| Recall@1 / @3 / @5 (local, `config_b`) | 0.438 / 0.750 / 0.750 |
| Recall@1 / @3 / @5 (OpenAI, `config_b`) | 0.375 / 0.750 / 0.875 |
| Threshold de abstención | 0.60 (calibrado con sweep sobre 21 preguntas) |
| Costo por consulta (Gemini) | ~USD 0.0002–0.0006; USD 0.00 en abstenciones |

**Tarea 2 — RAG Radar**

| Métrica | Resultado |
|---|---|
| Adquisición | 20,441 procesos (3 meses) + ~1,300 (API de actualizaciones recientes) |
| Validación | 884 duplicados removidos → ~20,870 filas finales |
| Departamentos representados | 25 / 25 |
| Recall@1 / @3 / @5 (RAG híbrido) | 0.250 / 0.333 / 0.417 |
| Threshold de abstención | 0.45 |
| Indicador de riesgo monopostor | Tumbes 36.7%, Lima 30.2% (departamentos con mayor share) |
| Costo por consulta (Gemini) | ~USD 0.0004–0.0008 |

Los reportes fuente de cada número están en `eval/results/`,
`data/processed/reports/` y `data/outputs/` de cada tarea; el log de costos
con llamadas reales está en `logs/costs.csv`.
