# HW_03 — Normative RAG and Public Procurement Radar

Proyecto integrador (curso Data Science, D2CML) que combina dos tareas conectadas
aplicando Retrieval-Augmented Generation (RAG) al dominio de contrataciones públicas en Perú.

- **Tarea 1 — RAG Normativo** (`tarea1_rag_normativo/`): asistente que responde preguntas
  sobre normas de contrataciones públicas citando documento y página, absteniéndose cuando
  la respuesta no está en el corpus indexado.
- **Tarea 2 — RAG Radar** (`tarea2_radar/`): dashboard que analiza datos abiertos de
  contrataciones estatales (portal OECE) combinando filtros estructurados con búsqueda
  semántica sobre descripciones de procesos, reutilizando el motor de la Tarea 1.

> Estado: repositorio en construcción, se completa por fases. Este README se actualiza a
> medida que cada fase queda cerrada (ver checklist más abajo).

## Video de presentación

_Pendiente — se agregará el enlace aquí una vez grabado (≤12 minutos, regla pipeline-primero)._

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
# 1. Clonar el repo y entrar a la carpeta
git clone <URL_DEL_REPO>
cd HW3

# 2. Crear y activar entorno virtual (en la raíz del repo)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
copy .env.example .env
# Editar .env y completar las claves necesarias (ver sección "Credenciales" abajo)
```

## Credenciales

El proyecto se desarrolló de forma incremental sin requerir API keys de pago en las
primeras fases (extracción, limpieza, chunking, embeddings locales, retrieval,
evaluación de Recall@k). Actualmente conectado con una **API key de OpenAI**:

| Variable | Uso | Estado |
|---|---|---|
| `OPENAI_API_KEY` | Generación de la respuesta final del LLM (`gpt-4o-mini`, Fase 3) | ✅ conectada |
| `OPENAI_EMBEDDINGS_API_KEY` | Comparación local vs. `text-embedding-3-small` (Fase 4) | ✅ conectada (usa la misma key que `OPENAI_API_KEY` si no se define aparte) |
| `ANTHROPIC_API_KEY` | Alternativa de LLM (no usada; `llm.provider` en `config.yaml` está en `"openai"`) | vacía, opcional |
| `OECE_API_KEY` | Solo si el portal OECE introduce autenticación (hoy no la requiere) | vacía, no aplica aún |

El motor sigue usando el modelo de embeddings **local** por defecto para la app
(`embeddings.active_provider: "local"`) — OpenAI se usó puntualmente para la
comparación de la Fase 4, no como proveedor de producción (ver esa sección).

## Cómo ejecutar — Tarea 1 (RAG Normativo)

### Descarga de las fuentes oficiales

Los PDFs de Ley N.° 32069 y DS N.° 001-2026-EF **no son descargables con `curl`/`requests`
directo**: `busquedas.elperuano.pe` genera la URL real del archivo (`/api/archivo/file/<token>/...`)
mediante JavaScript, con un token firmado de corta duración. Pasos para obtenerlos:

1. Abrir en el navegador la URL de `config.yaml` -> `documents[].url` (páginas
   `dispositivo/SE/...` o `dispositivo/NL/...`).
2. Click en el botón **PDF** del visor.
3. Guardar el PDF resultante en `tarea1_rag_normativo/data/raw/` con el nombre indicado
   en `raw_filename` (`ley_32069.pdf`, `ds_001_2026_ef.pdf`).

Ya se descargaron y están versionados en el repo, así que este paso no es necesario para
reproducir el resto del pipeline — solo aplica si se quiere volver a descargar desde cero.

### Ejecutar el pipeline offline

```powershell
cd tarea1_rag_normativo
python build_index.py --stage extract   # Fase 1a: extracción + source check
python build_index.py --stage clean     # Fase 1b: limpieza de headers + reporte de calidad
python build_index.py --stage chunk     # Fase 2a: chunking (2 configuraciones)
python build_index.py --stage embed     # Fase 2b: embeddings locales + índice FAISS
```

`--stage embed` genera, por cada configuración de chunking, un índice FAISS
(`IndexFlatIP` sobre embeddings normalizados L2 = similitud coseno) en
`data/processed/index/<config_name>/` (no versionado en git — se regenera con el
comando anterior; solo tarda ~7 minutos la primera vez porque descarga el modelo, ~14s
en corridas posteriores). Es **idempotente y reanudable**: los embeddings ya calculados
se guardan por `chunk_id`, así que una segunda corrida no reencoda nada (verificado:
0 nuevos / 890 reusados en `config_a`, 0 nuevos / 788 reusados en `config_b`).

| Config | Chunks totales | Tiempo (1ª corrida, con descarga de modelo) | Tiempo (corrida repetida) |
|---|---|---|---|
| `config_a` | 890 | 266.6s | 14.0s |
| `config_b` | 788 | 138.1s | 0.0s |

Prueba de humo end-to-end (extracción -> limpieza -> chunking -> embedding -> FAISS):
la consulta *"qué es la subcontratación en las contrataciones públicas"* recuperó como
primer resultado (score coseno 0.785) el chunk de `ley_32069`, página 2, que contiene
exactamente la definición legal de "Subcontratación" — confirma que el pipeline
completo de retrieval funciona antes de conectar el LLM (Fase 3).

### Motor RAG (Fase 3)

`src/engine.py` expone una única función pública, `answer_question(query, config)`,
sin ninguna dependencia de UI (verificado: importar `src.engine` no carga Streamlit).
Encapsula retrieval + abstención por threshold + citas + nota de alcance + generación
con LLM (OpenAI `gpt-4o-mini`, conectado con API key real) + logging de costo; los
errores de API se devuelven como `llm_error` estructurado en vez de fallar.

Prueba con 3 preguntas de control (con el LLM real ya conectado):

| Pregunta | Resultado |
|---|---|
| "¿Qué es la subcontratación...?" (in-domain) | No se abstiene. Respuesta correcta citando Ley N.° 32069, página 2. Costo real: **944 in / 95 out tokens ≈ USD 0.0002**, 26.1s |
| "¿Cuál es la capital de Francia?" (out-of-domain) | **Se abstiene** (sim=0.232 < threshold 0.60) — **0 tokens gastados**, 0.23s |
| "¿Qué dice el reglamento sobre el procedimiento de selección?" | No se abstiene en retrieval (sim=0.751, trae contenido de la Ley), pero el LLM **reconoce el límite de alcance** y responde: *"No puedo proporcionar información específica sobre el reglamento... el reglamento no forma parte del corpus indexado."* Costo: 843 in / 50 out tokens ≈ USD 0.00016 |

La tercera pregunta es la validación en vivo del diseño de dos líneas de defensa: el
retrieval por similitud **no** distingue que la pregunta es sobre el Reglamento
(excluido), pero el `scope_note` inyectado en el prompt hace que el LLM sí lo detecte
y lo aclare en vez de responder como si estuviera dentro del corpus — confirma que la
segunda línea de defensa (generación) funciona como se diseñó en la Fase 3, antes de
tener resultados del sweep de threshold de la Fase 4.

Log de costos con llamadas reales versionado en `logs/costs.csv` (deliverable
explícito del enunciado) — cada fila es una consulta real, con modelo, tokens,
costo USD y latencia; se sigue acumulando con cada uso de la app o del motor.

Manejo de versiones: cada fragmento citado incluye `version` (`"original"` para la Ley,
`"modificatoria"` para el DS) tomado de la metadata del documento — no hay artículos
duplicados en el corpus con dos versiones en conflicto porque el DS modifica el
*Reglamento* (fuera de alcance), no el texto de la Ley indexado; esta distinción se
documenta explícitamente para no sugerir falsamente que ambos documentos compiten por
la misma disposición.

### Evaluación y calibración de threshold (Fase 4)

Conjunto de evaluación real en `eval/eval_set.json`: **21 preguntas** (16 in-domain +
5 out-of-domain), redactadas a partir de contenido verificado de los documentos —
4 sobre artículos modificados por el DS (≥3 requerido), 5 en tono de pequeño
empresario (≥5 requerido), 7 preguntas generales sobre la Ley, y 5 fuera de dominio
(2 totalmente ajenas + 3 "cercanas" al dominio pero fuera de alcance). Ejecutar con:

```powershell
python eval/run_eval.py   # sin llamar al LLM; solo retrieval
```

**Recall@k** (`eval/results/recall_summary.csv`):

| Config | Recall@1 | Recall@3 | Recall@5 |
|---|---|---|---|
| `config_a` (96 tok) | 0.375 | 0.625 | **0.812** |
| `config_b` (120 tok, activa) | **0.438** | **0.750** | 0.750 |

**Calibración de threshold** (`eval/results/threshold_sweep.csv`): se subió de
0.55 (placeholder) a **0.60**, el borde de la meseta donde la abstención incorrecta
sobre las 16 preguntas in-domain reales sigue en 0%, con el mayor margen posible.

**Hallazgo real (limitación documentada, no oculta):** el threshold por similitud
separa bien preguntas totalmente ajenas (`"capital de Francia"` sim≈0.22 → se
abstiene) pero **no separa las 3 preguntas "cercanas al dominio"** (sobre el
Reglamento excluido, sobre Chile, sobre beneficios tributarios): tienen similitud
0.70–0.78, **más alta que varias preguntas in-domain reales**. Subir el threshold
para atraparlas sacrificaría 19–38% de las preguntas legítimas — peor trade-off.
Por eso el diseño no depende solo del threshold: `scope_note` viaja en *todas* las
respuestas y el `system_prompt` instruye al LLM a aclarar el alcance — el
retrieval es la primera línea de defensa, la generación es la segunda. Detalle
completo en `eval/results/eval_notes.md`.

**Comparación de embeddings (local vs. OpenAI), completa:**

```powershell
python eval/run_openai_comparison.py   # costo real: ~USD 0.004 en total
```

| Aspecto | Local (768 dim) | OpenAI `text-embedding-3-small` (1536 dim) |
|---|---|---|
| Recall@1 / @3 / @5 (`config_b`) | 0.438 / 0.750 / 0.750 | 0.375 / 0.750 / **0.875** |
| Recall@1 / @3 / @5 (`config_a`) | 0.375 / 0.625 / **0.812** | **0.500** / 0.625 / 0.688 |
| Tiempo de indexación (890/788 chunks) | 266.6s / 138.1s (CPU) | **14.6s / 8.9s** (API) |
| Costo real | USD 0.00 | USD 0.0021 / USD 0.0023 |
| Latencia por consulta | ~0.1–0.3s | 0.48–0.78s (red) |

**No hay ganador universal** (OpenAI mejor en Recall@5 con `config_b`, local mejor
en Recall@5 con `config_a` y en Recall@1 con `config_b`) — se mantiene el modelo
**local** como proveedor activo del motor porque el proyecto corre sin costo
variable por consulta; detalle completo y justificación en
`eval/results/eval_notes.md`.

`--stage extract` extrae el texto por página con PyMuPDF (preservando el número de
página impreso) y genera:

- `data/processed/extraction/<doc_id>.jsonl` — texto crudo por página
- `data/processed/reports/source_check.csv` — reporte de source check

`--stage clean` remueve los artefactos de maquetación de El Peruano (header repetido,
sello de firma digital, códigos de publicación OP, ligaduras tipográficas) y genera:

- `data/processed/clean/<doc_id>.jsonl` — texto limpio por página (con flag `is_cover_page`)
- `data/processed/reports/extraction_quality.csv` — reporte de calidad por documento
- `data/processed/reports/extraction_quality_notes.md` — limitaciones conocidas y decisiones

### Resultado del source check (Fase 1)

| Documento | Páginas | Caracteres totales | Prom. car/página | Mín. car/página | Máx. car/página | Páginas sin texto |
|---|---|---|---|---|---|---|
| Ley N.° 32069 | 36 | 316,158 | 8,782.2 | 948 | 13,087 | 0 |
| DS N.° 001-2026-EF | 16 | 134,755 | 8,422.2 | 8,031 | 9,144 | 0 |

Orden de lectura: se usa `page.get_text("text", sort=True)` de PyMuPDF, que ordena los
bloques de texto por posición vertical/horizontal antes de concatenarlos — necesario
porque las ediciones de El Peruano usan layout a una columna con encabezados repetidos.

Cada página trae pegado el header repetido de El Peruano (ej. `"El Peruano / Lunes 24 de
junio de 2024  NORMAS LEGALES  5"`), removido en la etapa de limpieza (ver abajo).

### Resultado de la limpieza (Fase 1)

| Documento | Páginas | Header removido | Sello firma removido | Códigos OP removidos | Ligaduras normalizadas | Portada excluida |
|---|---|---|---|---|---|---|
| Ley N.° 32069 | 36 | 35/36 (1 es portada) | 1 | 1 | 411 | página 1 |
| DS N.° 001-2026-EF | 16 | 16/16 | 1 | 2 | 0 | ninguna |

Limitaciones conocidas documentadas en `extraction_quality_notes.md`: espaciado
irregular alrededor de ligaduras tipográficas (ej. `ﬁ`) en <0.15% de los caracteres, y un
posible "bleed" de contenido de una norma vecina en la última página del DS (dos normas
distintas comparten la misma página física de la edición impresa) — se dejó marcado para
revisión manual en vez de recortarlo automáticamente, por seguridad de datos.

### Chunking (Fase 2a)

**Hallazgo clave:** el modelo de embeddings local configurado
(`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`) tiene
`max_seq_length = 128` tokens (verificado en su `sentence_bert_config.json` en Hugging
Face, no solo asumido). Los tamaños de chunk se fijaron **por debajo de ese límite**
(96 y 120 tokens) para que ningún chunk se trunque silenciosamente al generar su
embedding; `build_index.py --stage chunk` valida esto en cada corrida y falla explícito
si una configuración excede `max_tokens`.

El chunking se hace **por página** (nunca cruza el límite de una página) para que la
metadata de página de cada chunk sea siempre exacta y no ambigua, usando el tokenizer
real del modelo (offsets exactos, no una aproximación por caracteres) y excluyendo la
página de portada de la Ley. El chunk ID es determinístico
(`"{doc_id}:{config_name}:p{página}:c{índice}"`), por lo que re-ejecutar el build
produce exactamente los mismos IDs y texto (verificado con una segunda corrida).

| Config | chunk_size / overlap (tokens) | Chunks Ley 32069 | Chunks DS 001-2026-EF | Prom. tokens/chunk |
|---|---|---|---|---|
| `config_a` | 96 / 16 | 584 | 306 | ~93.8 |
| `config_b` | 120 / 30 | 516 | 272 | ~117.0 |

Comparación completa en `data/processed/reports/chunking_comparison.csv`. La
configuración activa por defecto es `config_b` (chunks más grandes, menos fragmentación,
más contexto por chunk); se compara contra `config_a` en la Fase 4 (evaluación de
Recall@k) para decidir cuál conservar.

## Cómo ejecutar — Tarea 2 (RAG Radar)

### Adquisición de datos (Fase 1)

```powershell
cd tarea2_radar
python acquire_data.py --stage bulk     # descarga masiva: 3 meses de 2026
python acquire_data.py --stage recent   # API de actualizaciones recientes (últimos 7 días)
python acquire_data.py --stage combine  # une ambos en una fila por proceso (por ocid)
python acquire_data.py --stage report   # resumen de tiempo/requests/tamaños
# o los 4 juntos:
python acquire_data.py --stage all
```

**Fuente real:** portal OECE (`contratacionesabiertas.oece.gob.pe`), estándar OCDS.
Verificado con `curl` antes de programar nada:

- **Descarga masiva** (`GET /api/v1/file/{source}/{type}/{year}/{month}`): sin
  autenticación, a diferencia de El Peruano (Tarea 1). Se descargaron 3 meses
  reales: **2026-06, 2026-07, 2026-08** (~30 MB comprimidos en total).
- **API de actualizaciones recientes** (`GET /api/v1/records?startDate=...&endDate=...`):
  filtra server-side por fecha de convocatoria (confirmado con curl comparando
  resultados con y sin el filtro). Se usa para traer los últimos 7 días sin
  esperar a que se publique el próximo archivo mensual.
- **Modelo OCDS**: cada `record` del `recordPackage` ya es la vista *compilada*
  (`compiledRelease`) de todas las `releases` (eventos: planificación,
  convocatoria, adjudicación...) de un mismo `ocid` — por eso se parte de
  `record.compiledRelease` en vez de reconstruirlo a mano desde `releases`. El
  `ocid` es el identificador único del proceso de contratación completo.

**Hallazgo real (bug evitado):** la API **ignora el parámetro `size`** — siempre
pagina de a 20 resultados sin importar qué tamaño de página se pida. Cortar la
paginación con `len(records) < size_pedido` truncaba en silencio el 98.5% de los
resultados (20 de 1,309 reales en la ventana de 7 días). Se corrigió siguiendo
`links.next` tal como lo devuelve la API en cada página, en vez de reconstruir el
offset manualmente. Verificado con curl antes y después del fix.

**Resultados reales:**

| Fuente | Procesos | Requests | Datos transferidos | Tiempo |
|---|---|---|---|---|
| Descarga masiva (3 meses) | 20,441 | 6 | 29.7 MB | 33.8 s |
| API actualizaciones recientes (7 días) | 1,309 | 67 (+66 desde caché en la 2ª corrida) | 6.0 MB | 41.5 s |
| **Combinado (una fila por `ocid`)** | **21,750** | | | |

Ambos mecanismos son **re-ejecutables sin duplicar**: `--stage bulk` no vuelve a
descargar un mes ya extraído en `data/raw/extracted/`; `--stage recent` cachea
cada página de la API en `data/raw/api_cache/` y una segunda corrida no genera
ningún request nuevo (verificado: 0 requests reales, 66/66 páginas leídas de
caché). Log real de tiempo/requests/tamaños en `logs/acquisition_log.csv` y
resumen agregado en `data/outputs/acquisition_summary.csv`.

Los ZIPs/JSON crudos (`data/raw/extracted/`, `data/raw/api_cache/`, ~290 MB) no
se versionan en git (se regeneran con el comando de arriba). El dataset
combinado intermedio (`processes.jsonl`/`.csv`, 21,750 filas) tampoco se
versiona porque queda superado por el dataset validado de la Fase 2 — ambos se
regeneran ejecutando `acquire_data.py` seguido de `validate_data.py`.

### Validación de calidad y normalización territorial (Fase 2)

```powershell
python validate_data.py
```

Detecta (sin corregir a ciegas) 6 reglas de calidad sobre las 21,750 filas
combinadas de la Fase 1:

| Regla | Casos | Acción tomada |
|---|---|---|
| Registros repetidos por el mismo proceso | **884** (827 grupos) | se conserva el más completo por grupo, resto descartado (auditoría en `duplicates_removed.csv`) |
| Proceso sin monto | 0 | ninguna |
| Proceso con monto cero | 3,419 | se conserva (válido en catálogos/convenios); documentado para no inflar promedios sin advertir |
| Proceso sin descripción | 0 | ninguna |
| Ubicación que no matchea un departamento | 0 | ninguna (`buyer.address.department` del OCDS ya viene curado por una extensión propia del OECE) |
| Inconsistencia de encoding/acentos | 0 | ninguna |

**Hallazgo real de duplicados:** se detectaron 827 grupos donde el mismo
comprador + misma nomenclatura + misma fecha de convocatoria aparecen bajo
**`ocid` distintos** — verificado en un caso concreto: dos ocids con el mismo
monto exacto (S/ 6,741,021.80) y mismo título, uno con `numberOfTenderers=null`
y otro con `numberOfTenderers=25` (evidencia de que SEACE V3 reemitió un id
interno nuevo para el mismo proceso real en una etapa posterior). Norma
aplicada: se conserva el registro con más campos poblados (heurística de
completitud: `numberOfTenderers` no nulo, tiene adjudicación, tiene proveedor
adjudicado), nunca se combinan/promedian silenciosamente.

**Normalización territorial:** `buyer.address.department` del estándar OCDS
peruano ya viene generado por una extensión propia (`ocds_department_extension`)
— de hecho, sobre 21,750 filas reales aparecieron **exactamente 25 valores
únicos**, los 25 departamentos oficiales, sin variantes de acento/mayúsculas ni
provincias mezcladas. La normalización (`normalize_department()` en
`src/validation.py`, comparación insensible a tildes/mayúsculas contra la lista
de `config.yaml`) igual se implementó y quedó lista para datos menos curados
(ej. si se agregan fuentes SEACE V2 a futuro), documentada con el reporte
`unmatched_locations.csv` que se generaría si hubiera algún caso.

Dataset final: `data/processed/processes_validated.jsonl`/`.csv`, **20,866
filas** (21,750 − 884 duplicados), con columnas nuevas `buyer_department`
(normalizado) e `is_encoding_issue`.

### RAG híbrido (Fase 3)

```powershell
python build_hybrid_index.py     # embebe título+descripción de los 20,866 procesos
python eval/run_hybrid_eval.py   # Recall@k sin LLM, sobre 12 preguntas conocidas
```

Reutiliza el modelo de embeddings local de la Tarea 1. Un embedding por proceso
(no por chunk, ya que la descripción cabe en ~50-70 tokens); **idempotente y
reanudable con checkpointing real cada 250 procesos** (no solo al final) —
necesario porque el build tardó ~31 minutos en esta máquina y el proceso en
segundo plano se interrumpió una vez por presión de memoria del sistema (7.9GB
RAM total); con el checkpointing, una interrupción ya no pierde el trabajo
previo.

`src/hybrid_engine.py` expone `answer_question(query, config, filters)`: los
filtros estructurados (`departamento`, `categoria`, `monto_min/max`,
`fecha_desde/hasta`) se aplican **siempre sobre metadata antes de tocar los
embeddings** — nunca se convierten en texto para buscar por similitud.

**Recall@k** (`eval/results/hybrid_retrieval_detail.csv`, 12 preguntas con
proceso relevante conocido): Recall@1=0.250, Recall@3=0.333, **Recall@5=0.417**
— sensiblemente más bajo que en la Tarea 1 (~0.75-0.81). Dos hallazgos reales,
diagnosticados a fondo (no solo la métrica):

1. **El lenguaje burocrático repetitivo confunde al embedding a esta escala.**
   Ejemplo real: la pregunta sobre "mejoramiento de agua potable en San
   Sebastián de Choropampa" (Cajamarca) recuperó un proceso *distinto*, de otra
   localidad de Cajamarca, con mayor similitud (0.772 vs. el correcto) —
   ambas descripciones son casi idénticas salvo el nombre del centro poblado,
   y el modelo no discrimina bien nombres de localidades pequeñas y poco
   frecuentes. **Se probó si el filtro de departamento lo arregla: no** — el
   proceso incorrecto también es de Cajamarca, así que el filtro reduce el
   universo (20,866 → 874) pero no alcanza a discriminar entre localidades
   dentro del mismo departamento.
2. **A esta escala, ningún threshold separa limpiamente in-domain de
   out-of-domain** (más severo que en la Tarea 1): la pregunta absurda
   *"¿cómo se prepara un ceviche peruano?"* obtiene similitud **0.696**, más
   alta que varias preguntas reales del dominio (ej. 0.466). El threshold se
   recalibró de 0.60 (Tarea 1) a **0.45** priorizando no abstenerse de
   preguntas reales, aceptando que aquí la defensa principal contra preguntas
   absurdas recae aún más en el LLM (citación por ocid, nunca en el
   threshold). Detalle completo en `eval/results/hybrid_eval_notes.md`.

**Nota sobre la API key:** al conectar el LLM para Fase 3 se detectó que la
key de OpenAI usada en la Tarea 1 ahora devuelve `401 invalid_api_key` — deja
de funcionar en algún punto entre tareas. El retrieval (Recall@k, que no
necesita LLM) funciona igual; la generación de respuesta queda pendiente de
una key válida.

### Indicador de riesgo — adjudicaciones monopostor (Fase 5)

```powershell
python compute_risk_indicator.py
```

Share de procesos adjudicados con exactamente 1 postor, por departamento y por
proveedor (`src/risk_indicator.py`). **Advertencia explícita, siempre visible
en el dashboard:** un solo postor no es evidencia de un delito — puede
reflejar un mercado con poca oferta o una contratación muy especializada; es
una señal estadística para priorizar revisión, nunca una acusación.

Top 5 departamentos por share real:

| Departamento | Share monopostor | Procesos (monopostor / adjudicados) |
|---|---|---|
| Tumbes | 36.7% | 44 / 120 |
| Lima | 30.2% | 1,175 / 3,886 |
| Amazonas | 16.2% | 35 / 216 |
| Arequipa | 14.2% | 78 / 548 |
| La Libertad | 10.3% | 50 / 486 |

**Mínimo de procesos justificado:** el Top 10 de proveedores solo considera
entidades con ≥5 procesos adjudicados en total (`risk_indicator.min_processes_per_entity`
en `config.yaml`) — con menos, un share de 100% no es estadísticamente
confiable (ej. 1 de 1 proceso). **Sin publicar nombres individuales:** los
proveedores que no calzan con un patrón de persona jurídica (sin "S.A.C.",
"S.R.L.", "CONSORCIO", etc. en el nombre) se anonimizan como
`"(persona natural #N — nombre no publicado)"` — de los primeros 5 del Top 10
real, 4 son personas naturales anonimizadas y 1 es una empresa
(`SISTEMAS ORACLE DEL PERÚ S.R.L.`, 100% de share en 8 procesos).

### Interfaz Streamlit (Fase 5)

```powershell
cd tarea1_rag_normativo
streamlit run app.py
```

Carga el índice ya construido (nunca lo reconstruye al iniciar). Tres pestañas:
**Consulta** (pregunta → respuesta + costo + latencia + fragmentos citados con
página/versión/similitud), **Calidad de datos** (tablas de `source_check.csv` y
`extraction_quality.csv` de la Fase 1), **Evaluación** (Recall@k, comparación de
embeddings y sweep de threshold de la Fase 4).

Probada en navegador real (no solo `streamlit run` sin verificar): la pregunta de
control *"Tengo una empresa pequeña, ¿hasta cuánto me pueden multar...?"* devolvió
la respuesta correcta citando el Art. 89.3, página 27, con costo real
USD 0.000185; la pregunta *"¿Cuál es la capital de Francia?"* se abstuvo
correctamente con costo USD 0.000000 y 0.51s de latencia (sin llamar al LLM).

## Pipelines

Ver [`docs/pipeline.md`](docs/pipeline.md) para los diagramas offline/online de ambas tareas.

## Resultados

_Pendiente — tabla de source check, comparación de embeddings, Recall@k, indicador de
riesgo monopostor y log de costos se agregan a medida que cada fase se completa._

## Checklist de avance

- [x] Estructura del repositorio y configuración base
- [x] Tarea 1 — Fase 1: fuentes, extracción y limpieza
- [x] Tarea 1 — Fase 2: chunking, embeddings e índice
- [x] Tarea 1 — Fase 3: motor RAG (threshold, versiones, scope, LLM conectado: OpenAI gpt-4o-mini)
- [x] Tarea 1 — Fase 4: evaluación (Recall@k, threshold, comparación local vs. OpenAI completa)
- [x] Tarea 1 — Fase 5: interfaz Streamlit
- [x] Tarea 2 — Fase 1: adquisición de datos
- [x] Tarea 2 — Fase 2: validación y normalización territorial
- [x] Tarea 2 — Fase 3: RAG híbrido y su evaluación
- [ ] Tarea 2 — Fase 4: dashboard Streamlit
- [x] Tarea 2 — Fase 5: indicador de riesgo monopostor
- [ ] Video de presentación
