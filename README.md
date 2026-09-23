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

Este proyecto se desarrolla de forma incremental **sin requerir API keys de pago** en
las primeras fases (extracción, limpieza, chunking, embeddings locales, retrieval,
evaluación de Recall@k). Las claves solo son necesarias para:

| Variable | Cuándo se necesita |
|---|---|
| `ANTHROPIC_API_KEY` u `OPENAI_API_KEY` | Generación de la respuesta final del LLM (Tarea 1, Fase 3) |
| `OPENAI_EMBEDDINGS_API_KEY` | Comparación local vs. `text-embedding-3-small` (Tarea 1, Fase 4) |
| `OECE_API_KEY` | Solo si el portal OECE introduce autenticación (hoy no la requiere) |

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
```

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

_Pendiente — se documenta al cerrar Fase 1 (adquisición de datos)._

## Pipelines

Ver [`docs/pipeline.md`](docs/pipeline.md) para los diagramas offline/online de ambas tareas.

## Resultados

_Pendiente — tabla de source check, comparación de embeddings, Recall@k, indicador de
riesgo monopostor y log de costos se agregan a medida que cada fase se completa._

## Checklist de avance

- [x] Estructura del repositorio y configuración base
- [x] Tarea 1 — Fase 1: fuentes, extracción y limpieza
- [x] Tarea 1 — Fase 2a: chunking (falta embeddings e índice)
- [ ] Tarea 1 — Fase 3: motor RAG (threshold, versiones, scope)
- [ ] Tarea 1 — Fase 4: evaluación y comparación de embeddings
- [ ] Tarea 1 — Fase 5: interfaz Streamlit
- [ ] Tarea 2 — Fase 1: adquisición de datos
- [ ] Tarea 2 — Fase 2: validación y normalización territorial
- [ ] Tarea 2 — Fase 3: RAG híbrido
- [ ] Tarea 2 — Fase 4: dashboard Streamlit
- [ ] Tarea 2 — Fase 5: indicador de riesgo monopostor
- [ ] Video de presentación
