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

# 2. Crear y activar entorno virtual
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

_Pendiente — se documenta al cerrar Fase 1/2 (extracción + índice)._

## Cómo ejecutar — Tarea 2 (RAG Radar)

_Pendiente — se documenta al cerrar Fase 1 (adquisición de datos)._

## Pipelines

Ver [`docs/pipeline.md`](docs/pipeline.md) para los diagramas offline/online de ambas tareas.

## Resultados

_Pendiente — tabla de source check, comparación de embeddings, Recall@k, indicador de
riesgo monopostor y log de costos se agregan a medida que cada fase se completa._

## Checklist de avance

- [x] Estructura del repositorio y configuración base
- [ ] Tarea 1 — Fase 1: fuentes, extracción y limpieza
- [ ] Tarea 1 — Fase 2: chunking, embeddings e índice
- [ ] Tarea 1 — Fase 3: motor RAG (threshold, versiones, scope)
- [ ] Tarea 1 — Fase 4: evaluación y comparación de embeddings
- [ ] Tarea 1 — Fase 5: interfaz Streamlit
- [ ] Tarea 2 — Fase 1: adquisición de datos
- [ ] Tarea 2 — Fase 2: validación y normalización territorial
- [ ] Tarea 2 — Fase 3: RAG híbrido
- [ ] Tarea 2 — Fase 4: dashboard Streamlit
- [ ] Tarea 2 — Fase 5: indicador de riesgo monopostor
- [ ] Video de presentación
