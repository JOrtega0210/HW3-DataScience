# Pipelines

Diagramas de los pipelines offline/online de ambas tareas. Se completan a medida que se
implementa cada fase; sirven de base para la sección "Pipeline Task 1/2" del video (regla
pipeline-primero).

## Tarea 1 — RAG Normativo

```mermaid
flowchart LR
    subgraph Offline["Offline: indexación (build_index.py)"]
        A[PDFs oficiales] --> B[Extracción con page numbers]
        B --> C[Limpieza de headers]
        C --> D[Chunking configs A/B]
        D --> E[Embeddings locales]
        E --> F[(Índice vectorial)]
    end
    subgraph Online["Online: consultas (app.py)"]
        G[Pregunta usuario] --> H[Embedding de la query]
        H --> I{similitud >= threshold?}
        I -- no --> J[Abstención]
        I -- sí --> K[Top-k fragmentos + metadata]
        K --> L[LLM genera respuesta citada]
    end
    F --> I
```

_Pendiente: completar con decisiones reales (threshold calibrado, tamaños de chunk elegidos)
una vez cerradas las Fases 2-3._

## Tarea 2 — RAG Radar

```mermaid
flowchart LR
    subgraph Offline["Offline: adquisición + preparación"]
        A2[Descargas OECE + API] --> B2[Validación y normalización territorial]
        B2 --> C2[Una fila por proceso]
        C2 --> D2[Embeddings locales de descripciones]
        D2 --> E2[(Índice vectorial + metadata)]
        C2 --> F2[Indicador monopostor]
    end
    subgraph Online["Online: dashboard (app.py)"]
        G2[Filtros sidebar] --> H2[Tabla / mapa / distribución]
        I2[Pregunta en lenguaje natural] --> J2[Embedding + filtros estructurados]
        J2 --> K2{similitud >= threshold?}
        K2 -- no --> L2[Abstención]
        K2 -- sí --> M2[Procesos recuperados citados por ocid]
    end
    E2 --> K2
    F2 --> G2
```

_Pendiente: completar con métricas reales de Recall@k y resultado del indicador de riesgo._
