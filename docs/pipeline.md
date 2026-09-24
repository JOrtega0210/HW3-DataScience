# Pipelines

Diagramas de los flujos offline (indexación/preparación de datos) y online
(consultas) de ambas tareas.

## Tarea 1 — RAG Normativo

```mermaid
flowchart LR
    subgraph Offline["Offline: indexación (build_index.py)"]
        A[PDFs oficiales] --> B[Extracción con page numbers]
        B --> C[Limpieza de headers]
        C --> D["Chunking: 96 / 120 tokens"]
        D --> E[Embeddings locales]
        E --> F[(Índice vectorial FAISS)]
    end
    subgraph Online["Online: consultas (app.py)"]
        G[Pregunta del usuario] --> H[Embedding de la consulta]
        H --> I{"similitud >= 0.60?"}
        I -- no --> J[Abstención]
        I -- sí --> K[Top-5 fragmentos + metadata]
        K --> L[LLM genera respuesta citada]
    end
    F --> I
```

Threshold de abstención calibrado en 0.60; configuración de chunking activa
de 120 tokens con overlap de 30 (ver README, sección Tarea 1).

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
        G2[Filtros del sidebar] --> H2[Tabla / mapa / distribución]
        I2[Pregunta en lenguaje natural] --> J2[Filtros estructurados + embedding]
        J2 --> K2{"similitud >= 0.45?"}
        K2 -- no --> L2[Abstención]
        K2 -- sí --> M2[Procesos recuperados citados por ocid]
    end
    E2 --> K2
    F2 --> G2
```

Threshold de abstención calibrado en 0.45 (ver README, sección Tarea 2, para
la justificación de por qué el threshold de la Tarea 1 no transfiere a esta
escala).
