"""Pipeline offline de indexación (Tarea 1).

Uso:
    python build_index.py

Pasos (se implementan por fases, ver docs/pipeline.md):
    1. Extracción de texto desde data/raw/*.pdf preservando número de página.
    2. Limpieza de headers de El Peruano.
    3. Chunking (config activa en config.yaml -> chunking.active).
    4. Embeddings locales + construcción de índice idempotente/reanudable.
"""

if __name__ == "__main__":
    raise NotImplementedError(
        "Pendiente: se implementa en las Fases 1-2 de la Tarea 1."
    )
