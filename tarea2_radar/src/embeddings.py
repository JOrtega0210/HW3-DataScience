"""Embeddings locales, reutilizando el mismo modelo calibrado en la Tarea 1
(sentence-transformers/paraphrase-multilingual-mpnet-base-v2). Tarea 2, Fase 3.

Sin dependencias de UI: solo sentence-transformers + numpy.
"""

from __future__ import annotations

import numpy as np

_model_cache: dict = {}


def get_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed_texts(texts: list[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    """Embeddings normalizados L2 (similitud coseno = producto interno)."""
    model = get_embedding_model(model_name)
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")
