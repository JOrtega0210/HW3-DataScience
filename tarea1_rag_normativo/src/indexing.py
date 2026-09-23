"""Construcción y carga del índice vectorial (FAISS), idempotente y reanudable.

Tarea 1, Fase 2 / Tarea 2, Fase 3. Sin dependencias de UI: solo numpy/faiss + el
módulo de embeddings.

Idempotencia/resumabilidad: los embeddings ya calculados se guardan por chunk_id en
disco (`ids.json` + `embeddings.npy`). Al reconstruir, solo se embeben los chunk_ids
nuevos; si el proceso se interrumpe a mitad de camino, la siguiente corrida retoma
desde los que ya quedaron guardados en vez de recalcular todo el corpus. Los chunk_ids
que ya no existen (ej. tras editar manualmente un chunk) se descartan del índice.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.embeddings import embed_texts


def _load_existing(index_dir: Path) -> tuple[list[str], np.ndarray | None]:
    ids_path = index_dir / "ids.json"
    emb_path = index_dir / "embeddings.npy"
    if ids_path.exists() and emb_path.exists():
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        embeddings = np.load(emb_path)
        return ids, embeddings
    return [], None


def build_or_update_index(
    chunks: list[dict],
    index_dir: Path,
    model_name: str,
    passage_prefix: str = "",
    batch_size: int = 32,
) -> dict:
    """`chunks`: lista de dicts con al menos chunk_id y text (más metadata a preservar)."""
    index_dir.mkdir(parents=True, exist_ok=True)
    existing_ids, existing_embeddings = _load_existing(index_dir)
    existing_pos = {cid: i for i, cid in enumerate(existing_ids)}

    current_set = {c["chunk_id"] for c in chunks}

    # Conservar embeddings de chunk_ids que siguen vigentes; podar los que ya no existen.
    kept_ids = [cid for cid in existing_ids if cid in current_set]
    kept_embeddings = (
        existing_embeddings[[existing_pos[cid] for cid in kept_ids]]
        if existing_embeddings is not None and kept_ids
        else None
    )

    kept_set = set(kept_ids)
    new_chunks = [c for c in chunks if c["chunk_id"] not in kept_set]

    new_embeddings = embed_texts(
        [c["text"] for c in new_chunks],
        model_name=model_name,
        prefix=passage_prefix,
        batch_size=batch_size,
    )

    if kept_embeddings is not None and len(kept_embeddings):
        all_embeddings = (
            np.vstack([kept_embeddings, new_embeddings]) if len(new_embeddings) else kept_embeddings
        )
    else:
        all_embeddings = new_embeddings

    all_ids = kept_ids + [c["chunk_id"] for c in new_chunks]

    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    ordered_metadata = [chunk_by_id[cid] for cid in all_ids]

    np.save(index_dir / "embeddings.npy", all_embeddings)
    (index_dir / "ids.json").write_text(json.dumps(all_ids, ensure_ascii=False), encoding="utf-8")
    with (index_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for m in ordered_metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    import faiss

    dim = all_embeddings.shape[1] if len(all_embeddings) else embed_texts([], model_name).shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    if len(all_embeddings):
        faiss_index.add(all_embeddings)
    faiss.write_index(faiss_index, str(index_dir / "faiss.index"))

    return {
        "total_chunks": len(all_ids),
        "reused_embeddings": len(kept_ids),
        "new_embeddings": len(new_chunks),
    }


def load_index(index_dir: Path):
    """Carga el índice ya construido (usado por el motor RAG online, sin recalcular nada)."""
    import faiss

    faiss_index = faiss.read_index(str(index_dir / "faiss.index"))
    ids = json.loads((index_dir / "ids.json").read_text(encoding="utf-8"))
    metadata = []
    with (index_dir / "metadata.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))
    return faiss_index, ids, metadata
