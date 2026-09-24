"""Construcción del índice del RAG híbrido (Tarea 2, Fase 3): un embedding por
proceso de contratación (título + descripción), idempotente y reanudable —
igual que el índice de la Tarea 1, pero clave por `ocid` en vez de `chunk_id`.

Metadata guardada para poder filtrar SIEMPRE por condiciones estructuradas
(departamento, monto, fecha, categoría) antes de tocar los embeddings: esas
condiciones nunca se embeben, solo se usan para filtrar el universo de
candidatos antes de calcular similitud coseno.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.embeddings import embed_texts

METADATA_FIELDS = [
    "ocid",
    "buyer_name",
    "buyer_department",
    "amount",
    "currency",
    "date_published",
    "main_procurement_category",
    "tender_title",
    "tender_description",
    "num_awards",
    "number_of_tenderers",
    "is_single_bidder_awarded",
]


def _embedding_text(row: dict) -> str:
    title = row.get("tender_title") or ""
    desc = row.get("tender_description") or ""
    return f"{title}. {desc}".strip()


def _load_existing(index_dir: Path) -> tuple[list[str], np.ndarray | None]:
    ids_path = index_dir / "ids.json"
    emb_path = index_dir / "embeddings.npy"
    if ids_path.exists() and emb_path.exists():
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        embeddings = np.load(emb_path)
        return ids, embeddings
    return [], None


def _save(index_dir: Path, all_ids: list[str], all_embeddings: np.ndarray, row_by_ocid: dict) -> None:
    ordered_metadata = [
        {k: row_by_ocid[oid].get(k) for k in METADATA_FIELDS} for oid in all_ids
    ]
    np.save(index_dir / "embeddings.npy", all_embeddings)
    (index_dir / "ids.json").write_text(json.dumps(all_ids, ensure_ascii=False), encoding="utf-8")
    with (index_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for m in ordered_metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def build_or_update_index(
    rows: list[dict],
    index_dir: Path,
    model_name: str,
    batch_size: int = 64,
    checkpoint_every: int = 500,
    progress_callback=None,
) -> dict:
    """Idempotente/reanudable de verdad: los `ocid` nuevos se embeben en lotes
    de `checkpoint_every` y se guardan en disco después de CADA lote (no solo
    al final) — si el proceso se interrumpe a mitad de camino, la siguiente
    corrida retoma desde el último lote guardado en vez de perder todo el
    trabajo. Los `ocid` que ya no están en `rows` se podan del índice.
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    existing_ids, existing_embeddings = _load_existing(index_dir)
    existing_pos = {oid: i for i, oid in enumerate(existing_ids)}

    current_set = {r["ocid"] for r in rows}
    kept_ids = [oid for oid in existing_ids if oid in current_set]
    kept_embeddings = (
        existing_embeddings[[existing_pos[oid] for oid in kept_ids]]
        if existing_embeddings is not None and kept_ids
        else None
    )

    kept_set = set(kept_ids)
    new_rows = [r for r in rows if r["ocid"] not in kept_set]

    row_by_ocid = {r["ocid"]: r for r in rows}
    all_ids = list(kept_ids)
    all_embeddings = kept_embeddings

    for start in range(0, len(new_rows), checkpoint_every):
        batch = new_rows[start : start + checkpoint_every]
        batch_embeddings = embed_texts(
            [_embedding_text(r) for r in batch], model_name=model_name, batch_size=batch_size
        )
        if all_embeddings is not None and len(all_embeddings):
            all_embeddings = np.vstack([all_embeddings, batch_embeddings])
        else:
            all_embeddings = batch_embeddings
        all_ids.extend(r["ocid"] for r in batch)

        _save(index_dir, all_ids, all_embeddings, row_by_ocid)
        if progress_callback:
            progress_callback(len(all_ids), len(kept_ids) + len(new_rows))

    if not new_rows and all_embeddings is not None:
        # Nada nuevo que embeber, pero igual se guarda por si se podó algo.
        _save(index_dir, all_ids, all_embeddings, row_by_ocid)

    return {
        "total_processes": len(all_ids),
        "reused_embeddings": len(kept_ids),
        "new_embeddings": len(new_rows),
    }


def load_index(index_dir: Path) -> tuple[np.ndarray, list[str], list[dict]]:
    embeddings = np.load(index_dir / "embeddings.npy")
    ids = json.loads((index_dir / "ids.json").read_text(encoding="utf-8"))
    metadata = []
    with (index_dir / "metadata.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))
    return embeddings, ids, metadata
