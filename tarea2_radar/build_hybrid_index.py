"""Construcción del índice del RAG híbrido (Tarea 2, Fase 3).

Uso:
    python build_hybrid_index.py

Embebe título+descripción de cada proceso de `processes_validated.jsonl` con el
modelo local de la Tarea 1. Idempotente y reanudable: si se interrumpe, la
siguiente corrida retoma desde los `ocid` ya embebidos guardados en disco.
"""

from __future__ import annotations

import os

# Evita un cuelgue clásico de torch/numpy en Windows cuando hay dos runtimes
# OpenMP cargados (MKL de numpy + el de torch) — debe fijarse ANTES de importar
# nada de sentence-transformers/torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import time
from pathlib import Path

import yaml

from src.hybrid_index import build_or_update_index


def load_config(path: Path = Path("config.yaml")) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    import json

    config = load_config()
    processed_dir = Path(config["paths"]["processed_dir"])
    index_dir = Path(config["paths"]["index_dir"])

    in_path = processed_dir / "processes_validated.jsonl"
    with in_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    print(f"[hybrid-index] {len(rows)} procesos a indexar desde {in_path}", flush=True)

    model_name = config["embeddings"]["model_name"]

    def report_progress(done: int, total: int) -> None:
        print(f"[hybrid-index] checkpoint: {done}/{total} procesos embebidos ({time.strftime('%H:%M:%S')})", flush=True)

    t0 = time.time()
    stats = build_or_update_index(
        rows, index_dir, model_name=model_name, batch_size=16,
        checkpoint_every=250, progress_callback=report_progress,
    )
    elapsed = time.time() - t0

    print(
        f"[hybrid-index] {stats['new_embeddings']} nuevos, "
        f"{stats['reused_embeddings']} reusados, {stats['total_processes']} total "
        f"({elapsed:.1f}s)"
    )


if __name__ == "__main__":
    main()
