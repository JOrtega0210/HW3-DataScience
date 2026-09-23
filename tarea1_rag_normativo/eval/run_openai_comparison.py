"""Comparación real local vs. OpenAI text-embedding-3-small (Tarea 1, Fase 4).

Para cada configuración de chunking: embebe todos los chunks con la API de OpenAI,
construye un índice FAISS temporal (no se persiste como índice de producción — el
motor online sigue usando el modelo local, ver config.yaml `embeddings.active_provider`),
mide tiempo/costo real, y corre el mismo eval_set.json para obtener Recall@k
comparable con la corrida local ya guardada en `eval/results/recall_summary.csv`.

Uso:
    cd tarea1_rag_normativo
    python eval/run_openai_comparison.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TASK_ROOT = Path(__file__).resolve().parent.parent
load_dotenv()


def load_config() -> dict:
    with (TASK_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_eval_set() -> list[dict]:
    with (TASK_ROOT / "eval" / "eval_set.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def embed_openai(texts: list[str], model_name: str, api_key: str, batch_size: int = 100):
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    all_embeddings = []
    total_tokens = 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model_name, input=batch)
        all_embeddings.extend(d.embedding for d in resp.data)
        total_tokens += resp.usage.total_tokens
    arr = np.array(all_embeddings, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / norms  # normalizar L2 para que el producto interno = coseno
    return arr, total_tokens


def build_faiss(embeddings: np.ndarray):
    import faiss

    index = faiss.IndexFlatIP(embeddings.shape[1])
    if len(embeddings):
        index.add(embeddings)
    return index


def evaluate(index, metadata: list[dict], eval_set: list[dict], model_name: str, api_key: str) -> list[dict]:
    rows = []
    for q in eval_set:
        qvec, _ = embed_openai([q["question"]], model_name, api_key, batch_size=1)
        scores, idxs = index.search(qvec, 5)
        top5 = [metadata[idx] for idx in idxs[0] if idx >= 0]

        hit1 = hit3 = hit5 = None
        if q["expected_doc_id"] is not None:
            expected = (q["expected_doc_id"], q["expected_page"])
            keys = [(m["doc_id"], m["page_number"]) for m in top5]
            hit1 = expected in keys[:1]
            hit3 = expected in keys[:3]
            hit5 = expected in keys[:5]

        rows.append(
            {
                "question_id": q["id"],
                "expected_doc_id": q["expected_doc_id"],
                "hit_at_1": hit1,
                "hit_at_3": hit3,
                "hit_at_5": hit5,
            }
        )
    return rows


def main() -> None:
    config = load_config()
    eval_set = load_eval_set()
    api_key = os.environ.get("OPENAI_EMBEDDINGS_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Falta OPENAI_EMBEDDINGS_API_KEY / OPENAI_API_KEY en .env")

    openai_cfg = config["embeddings"]["providers"]["openai"]
    model_name = openai_cfg["model_name"]
    price_per_1m = config["pricing"]["embeddings_openai_per_1m_usd"]

    results_dir = TASK_ROOT / "eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for config_name in config["chunking"]["configs"]:
        chunks_dir = TASK_ROOT / "data" / "processed" / "chunks" / config_name
        chunks: list[dict] = []
        for doc in config["documents"]:
            path = chunks_dir / f"{doc['id']}.jsonl"
            with path.open("r", encoding="utf-8") as f:
                chunks.extend(json.loads(line) for line in f)

        t0 = time.time()
        embeddings, total_tokens = embed_openai([c["text"] for c in chunks], model_name, api_key)
        index_seconds = time.time() - t0
        index = build_faiss(embeddings)

        t0 = time.time()
        detail = evaluate(index, chunks, eval_set, model_name, api_key)
        eval_seconds = time.time() - t0
        n_queries = len(eval_set)

        in_domain = [r for r in detail if r["expected_doc_id"] is not None]
        n = len(in_domain)
        cost_usd = total_tokens * price_per_1m / 1_000_000

        row = {
            "config_name": config_name,
            "provider": "openai",
            "model_name": model_name,
            "dimension": embeddings.shape[1],
            "total_chunks": len(chunks),
            "total_tokens_billed": total_tokens,
            "index_build_seconds": round(index_seconds, 2),
            "avg_query_latency_seconds": round(eval_seconds / n_queries, 3),
            "cost_usd": round(cost_usd, 6),
            "recall_at_1": round(sum(r["hit_at_1"] for r in in_domain) / n, 3) if n else None,
            "recall_at_3": round(sum(r["hit_at_3"] for r in in_domain) / n, 3) if n else None,
            "recall_at_5": round(sum(r["hit_at_5"] for r in in_domain) / n, 3) if n else None,
        }
        summary_rows.append(row)
        print(
            f"[openai-compare] {config_name}: Recall@1={row['recall_at_1']} "
            f"Recall@3={row['recall_at_3']} Recall@5={row['recall_at_5']} "
            f"costo=${row['cost_usd']} tokens={row['total_tokens_billed']} "
            f"build={row['index_build_seconds']}s"
        )

    report_path = results_dir / "openai_embeddings_comparison.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[openai-compare] reporte guardado en {report_path}")


if __name__ == "__main__":
    main()
