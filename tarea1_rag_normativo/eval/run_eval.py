"""Evaluación de retrieval (Tarea 1, Fase 4) — SIN llamar al LLM.

Para cada configuración de chunking (config_a, config_b) y cada pregunta del
conjunto de evaluación (`eval_set.json`):
  - recupera el top-5 del índice FAISS ya construido,
  - calcula Recall@1/3/5 sobre las preguntas in-domain (doc_id + página esperados),
  - guarda la similitud top-1 de cada pregunta para el sweep de threshold.

Con esas similitudes, hace un sweep de thresholds para elegir el que mejor separa
preguntas in-domain (no deberían abstenerse) de out-of-domain (deberían abstenerse),
calibrando `retrieval.similarity_threshold` de config.yaml (Fase 3).

Uso:
    cd tarea1_rag_normativo
    python eval/run_eval.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import embed_texts
from src.indexing import load_index

TASK_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with (TASK_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_eval_set() -> list[dict]:
    with (TASK_ROOT / "eval" / "eval_set.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_top5(query: str, model_name: str, query_prefix: str, faiss_index, metadata):
    qvec = embed_texts([query], model_name=model_name, prefix=query_prefix)
    scores, idxs = faiss_index.search(qvec, 5)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        m = metadata[idx]
        results.append({"doc_id": m["doc_id"], "page_number": m["page_number"], "similarity": float(score)})
    return results


def evaluate_config(config: dict, config_name: str, eval_set: list[dict]) -> tuple[list[dict], dict]:
    index_dir = Path(config["paths"]["index_dir"]) / config_name
    faiss_index, ids, metadata = load_index(index_dir)

    active_provider = config["embeddings"]["active_provider"]
    provider_cfg = config["embeddings"]["providers"][active_provider]
    model_name = provider_cfg["model_name"]
    query_prefix = provider_cfg.get("query_prefix", "")

    detail_rows = []
    for q in eval_set:
        top5 = retrieve_top5(q["question"], model_name, query_prefix, faiss_index, metadata)
        top1_similarity = top5[0]["similarity"] if top5 else 0.0

        hit1 = hit3 = hit5 = None
        if q["expected_doc_id"] is not None:
            expected = (q["expected_doc_id"], q["expected_page"])
            retrieved_keys = [(r["doc_id"], r["page_number"]) for r in top5]
            hit1 = expected in retrieved_keys[:1]
            hit3 = expected in retrieved_keys[:3]
            hit5 = expected in retrieved_keys[:5]

        detail_rows.append(
            {
                "config_name": config_name,
                "question_id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "expected_doc_id": q["expected_doc_id"],
                "expected_page": q["expected_page"],
                "top1_doc_id": top5[0]["doc_id"] if top5 else None,
                "top1_page": top5[0]["page_number"] if top5 else None,
                "top1_similarity": round(top1_similarity, 4),
                "hit_at_1": hit1,
                "hit_at_3": hit3,
                "hit_at_5": hit5,
            }
        )

    in_domain_rows = [r for r in detail_rows if r["expected_doc_id"] is not None]
    n = len(in_domain_rows)
    recall = {
        "config_name": config_name,
        "n_in_domain": n,
        "n_out_domain": len(detail_rows) - n,
        "recall_at_1": round(sum(r["hit_at_1"] for r in in_domain_rows) / n, 3) if n else None,
        "recall_at_3": round(sum(r["hit_at_3"] for r in in_domain_rows) / n, 3) if n else None,
        "recall_at_5": round(sum(r["hit_at_5"] for r in in_domain_rows) / n, 3) if n else None,
    }
    return detail_rows, recall


def sweep_thresholds(detail_rows: list[dict], config_name: str) -> list[dict]:
    in_domain_sims = [r["top1_similarity"] for r in detail_rows if r["expected_doc_id"] is not None]
    out_domain_sims = [r["top1_similarity"] for r in detail_rows if r["expected_doc_id"] is None]

    rows = []
    threshold = 0.30
    while threshold <= 0.85:
        in_domain_incorrect_abstain = sum(1 for s in in_domain_sims if s < threshold) / len(in_domain_sims)
        out_domain_correct_abstain = (
            sum(1 for s in out_domain_sims if s < threshold) / len(out_domain_sims) if out_domain_sims else None
        )
        rows.append(
            {
                "config_name": config_name,
                "threshold": round(threshold, 3),
                "in_domain_incorrect_abstain_rate": round(in_domain_incorrect_abstain, 3),
                "out_domain_correct_abstain_rate": (
                    round(out_domain_correct_abstain, 3) if out_domain_correct_abstain is not None else None
                ),
            }
        )
        threshold += 0.025
    return rows


def main() -> None:
    config = load_config()
    eval_set = load_eval_set()
    results_dir = TASK_ROOT / "eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_detail_rows = []
    all_recall_rows = []
    all_sweep_rows = []

    for config_name in config["chunking"]["configs"]:
        detail_rows, recall_row = evaluate_config(config, config_name, eval_set)
        all_detail_rows.extend(detail_rows)
        all_recall_rows.append(recall_row)
        all_sweep_rows.extend(sweep_thresholds(detail_rows, config_name))

        print(
            f"[eval] {config_name}: Recall@1={recall_row['recall_at_1']} "
            f"Recall@3={recall_row['recall_at_3']} Recall@5={recall_row['recall_at_5']} "
            f"(n_in_domain={recall_row['n_in_domain']}, n_out_domain={recall_row['n_out_domain']})"
        )

    def write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(results_dir / "retrieval_detail.csv", all_detail_rows)
    write_csv(results_dir / "recall_summary.csv", all_recall_rows)
    write_csv(results_dir / "threshold_sweep.csv", all_sweep_rows)
    print(f"[eval] reportes guardados en {results_dir}")


if __name__ == "__main__":
    main()
