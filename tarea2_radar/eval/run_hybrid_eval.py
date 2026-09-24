"""Evaluación del RAG híbrido (Tarea 2, Fase 3) — SIN llamar al LLM.

Para cada pregunta de `hybrid_eval_set.json`: aplica los filtros estructurados
(si tiene), recupera el top-5 semántico dentro de los candidatos filtrados, y
calcula Recall@1/3/5 contra el `expected_ocid` conocido. También hace un sweep
de threshold para verificar si el calibrado en la Tarea 1 (0.60) transfiere a
este dominio o hay que recalibrar.

Uso:
    cd tarea2_radar
    python eval/run_hybrid_eval.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import embed_texts
from src.hybrid_engine import StructuredFilters, passes_filters
from src.hybrid_index import load_index

TASK_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with (TASK_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_eval_set() -> list[dict]:
    with (TASK_ROOT / "eval" / "hybrid_eval_set.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    config = load_config()
    eval_set = load_eval_set()

    index_dir = Path(config["paths"]["index_dir"])
    embeddings, ids, metadata = load_index(index_dir)
    model_name = config["embeddings"]["model_name"]

    results_dir = TASK_ROOT / "eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    for q in eval_set:
        filters = StructuredFilters(**q.get("filters", {}))
        candidate_idx = [i for i, m in enumerate(metadata) if passes_filters(m, filters)]

        qvec = embed_texts([q["question"]], model_name=model_name)[0]
        candidate_embeddings = embeddings[candidate_idx]
        sims = candidate_embeddings @ qvec
        order = np.argsort(-sims)[:5]
        top5_ocids = [ids[candidate_idx[pos]] for pos in order]
        top1_similarity = float(sims[order[0]]) if len(order) else 0.0

        detail_rows.append(
            {
                "question_id": q["id"],
                "question": q["question"],
                "filters": json.dumps(q.get("filters", {}), ensure_ascii=False),
                "n_candidates_after_filter": len(candidate_idx),
                "expected_ocid": q["expected_ocid"],
                "top1_ocid": top5_ocids[0] if top5_ocids else None,
                "top1_similarity": round(top1_similarity, 4),
                "hit_at_1": q["expected_ocid"] in top5_ocids[:1],
                "hit_at_3": q["expected_ocid"] in top5_ocids[:3],
                "hit_at_5": q["expected_ocid"] in top5_ocids[:5],
            }
        )

    n = len(detail_rows)
    recall_1 = sum(r["hit_at_1"] for r in detail_rows) / n
    recall_3 = sum(r["hit_at_3"] for r in detail_rows) / n
    recall_5 = sum(r["hit_at_5"] for r in detail_rows) / n
    print(f"[hybrid-eval] n={n} Recall@1={recall_1:.3f} Recall@3={recall_3:.3f} Recall@5={recall_5:.3f}")

    with (results_dir / "hybrid_retrieval_detail.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    # Sweep de threshold sobre las similitudes top-1 observadas (todas in-domain aquí,
    # ya que las 12 preguntas tienen un proceso relevante conocido).
    sims_observed = [r["top1_similarity"] for r in detail_rows]
    threshold_cfg = config["retrieval"]["similarity_threshold"]
    sweep_rows = []
    t = 0.30
    while t <= 0.85:
        incorrect_abstain = sum(1 for s in sims_observed if s < t) / n
        sweep_rows.append({"threshold": round(t, 3), "in_domain_incorrect_abstain_rate": round(incorrect_abstain, 3)})
        t += 0.025
    with (results_dir / "hybrid_threshold_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "in_domain_incorrect_abstain_rate"])
        writer.writeheader()
        writer.writerows(sweep_rows)

    at_config_threshold = next(r for r in sweep_rows if abs(r["threshold"] - round(threshold_cfg, 3)) < 0.0126)
    print(
        f"[hybrid-eval] threshold actual de config.yaml={threshold_cfg}: "
        f"abstención incorrecta={at_config_threshold['in_domain_incorrect_abstain_rate']:.3f} "
        f"(¿transfiere bien de Tarea 1? {'sí' if at_config_threshold['in_domain_incorrect_abstain_rate'] == 0 else 'NO, revisar'})"
    )
    print(f"[hybrid-eval] reportes guardados en {results_dir}")


if __name__ == "__main__":
    main()
