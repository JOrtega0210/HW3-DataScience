"""Motor RAG híbrido (Tarea 2, Fase 3).

Interfaz única expuesta por una función: `answer_question()`. Sin dependencias
de UI. Condiciones estructuradas (departamento, monto, fecha, categoría)
SIEMPRE se aplican como filtros de metadata ANTES de tocar los embeddings —
nunca se convierten en texto para buscar por similitud, tal como exige el
enunciado ("condiciones numéricas/territoriales como filtros, no embeddings").

Reutiliza el modelo de embeddings local calibrado en la Tarea 1. El threshold
de abstención parte del de la Tarea 1 y se recalibra en Fase 3 si no transfiere
bien a este dominio (ver eval/run_hybrid_eval.py).
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.embeddings import embed_texts
from src.hybrid_index import load_index

load_dotenv()  # busca .env en el cwd y directorios padre (ej. raíz del repo)


@dataclass
class StructuredFilters:
    departamento: str | None = None
    categoria: str | None = None  # goods / works / services
    monto_min: float | None = None
    monto_max: float | None = None
    fecha_desde: str | None = None  # ISO date, compara contra date_published
    fecha_hasta: str | None = None


@dataclass
class RetrievedProcess:
    ocid: str
    buyer_name: str
    buyer_department: str | None
    amount: float | None
    currency: str | None
    date_published: str | None
    main_procurement_category: str | None
    tender_title: str
    similarity: float


@dataclass
class HybridResult:
    query: str
    filters_applied: dict
    n_candidates_after_filter: int
    abstained: bool
    abstain_reason: str | None
    retrieved: list[RetrievedProcess]
    answer: str | None
    llm_error: str | None
    model_name: str | None
    input_tokens: int | None
    output_tokens: int | None
    usd_cost: float | None
    latency_seconds: float


def passes_filters(meta: dict, filters: StructuredFilters) -> bool:
    if filters.departamento and meta.get("buyer_department") != filters.departamento:
        return False
    if filters.categoria and meta.get("main_procurement_category") != filters.categoria:
        return False
    amount = meta.get("amount")
    if filters.monto_min is not None and (amount is None or amount < filters.monto_min):
        return False
    if filters.monto_max is not None and (amount is None or amount > filters.monto_max):
        return False
    date_published = meta.get("date_published")
    if filters.fecha_desde and (not date_published or date_published < filters.fecha_desde):
        return False
    if filters.fecha_hasta and (not date_published or date_published > filters.fecha_hasta):
        return False
    return True


class HybridRagEngine:
    def __init__(self, config: dict):
        self.config = config
        index_dir = Path(config["paths"]["index_dir"])
        self.embeddings, self.ids, self.metadata = load_index(index_dir)
        self.model_name = config["embeddings"]["model_name"]
        self.top_k = config["retrieval"]["top_k"]
        self.threshold = config["retrieval"]["similarity_threshold"]
        self.costs_log_path = Path(config["logging"]["costs_log_file"])

    def _retrieve(self, query: str, filters: StructuredFilters) -> tuple[list[RetrievedProcess], int]:
        candidate_idx = [i for i, m in enumerate(self.metadata) if passes_filters(m, filters)]
        if not candidate_idx:
            return [], 0

        qvec = embed_texts([query], model_name=self.model_name)[0]
        candidate_embeddings = self.embeddings[candidate_idx]
        sims = candidate_embeddings @ qvec  # coseno, porque ya están normalizados L2

        order = np.argsort(-sims)[: self.top_k]
        results = []
        for pos in order:
            global_idx = candidate_idx[pos]
            m = self.metadata[global_idx]
            results.append(
                RetrievedProcess(
                    ocid=m["ocid"],
                    buyer_name=m.get("buyer_name"),
                    buyer_department=m.get("buyer_department"),
                    amount=m.get("amount"),
                    currency=m.get("currency"),
                    date_published=m.get("date_published"),
                    main_procurement_category=m.get("main_procurement_category"),
                    tender_title=m.get("tender_title"),
                    similarity=float(sims[pos]),
                )
            )
        return results, len(candidate_idx)

    def _generate_answer(
        self, query: str, retrieved: list[RetrievedProcess]
    ) -> tuple[str | None, str | None, str, int | None, int | None, float | None]:
        llm_cfg = self.config["llm"]
        model_name = llm_cfg["model_name"]
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return (
                None,
                "Sin OPENAI_API_KEY configurada en .env: se devuelven los procesos "
                "recuperados pero no se genera respuesta de LLM.",
                model_name,
                None,
                None,
                None,
            )

        context = "\n\n".join(
            f"[ocid: {r.ocid} | {r.buyer_name} | {r.buyer_department} | "
            f"{r.currency} {r.amount} | {r.date_published} | similitud {r.similarity:.3f}]\n"
            f"{r.tender_title}"
            for r in retrieved
        )
        user_prompt = f"Procesos recuperados:\n\n{context}\n\nPregunta: {query}"

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name,
                temperature=llm_cfg["temperature"],
                max_tokens=llm_cfg["max_tokens"],
                messages=[
                    {"role": "system", "content": llm_cfg["system_prompt"]},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - error de API como dato, no excepción
            return (None, f"Error llamando a la API de OpenAI ({model_name}): {exc}", model_name, None, None, None)

        answer = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        pricing = self.config["pricing"]
        usd_cost = (
            input_tokens * pricing["llm_input_per_1m_usd"] / 1_000_000
            + output_tokens * pricing["llm_output_per_1m_usd"] / 1_000_000
        )
        return (answer, None, model_name, input_tokens, output_tokens, usd_cost)

    def _log_cost(self, result: HybridResult) -> None:
        self.costs_log_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.costs_log_path.exists()
        with self.costs_log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(
                    ["timestamp", "query", "abstained", "model_name", "input_tokens",
                     "output_tokens", "usd_cost", "latency_seconds"]
                )
            writer.writerow(
                [
                    time.strftime("%Y-%m-%dT%H:%M:%S"), result.query, result.abstained,
                    result.model_name, result.input_tokens, result.output_tokens,
                    result.usd_cost, round(result.latency_seconds, 3),
                ]
            )

    def answer_question(
        self,
        query: str,
        filters: StructuredFilters | None = None,
        threshold_override: float | None = None,
    ) -> HybridResult:
        filters = filters or StructuredFilters()
        threshold = threshold_override if threshold_override is not None else self.threshold
        t0 = time.time()
        retrieved, n_candidates = self._retrieve(query, filters)
        top_similarity = retrieved[0].similarity if retrieved else 0.0

        filters_dict = {k: v for k, v in vars(filters).items() if v is not None}

        if not retrieved:
            result = HybridResult(
                query=query, filters_applied=filters_dict, n_candidates_after_filter=0,
                abstained=True,
                abstain_reason="ningún proceso cumple los filtros estructurados aplicados.",
                retrieved=[], answer=None, llm_error=None, model_name=None,
                input_tokens=None, output_tokens=None, usd_cost=None,
                latency_seconds=time.time() - t0,
            )
            self._log_cost(result)
            return result

        if top_similarity < threshold:
            result = HybridResult(
                query=query, filters_applied=filters_dict, n_candidates_after_filter=n_candidates,
                abstained=True,
                abstain_reason=(
                    f"similitud máxima {top_similarity:.3f} por debajo del threshold {threshold} "
                    f"(entre {n_candidates} procesos que sí cumplen los filtros)."
                ),
                retrieved=retrieved, answer=None, llm_error=None, model_name=None,
                input_tokens=None, output_tokens=None, usd_cost=None,
                latency_seconds=time.time() - t0,
            )
            self._log_cost(result)
            return result

        answer, llm_error, model_name, in_tok, out_tok, usd = self._generate_answer(query, retrieved)
        result = HybridResult(
            query=query, filters_applied=filters_dict, n_candidates_after_filter=n_candidates,
            abstained=False, abstain_reason=None, retrieved=retrieved,
            answer=answer, llm_error=llm_error, model_name=model_name,
            input_tokens=in_tok, output_tokens=out_tok, usd_cost=usd,
            latency_seconds=time.time() - t0,
        )
        self._log_cost(result)
        return result


_engine_cache: HybridRagEngine | None = None


def answer_question(
    query: str,
    config: dict,
    filters: StructuredFilters | None = None,
    threshold_override: float | None = None,
) -> HybridResult:
    """Única función pública del motor RAG híbrido.

    `threshold_override` permite ajustar el threshold de abstención por
    consulta (ej. desde un slider de la UI) sin reconstruir el motor ni
    recargar el índice — el índice se carga una sola vez y se cachea.
    """
    global _engine_cache
    if _engine_cache is None:
        _engine_cache = HybridRagEngine(config)
    return _engine_cache.answer_question(query, filters, threshold_override)
