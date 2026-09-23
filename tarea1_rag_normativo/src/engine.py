"""Motor RAG online (Tarea 1, Fase 3).

Interfaz única expuesta por una función: `answer_question()`. Sin dependencias de
UI (solo numpy/faiss/sentence-transformers vía los módulos propios + stdlib) — se
puede importar y testear sin Streamlit instalado.

Responsabilidades:
- Embeber la consulta y recuperar los top-k fragmentos del índice ya construido
  offline (nunca reconstruye el índice).
- Abstención por threshold de similitud ANTES de llamar al LLM (si no hay evidencia
  suficiente, no se gasta ni un token de generación).
- Citas explícitas de documento + página + versión en cada fragmento devuelto.
- Nota de alcance del corpus siempre presente en el resultado (qué queda fuera).
- Errores de API devueltos como datos estructurados (`llm_error`), nunca excepciones
  sin capturar.
- Logging de costo observable (modelo, tokens, latencia, USD) en `logs/costs.csv`.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

from src.embeddings import embed_texts
from src.indexing import load_index


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    version: str
    page_number: int
    similarity: float
    text: str


@dataclass
class RAGResult:
    query: str
    abstained: bool
    abstain_reason: str | None
    retrieved_chunks: list[RetrievedChunk]
    answer: str | None
    llm_error: str | None
    scope_note: str
    model_name: str | None
    input_tokens: int | None
    output_tokens: int | None
    usd_cost: float | None
    latency_seconds: float


class RagEngine:
    """Implementación interna. Usar `answer_question()` como punto de entrada."""

    def __init__(self, config: dict, index_config_name: str | None = None):
        self.config = config
        self.index_config_name = index_config_name or config["chunking"]["active"]
        index_dir = Path(config["paths"]["index_dir"]) / self.index_config_name
        self.faiss_index, self.ids, self.metadata = load_index(index_dir)

        active_provider = config["embeddings"]["active_provider"]
        provider_cfg = config["embeddings"]["providers"][active_provider]
        self.embedding_model_name = provider_cfg["model_name"]
        self.query_prefix = provider_cfg.get("query_prefix", "")

        self.top_k = config["retrieval"]["top_k"]
        self.threshold = config["retrieval"]["similarity_threshold"]
        self.scope_note = config["retrieval"]["scope"]["excluded_note"]

        self.doc_by_id = {d["id"]: d for d in config["documents"]}
        self.costs_log_path = Path(config["paths"]["logs_dir"]) / "costs.csv"

    def _retrieve(self, query: str) -> list[RetrievedChunk]:
        qvec = embed_texts([query], model_name=self.embedding_model_name, prefix=self.query_prefix)
        scores, idxs = self.faiss_index.search(qvec, self.top_k)
        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            m = self.metadata[idx]
            doc = self.doc_by_id.get(m["doc_id"], {})
            results.append(
                RetrievedChunk(
                    chunk_id=m["chunk_id"],
                    doc_id=m["doc_id"],
                    doc_title=doc.get("title", m["doc_id"]),
                    version=m["version"],
                    page_number=m["page_number"],
                    similarity=float(score),
                    text=m["text"],
                )
            )
        return results

    def _generate_answer(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> tuple[str | None, str | None, str, int | None, int | None, float | None]:
        """Pendiente de conectar a un proveedor real (ver Credenciales en README).
        Devuelve siempre una tupla; los errores son datos, no excepciones."""
        llm_cfg = self.config["llm"]
        model_name = llm_cfg["model_name"]
        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return (
                None,
                "Sin API key configurada (ANTHROPIC_API_KEY / OPENAI_API_KEY vacías en "
                ".env): se devuelven los fragmentos recuperados pero no se genera "
                "respuesta de LLM todavía.",
                model_name,
                None,
                None,
                None,
            )
        return (
            None,
            "Llamada real al LLM aún no implementada (pendiente de una parte "
            "posterior una vez confirmado el proveedor a usar).",
            model_name,
            None,
            None,
            None,
        )

    def _log_cost(self, result: RAGResult) -> None:
        self.costs_log_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.costs_log_path.exists()
        with self.costs_log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(
                    [
                        "timestamp",
                        "query",
                        "abstained",
                        "model_name",
                        "input_tokens",
                        "output_tokens",
                        "usd_cost",
                        "latency_seconds",
                    ]
                )
            writer.writerow(
                [
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                    result.query,
                    result.abstained,
                    result.model_name,
                    result.input_tokens,
                    result.output_tokens,
                    result.usd_cost,
                    round(result.latency_seconds, 3),
                ]
            )

    def answer_question(self, query: str) -> RAGResult:
        t0 = time.time()
        chunks = self._retrieve(query)
        top_similarity = chunks[0].similarity if chunks else 0.0

        if not chunks or top_similarity < self.threshold:
            result = RAGResult(
                query=query,
                abstained=True,
                abstain_reason=(
                    f"similitud máxima {top_similarity:.3f} por debajo del threshold "
                    f"{self.threshold} (o el índice no devolvió resultados)."
                ),
                retrieved_chunks=chunks,
                answer=None,
                llm_error=None,
                scope_note=self.scope_note,
                model_name=None,
                input_tokens=None,
                output_tokens=None,
                usd_cost=None,
                latency_seconds=time.time() - t0,
            )
            self._log_cost(result)
            return result

        answer, llm_error, model_name, in_tok, out_tok, usd = self._generate_answer(query, chunks)
        result = RAGResult(
            query=query,
            abstained=False,
            abstain_reason=None,
            retrieved_chunks=chunks,
            answer=answer,
            llm_error=llm_error,
            scope_note=self.scope_note,
            model_name=model_name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            usd_cost=usd,
            latency_seconds=time.time() - t0,
        )
        self._log_cost(result)
        return result


_engine_cache: dict[str, RagEngine] = {}


def answer_question(query: str, config: dict, index_config_name: str | None = None) -> RAGResult:
    """Única función pública del motor RAG.

    Recibe una pregunta en lenguaje natural y devuelve un `RAGResult` estructurado
    con: fragmentos citados (documento, página, versión, similitud), si se abstuvo
    y por qué, la nota de alcance del corpus, y costo/latencia. No importa nada de
    Streamlit ni de ninguna UI.
    """
    key = index_config_name or config["chunking"]["active"]
    if key not in _engine_cache:
        _engine_cache[key] = RagEngine(config, index_config_name=key)
    return _engine_cache[key].answer_question(query)
