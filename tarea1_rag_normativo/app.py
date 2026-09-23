"""Interfaz Streamlit del RAG Normativo (Tarea 1, Fase 5).

Uso:
    streamlit run app.py

Carga el índice ya construido por `build_index.py` (nunca lo reconstruye al iniciar).
Muestra: respuesta del LLM, fragmentos citados (documento, página, similitud),
estado de abstención, costo real, y un panel con el reporte de extracción (Fase 1)
y los resultados de evaluación (Fase 4).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from src.engine import answer_question

APP_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="RAG Normativo — Contrataciones Públicas", layout="wide")


@st.cache_resource
def get_config() -> dict:
    with (APP_DIR / "config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


config = get_config()

st.title("RAG Normativo — Contrataciones Públicas (Perú)")
st.caption(
    "Ley N.° 32069 (Ley General de Contrataciones Públicas) + Decreto Supremo N.° "
    "001-2026-EF · índice ya construido con `build_index.py`, no se reconstruye aquí."
)

tab_consulta, tab_calidad, tab_evaluacion = st.tabs(
    ["Consulta", "Calidad de datos (Fase 1)", "Evaluación (Fase 4)"]
)

with tab_consulta:
    with st.sidebar:
        st.subheader("Configuración activa")
        st.write(f"**Chunking:** `{config['chunking']['active']}`")
        st.write(f"**Modelo de embeddings:** `{config['embeddings']['providers'][config['embeddings']['active_provider']]['model_name']}`")
        st.write(f"**Modelo LLM:** `{config['llm']['model_name']}` ({config['llm']['provider']})")
        st.write(f"**Threshold de similitud:** `{config['retrieval']['similarity_threshold']}`")
        st.info(config["retrieval"]["scope"]["excluded_note"])

    query = st.text_input(
        "Pregunta sobre contrataciones públicas",
        placeholder="Ej: ¿Qué es la subcontratación en las contrataciones públicas?",
    )
    ask = st.button("Preguntar", type="primary")

    if ask and query.strip():
        with st.spinner("Buscando en el índice y generando respuesta..."):
            result = answer_question(query, config)

        if result.abstained:
            st.warning(f"**El motor se abstuvo de responder.** {result.abstain_reason}")
        elif result.llm_error:
            st.error(f"**Error al generar la respuesta con el LLM:** {result.llm_error}")
        else:
            st.success(result.answer)

        cost_col, latency_col = st.columns(2)
        with cost_col:
            if result.usd_cost is not None:
                st.metric(
                    "Costo de esta consulta",
                    f"USD {result.usd_cost:.6f}",
                    help=f"{result.input_tokens} tokens de entrada, {result.output_tokens} de salida ({result.model_name})",
                )
            else:
                st.metric("Costo de esta consulta", "USD 0.000000", help="Sin llamada al LLM (abstención o error)")
        with latency_col:
            st.metric("Latencia total", f"{result.latency_seconds:.2f} s")

        st.caption(result.scope_note)

        st.subheader(f"Fragmentos recuperados (top-{len(result.retrieved_chunks)})")
        for i, chunk in enumerate(result.retrieved_chunks, start=1):
            with st.expander(
                f"{i}. {chunk.doc_title} — página {chunk.page_number} "
                f"(versión: {chunk.version}, similitud: {chunk.similarity:.3f})"
            ):
                st.text(chunk.text)
    elif ask:
        st.warning("Escribe una pregunta antes de consultar.")

with tab_calidad:
    st.subheader("Source check (extracción, Fase 1)")
    df = read_csv_if_exists(APP_DIR / "data" / "processed" / "reports" / "source_check.csv")
    if df is not None:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Todavía no se generó `source_check.csv`. Corre `python build_index.py --stage extract`.")

    st.subheader("Calidad de la limpieza (headers, ligaduras, portadas)")
    df = read_csv_if_exists(APP_DIR / "data" / "processed" / "reports" / "extraction_quality.csv")
    if df is not None:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Todavía no se generó `extraction_quality.csv`. Corre `python build_index.py --stage clean`.")

    notes_path = APP_DIR / "data" / "processed" / "reports" / "extraction_quality_notes.md"
    if notes_path.exists():
        with st.expander("Limitaciones conocidas de la extracción/limpieza"):
            st.markdown(notes_path.read_text(encoding="utf-8"))

    st.subheader("Comparación de chunking (Fase 2)")
    df = read_csv_if_exists(APP_DIR / "data" / "processed" / "reports" / "chunking_comparison.csv")
    if df is not None:
        st.dataframe(df, use_container_width=True)

with tab_evaluacion:
    st.subheader("Recall@k por configuración de chunking (sin LLM)")
    df = read_csv_if_exists(APP_DIR / "eval" / "results" / "recall_summary.csv")
    if df is not None:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Todavía no se generó `recall_summary.csv`. Corre `python eval/run_eval.py`.")

    st.subheader("Comparación de embeddings: local vs. OpenAI")
    df = read_csv_if_exists(APP_DIR / "eval" / "results" / "openai_embeddings_comparison.csv")
    if df is not None:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Todavía no se generó. Corre `python eval/run_openai_comparison.py` (requiere API key de OpenAI).")

    st.subheader("Sweep de calibración del threshold de abstención")
    df = read_csv_if_exists(APP_DIR / "eval" / "results" / "threshold_sweep.csv")
    if df is not None:
        st.line_chart(
            df.pivot(index="threshold", columns="config_name", values="in_domain_incorrect_abstain_rate")
        )
        st.caption("Tasa de abstención incorrecta (in-domain) por threshold — más bajo es mejor.")
        st.dataframe(df, use_container_width=True)

    notes_path = APP_DIR / "eval" / "results" / "eval_notes.md"
    if notes_path.exists():
        with st.expander("Notas completas de evaluación (hallazgos, limitaciones)"):
            st.markdown(notes_path.read_text(encoding="utf-8"))
