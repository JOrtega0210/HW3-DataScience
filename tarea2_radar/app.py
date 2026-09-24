"""Dashboard Streamlit del RAG Radar (Tarea 2, Fase 4).

Uso:
    streamlit run app.py

Lee únicamente archivos precomputados (Fases 1, 2 y 5) y el índice del RAG
híbrido ya construido (Fase 3) — no descarga ni reconstruye nada al iniciar.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml
from unidecode import unidecode

from src.hybrid_engine import StructuredFilters, answer_question

APP_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title="RAG Radar — Contrataciones Abiertas", layout="wide")

CATEGORY_COLORS = {"goods": "#4C78A8", "services": "#F58518", "works": "#54A24B"}
CATEGORY_LABELS_ES = {"goods": "Bienes", "services": "Servicios", "works": "Obras"}


@st.cache_resource
def get_config() -> dict:
    with (APP_DIR / "config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_processes() -> pd.DataFrame:
    path = APP_DIR / "data" / "processed" / "processes_validated.jsonl"
    rows = [json.loads(line) for line in path.open("r", encoding="utf-8")]
    df = pd.DataFrame(rows)
    df["date_published"] = pd.to_datetime(df["date_published"], errors="coerce", utc=True).dt.tz_localize(None)
    df["geo_key"] = df["buyer_department"].fillna("").map(lambda d: unidecode(d).upper())
    df["month"] = df["date_published"].dt.strftime("%Y-%m")
    return df


@st.cache_data
def load_geojson() -> dict:
    with (APP_DIR / "data" / "outputs" / "geo" / "peru_departamentos.geojson").open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_csv_if_exists(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


config = get_config()
df = load_processes()
geojson = load_geojson()

st.title("RAG Radar — Contrataciones Abiertas (OECE)")
st.caption(
    "Datos precomputados de las Fases 1-2 (adquisición + validación) y Fase 5 "
    "(indicador de riesgo); este dashboard no descarga ni reconstruye nada al iniciar."
)

with st.sidebar:
    st.subheader("Filtros")
    departments = sorted(df["buyer_department"].dropna().unique())
    dept_sel = st.multiselect("Departamento", departments)
    cats = sorted(df["main_procurement_category"].dropna().unique())
    cat_sel = st.multiselect(
        "Categoría", cats, format_func=lambda c: CATEGORY_LABELS_ES.get(c, c)
    )
    max_amount = float(df["amount"].max())
    amt_range = st.slider("Rango de monto (PEN)", 0.0, max_amount, (0.0, max_amount))
    min_date = df["date_published"].min().date()
    max_date = df["date_published"].max().date()
    date_range = st.date_input("Rango de fecha de convocatoria", value=(min_date, max_date))
    sim_threshold = st.slider(
        "Threshold de similitud (pregunta RAG)", 0.0, 1.0,
        float(config["retrieval"]["similarity_threshold"]), 0.01,
    )

filtered = df.copy()
if dept_sel:
    filtered = filtered[filtered["buyer_department"].isin(dept_sel)]
if cat_sel:
    filtered = filtered[filtered["main_procurement_category"].isin(cat_sel)]
filtered = filtered[
    (filtered["amount"].fillna(0) >= amt_range[0]) & (filtered["amount"].fillna(0) <= amt_range[1])
]
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    filtered = filtered[
        (filtered["date_published"] >= pd.Timestamp(start)) & (filtered["date_published"] <= pd.Timestamp(end))
    ]

if filtered.empty:
    st.warning(
        "Ningún proceso cumple los filtros seleccionados. Ajusta el rango de monto, "
        "fecha, departamento o categoría en la barra lateral."
    )
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Procesos", f"{len(filtered):,}")
col2.metric("Monto total (PEN)", f"{filtered['amount'].sum():,.0f}")
col3.metric("Departamentos representados", filtered["buyer_department"].nunique())
awarded = filtered[filtered["num_awards"] > 0]
share = awarded["is_single_bidder_awarded"].mean() if len(awarded) else 0.0
col4.metric(
    "Share monopostor (adjudicados)", f"{share:.1%}",
    help="Procesos adjudicados con exactamente 1 postor / total de procesos adjudicados, dentro del filtro actual.",
)

tab_mapa, tab_pregunta, tab_tabla, tab_dist, tab_riesgo, tab_calidad = st.tabs(
    ["Mapa", "Pregunta (RAG híbrido)", "Tabla", "Distribución", "Indicador de riesgo", "Calidad de datos"]
)

with tab_mapa:
    metric = st.radio("Métrica del mapa", ["procesos", "monto"], horizontal=True)
    by_dept = (
        filtered.groupby(["buyer_department", "geo_key"])
        .agg(procesos=("ocid", "count"), monto=("amount", "sum"))
        .reset_index()
    )
    fig = px.choropleth(
        by_dept, geojson=geojson, locations="geo_key", featureidkey="properties.NOMBDEP",
        color=metric, color_continuous_scale="Blues",
        hover_name="buyer_department", hover_data={"procesos": True, "monto": ":,.0f", "geo_key": False},
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

with tab_pregunta:
    st.write("Pregunta en lenguaje natural sobre los procesos filtrados por condiciones estructuradas.")
    q_col1, q_col2 = st.columns(2)
    with q_col1:
        f_dept = st.selectbox("Departamento (filtro exacto, opcional)", [None] + departments)
        f_cat = st.selectbox(
            "Categoría (filtro exacto, opcional)", [None] + cats,
            format_func=lambda c: "Todas" if c is None else CATEGORY_LABELS_ES.get(c, c),
        )
    with q_col2:
        f_monto_min = st.number_input("Monto mínimo (PEN, opcional)", min_value=0.0, value=0.0)
        f_monto_max = st.number_input("Monto máximo (PEN, opcional, 0 = sin tope)", min_value=0.0, value=0.0)

    query = st.text_input("Pregunta", placeholder="Ej: ¿hay algún proceso para reemplazar un puente en Lima?")
    ask = st.button("Preguntar", type="primary")

    if ask and query.strip():
        filters = StructuredFilters(
            departamento=f_dept,
            categoria=f_cat,
            monto_min=f_monto_min if f_monto_min > 0 else None,
            monto_max=f_monto_max if f_monto_max > 0 else None,
        )
        with st.spinner("Filtrando candidatos y buscando por similitud semántica..."):
            result = answer_question(query, config, filters, threshold_override=sim_threshold)

        st.caption(f"Candidatos tras aplicar filtros estructurados: {result.n_candidates_after_filter}")
        if result.abstained:
            st.warning(f"**El motor se abstuvo de responder.** {result.abstain_reason}")
        elif result.llm_error:
            st.error(f"**Error al generar la respuesta:** {result.llm_error}")
        else:
            st.success(result.answer)

        if result.usd_cost is not None:
            st.metric("Costo de esta consulta", f"USD {result.usd_cost:.6f}")

        for r in result.retrieved:
            with st.expander(f"{r.ocid} — {r.tender_title} (similitud {r.similarity:.3f})"):
                st.write(f"**Comprador:** {r.buyer_name} ({r.buyer_department})")
                st.write(f"**Monto:** {r.currency} {r.amount:,.2f}" if r.amount is not None else "**Monto:** N/D")
                st.write(f"**Fecha de convocatoria:** {r.date_published}")
                st.write(f"**Categoría:** {CATEGORY_LABELS_ES.get(r.main_procurement_category, r.main_procurement_category)}")
    elif ask:
        st.warning("Escribe una pregunta antes de consultar.")

with tab_tabla:
    display_cols = [
        "ocid", "buyer_name", "buyer_department", "tender_title", "amount", "currency",
        "date_published", "main_procurement_category", "num_awards", "is_single_bidder_awarded",
    ]
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)
    st.download_button(
        "Descargar CSV (con filtros aplicados)",
        data=filtered[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="procesos_filtrados.csv",
        mime="text/csv",
    )

with tab_dist:
    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        by_cat = filtered.groupby("main_procurement_category").agg(monto=("amount", "sum")).reset_index()
        by_cat["categoria_es"] = by_cat["main_procurement_category"].map(CATEGORY_LABELS_ES)
        fig_cat = px.bar(
            by_cat, x="categoria_es", y="monto", color="main_procurement_category",
            color_discrete_map=CATEGORY_COLORS, title="Monto total por categoría",
        )
        fig_cat.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Monto (PEN)")
        st.plotly_chart(fig_cat, use_container_width=True)
    with dist_col2:
        by_month = filtered.groupby("month").agg(procesos=("ocid", "count")).reset_index().sort_values("month")
        fig_month = px.bar(by_month, x="month", y="procesos", title="Procesos por mes")
        fig_month.update_traces(marker_color="#4C78A8")
        fig_month.update_layout(xaxis_title=None, yaxis_title="Procesos")
        st.plotly_chart(fig_month, use_container_width=True)

    by_dept_amount = (
        filtered.groupby("buyer_department").agg(monto=("amount", "sum")).reset_index()
        .sort_values("monto", ascending=False).head(15)
    )
    fig_dept = px.bar(
        by_dept_amount, x="monto", y="buyer_department", orientation="h",
        title="Top 15 departamentos por monto",
    )
    fig_dept.update_traces(marker_color="#4C78A8")
    fig_dept.update_layout(yaxis_title=None, xaxis_title="Monto (PEN)", yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_dept, use_container_width=True)

with tab_riesgo:
    st.warning(
        "**Un solo postor no es evidencia de un delito.** Puede reflejar mercados con "
        "poca oferta o contrataciones muy especializadas. Este indicador es una señal "
        "estadística para priorizar revisión, no una acusación."
    )
    dept_risk = load_csv_if_exists(str(APP_DIR / "data" / "outputs" / "risk_monopostor_by_department.csv"))
    if dept_risk is not None:
        st.subheader("Share de monopostor por departamento")
        st.dataframe(dept_risk, use_container_width=True, hide_index=True)
    supplier_risk = load_csv_if_exists(str(APP_DIR / "data" / "outputs" / "risk_monopostor_top_suppliers.csv"))
    if supplier_risk is not None:
        min_p = config["risk_indicator"]["min_processes_per_entity"]
        st.subheader(f"Top {len(supplier_risk)} proveedores (mínimo {min_p} procesos adjudicados)")
        st.caption("Los proveedores que no calzan con un patrón de persona jurídica se anonimizan.")
        st.dataframe(supplier_risk, use_container_width=True, hide_index=True)
    if dept_risk is None and supplier_risk is None:
        st.info("Corre `python compute_risk_indicator.py` para generar este reporte.")

with tab_calidad:
    quality_report = load_csv_if_exists(config["logging"]["quality_report_file"])
    if quality_report is not None:
        st.subheader("Reporte de calidad (Fase 2)")
        st.dataframe(quality_report, use_container_width=True, hide_index=True)
    duplicates = load_csv_if_exists(str(APP_DIR / "data" / "outputs" / "duplicates_removed.csv"))
    if duplicates is not None:
        with st.expander(f"Detalle de {len(duplicates)} registros duplicados removidos"):
            st.dataframe(duplicates, use_container_width=True, hide_index=True)
    acquisition_summary = load_csv_if_exists(str(APP_DIR / "data" / "outputs" / "acquisition_summary.csv"))
    if acquisition_summary is not None:
        st.subheader("Resumen de adquisición (Fase 1)")
        st.dataframe(acquisition_summary, use_container_width=True, hide_index=True)
