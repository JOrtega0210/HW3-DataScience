"""Dashboard Streamlit del RAG Radar (Tarea 2, Fase 4).

Uso:
    streamlit run app.py

Debe: leer únicamente archivos precomputados (no descargar ni reconstruir al cargar),
con vistas de KPIs, mapa choropleth, question box (RAG híbrido), tabla ordenable/
descargable, distribución y panel de calidad de datos.
"""

import streamlit as st

st.set_page_config(page_title="RAG Radar — Contrataciones Abiertas", layout="wide")
st.title("RAG Radar — Contrataciones Abiertas (OECE)")
st.info(
    "Dashboard pendiente de implementación. Se completa al cerrar la Fase 4 de la Tarea 2, "
    "una vez que la adquisición, validación e índice (src/) estén listos."
)
