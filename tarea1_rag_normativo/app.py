"""Interfaz Streamlit del RAG Normativo (Tarea 1, Fase 5).

Uso:
    streamlit run app.py

Debe: cargar un índice ya construido (no reconstruir al iniciar), mostrar respuesta,
fragmentos citados (documento, página, similitud), estado de abstención y costo, más un
panel con el reporte de extracción (Fase 1) y resultados de evaluación (Fase 4).
"""

import streamlit as st

st.set_page_config(page_title="RAG Normativo — Contrataciones Públicas", layout="wide")
st.title("RAG Normativo — Contrataciones Públicas (Perú)")
st.info(
    "Interfaz pendiente de implementación. Se completa al cerrar la Fase 5, "
    "una vez que el motor RAG (src/) y el índice (build_index.py) estén listos."
)
