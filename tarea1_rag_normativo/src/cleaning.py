"""Limpieza de texto extraído (Tarea 1, Fase 1).

Remueve artefactos de maquetación introducidos por el diseño de doble columna de
El Peruano: el header de navegación repetido en cada página, el sello digital de
firma, ligaduras tipográficas (ﬁ, ﬂ, ...) y códigos de publicación (OP) que quedan
pegados al texto por la extracción en modo "sort" (orden de lectura).

Sin dependencias de UI: solo stdlib, para poder testear/importar de forma aislada.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.extraction import PageExtraction

LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
}

# Dos variantes del header repetido, según si la página par/impar de la edición
# pone primero el número de página o primero "El Peruano / <fecha>".
_HEADER_A = re.compile(
    r"^[ \t]*\d{1,4}[ \t]+NORMAS LEGALES[ \t]+[^\n/]+?/[ \t]*El Peruano[ \t]*\n+",
    re.MULTILINE,
)
_HEADER_B = re.compile(
    r"^[ \t]*El Peruano[ \t]*/[ \t]*[^\n]+?NORMAS LEGALES[ \t]+\d{1,4}[ \t]*\n+",
    re.MULTILINE,
)
_SIGNATURE_STAMP = re.compile(
    r"Firmado por:[ \t]*Editora[ \t\n]*Peru[ \t]*\n"
    r"(?:[ \t]*Fecha:[ \t]*\d{2}/\d{2}/\d{4}[ \t]*\d{1,2}:\d{2}[ \t]*\n)?",
    re.IGNORECASE,
)
# Código de publicación (OP) de El Peruano, ej. "2300373-1": nunca es parte del
# texto normativo, aparece como artefacto de columna vecina o footer de cierre.
_OP_CODE = re.compile(r"[ \t]*\b\d{6,8}-\d{1,3}\b[ \t]*")


def normalize_ligatures(text: str) -> str:
    # PyMuPDF inserta un espacio espurio inmediatamente DESPUÉS del glifo de
    # ligadura (problema conocido de font metrics), partiendo palabras como
    # "Efi cientes" o "fi n" en vez de "Eficientes"/"fin". Se elimina ese
    # espacio al reemplazar; el espacio ANTES de la ligadura no es consistente
    # (a veces es un separador real, a veces no) y se deja tal cual: ver
    # limitación documentada en el reporte de calidad de extracción.
    for lig, repl in LIGATURES.items():
        text = re.sub(re.escape(lig) + r"[ \t]?", repl, text)
    return text


@dataclass
class CleaningResult:
    doc_id: str
    page_number: int
    clean_text: str
    header_removed: bool
    signature_stamp_removed: bool
    op_codes_removed: int
    ligatures_normalized: int
    is_cover_page: bool


def clean_page(page: PageExtraction) -> CleaningResult:
    text = page.raw_text

    ligature_count = sum(text.count(lig) for lig in LIGATURES)
    text = normalize_ligatures(text)

    header_removed = False
    new_text, n = _HEADER_A.subn("", text, count=1)
    if n:
        header_removed = True
        text = new_text
    else:
        new_text, n = _HEADER_B.subn("", text, count=1)
        if n:
            header_removed = True
            text = new_text

    new_text, sig_n = _SIGNATURE_STAMP.subn("", text)
    signature_removed = sig_n > 0
    text = new_text

    op_matches = _OP_CODE.findall(text)
    text = _OP_CODE.sub(" ", text)

    # Colapsar espacios múltiples (layout justificado a doble columna), sin tocar saltos de línea.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # Heurística específica para estos dos documentos: la portada de la "separata
    # especial" de la Ley 32069 es puro título/gráfico, sin contenido normativo.
    is_cover = "SEPARATA ESPECIAL" in page.raw_text.upper()

    return CleaningResult(
        doc_id=page.doc_id,
        page_number=page.page_number,
        clean_text=text,
        header_removed=header_removed,
        signature_stamp_removed=signature_removed,
        op_codes_removed=len(op_matches),
        ligatures_normalized=ligature_count,
        is_cover_page=is_cover,
    )
