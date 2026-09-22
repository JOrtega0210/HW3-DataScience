"""Extracción de texto desde PDFs preservando número de página (Tarea 1, Fase 1).

Sin dependencias de UI: solo pymupdf + stdlib, para poder testear/importar el motor
de forma aislada.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf


@dataclass
class PageExtraction:
    doc_id: str
    page_number: int  # 1-indexed, tal como aparece impreso en el PDF
    char_count: int
    raw_text: str


def extract_pdf_pages(pdf_path: Path, doc_id: str) -> list[PageExtraction]:
    """Extrae texto por página. Usa sort=True para preservar el orden de lectura
    (top-to-bottom, left-to-right) incluso si el PDF tiene bloques en columnas."""
    pages: list[PageExtraction] = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text", sort=True)
            pages.append(
                PageExtraction(
                    doc_id=doc_id,
                    page_number=i,
                    char_count=len(text.strip()),
                    raw_text=text,
                )
            )
    return pages


def save_pages_jsonl(pages: list[PageExtraction], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")


def load_pages_jsonl(path: Path) -> list[PageExtraction]:
    pages = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            pages.append(PageExtraction(**json.loads(line)))
    return pages


@dataclass
class SourceCheckRow:
    doc_id: str
    num_pages: int
    total_chars: int
    avg_chars_per_page: float
    min_chars_per_page: int
    max_chars_per_page: int
    empty_pages: str  # lista separada por comas, o "ninguna"
    empty_page_count: int
    reading_order_method: str


def build_source_check(
    pages: list[PageExtraction], empty_page_threshold: int = 10
) -> SourceCheckRow:
    doc_id = pages[0].doc_id
    counts = [p.char_count for p in pages]
    empty_pages = [p.page_number for p in pages if p.char_count < empty_page_threshold]
    return SourceCheckRow(
        doc_id=doc_id,
        num_pages=len(pages),
        total_chars=sum(counts),
        avg_chars_per_page=round(sum(counts) / len(counts), 1) if counts else 0.0,
        min_chars_per_page=min(counts) if counts else 0,
        max_chars_per_page=max(counts) if counts else 0,
        empty_pages=",".join(str(n) for n in empty_pages) if empty_pages else "ninguna",
        empty_page_count=len(empty_pages),
        reading_order_method=(
            "pymupdf page.get_text('text', sort=True): ordena los bloques de texto "
            "por posición vertical y horizontal antes de concatenarlos."
        ),
    )
