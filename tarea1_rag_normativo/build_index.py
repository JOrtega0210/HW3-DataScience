"""Pipeline offline de indexación (Tarea 1).

Uso:
    python build_index.py --stage extract   # Fase 1: extracción + source check
    python build_index.py --stage all       # corre todas las etapas implementadas

Etapas (se implementan por fases, ver docs/pipeline.md):
    extract -> clean -> chunk -> embed
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import yaml

from src.extraction import build_source_check, extract_pdf_pages, save_pages_jsonl


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def stage_extract(config: dict) -> None:
    raw_dir = Path(config["paths"]["raw_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    extraction_dir = processed_dir / "extraction"
    reports_dir = processed_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for doc in config["documents"]:
        pdf_path = raw_dir / doc["raw_filename"]
        if not pdf_path.exists():
            raise FileNotFoundError(
                f"No se encontró {pdf_path}. Descargar el PDF oficial antes de extraer."
            )
        pages = extract_pdf_pages(pdf_path, doc_id=doc["id"])
        save_pages_jsonl(pages, extraction_dir / f"{doc['id']}.jsonl")
        row = build_source_check(pages)
        rows.append(row)
        print(
            f"[extract] {doc['id']}: {row.num_pages} páginas, "
            f"{row.total_chars} caracteres, "
            f"{row.empty_page_count} página(s) sin texto extraíble ({row.empty_pages})"
        )

    report_path = reports_dir / "source_check.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print(f"[extract] reporte de source check guardado en {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline offline de indexación (Tarea 1)")
    parser.add_argument(
        "--stage", choices=["extract", "clean", "chunk", "embed", "all"], default="all"
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(Path(args.config))

    if args.stage in ("extract", "all"):
        stage_extract(config)
    if args.stage in ("clean", "all"):
        print("[clean] pendiente — se implementa en la siguiente parte (limpieza de headers)")
    if args.stage in ("chunk", "all"):
        print("[chunk] pendiente — se implementa en Fase 2 (chunking)")
    if args.stage in ("embed", "all"):
        print("[embed] pendiente — se implementa en Fase 2 (embeddings e índice)")


if __name__ == "__main__":
    main()
