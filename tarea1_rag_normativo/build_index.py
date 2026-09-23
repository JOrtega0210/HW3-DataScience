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
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from src.chunking import chunk_document_pages, get_tokenizer
from src.cleaning import clean_page
from src.extraction import (
    build_source_check,
    extract_pdf_pages,
    load_pages_jsonl,
    save_pages_jsonl,
)


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


def stage_clean(config: dict) -> None:
    processed_dir = Path(config["paths"]["processed_dir"])
    extraction_dir = processed_dir / "extraction"
    clean_dir = processed_dir / "clean"
    reports_dir = processed_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    quality_rows = []
    for doc in config["documents"]:
        jsonl_path = extraction_dir / f"{doc['id']}.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(f"No se encontró {jsonl_path}. Corre --stage extract primero.")
        pages = load_pages_jsonl(jsonl_path)
        results = [clean_page(p) for p in pages]

        clean_dir.mkdir(parents=True, exist_ok=True)
        with (clean_dir / f"{doc['id']}.jsonl").open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

        n_pages = len(results)
        n_header_removed = sum(1 for r in results if r.header_removed)
        n_sig_removed = sum(1 for r in results if r.signature_stamp_removed)
        n_op_codes = sum(r.op_codes_removed for r in results)
        n_ligatures = sum(r.ligatures_normalized for r in results)
        cover_pages = [r.page_number for r in results if r.is_cover_page]
        pages_without_header = [
            r.page_number for r in results if not r.header_removed and not r.is_cover_page
        ]

        quality_rows.append(
            {
                "doc_id": doc["id"],
                "num_pages": n_pages,
                "pages_header_removed": n_header_removed,
                "pages_without_header_flag": ",".join(map(str, pages_without_header)) or "ninguna",
                "signature_stamps_removed": n_sig_removed,
                "op_codes_removed_total": n_op_codes,
                "ligature_chars_normalized": n_ligatures,
                "cover_pages_excluded": ",".join(map(str, cover_pages)) or "ninguna",
            }
        )
        print(
            f"[clean] {doc['id']}: header removido en {n_header_removed}/{n_pages} páginas, "
            f"{n_ligatures} ligaduras normalizadas, {n_op_codes} códigos OP removidos, "
            f"{len(cover_pages)} página(s) de portada excluida(s)"
        )
        if pages_without_header:
            print(f"[clean] AVISO {doc['id']}: revisar manualmente páginas {pages_without_header} (sin header detectado y no marcadas como portada)")

    report_path = reports_dir / "extraction_quality.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(quality_rows[0].keys()))
        writer.writeheader()
        writer.writerows(quality_rows)
    print(f"[clean] reporte de calidad guardado en {report_path}")


def stage_chunk(config: dict) -> None:
    processed_dir = Path(config["paths"]["processed_dir"])
    clean_dir = processed_dir / "clean"
    chunks_dir = processed_dir / "chunks"
    reports_dir = processed_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    active_provider = config["embeddings"]["active_provider"]
    provider_cfg = config["embeddings"]["providers"][active_provider]
    model_name = provider_cfg["model_name"]
    max_tokens = provider_cfg["max_tokens"]
    tokenizer = get_tokenizer(model_name)

    comparison_rows = []
    for config_name, chunk_cfg in config["chunking"]["configs"].items():
        chunk_size = chunk_cfg["chunk_size_tokens"]
        overlap = chunk_cfg["overlap_tokens"]
        if chunk_size > max_tokens:
            raise ValueError(
                f"chunking.configs.{config_name}.chunk_size_tokens={chunk_size} excede "
                f"max_tokens={max_tokens} del modelo de embeddings ({model_name}). "
                "Los chunks se truncarían silenciosamente al generar el embedding."
            )

        for doc in config["documents"]:
            clean_path = clean_dir / f"{doc['id']}.jsonl"
            if not clean_path.exists():
                raise FileNotFoundError(f"No se encontró {clean_path}. Corre --stage clean primero.")
            with clean_path.open("r", encoding="utf-8") as f:
                pages = [json.loads(line) for line in f]

            chunks = chunk_document_pages(
                doc_id=doc["id"],
                version=doc["version"],
                pages=pages,
                tokenizer=tokenizer,
                config_name=config_name,
                chunk_size_tokens=chunk_size,
                overlap_tokens=overlap,
            )

            out_dir = chunks_dir / config_name
            out_dir.mkdir(parents=True, exist_ok=True)
            with (out_dir / f"{doc['id']}.jsonl").open("w", encoding="utf-8") as f:
                for c in chunks:
                    f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

            token_counts = [c.token_count for c in chunks]
            row = {
                "config_name": config_name,
                "chunk_size_tokens": chunk_size,
                "overlap_tokens": overlap,
                "doc_id": doc["id"],
                "num_chunks": len(chunks),
                "avg_tokens": round(sum(token_counts) / len(token_counts), 1) if token_counts else 0,
                "min_tokens": min(token_counts) if token_counts else 0,
                "max_tokens_seen": max(token_counts) if token_counts else 0,
            }
            comparison_rows.append(row)
            print(
                f"[chunk] {config_name}/{doc['id']}: {row['num_chunks']} chunks, "
                f"{row['avg_tokens']} tokens promedio (min={row['min_tokens']}, max={row['max_tokens_seen']})"
            )

    report_path = reports_dir / "chunking_comparison.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"[chunk] reporte comparativo guardado en {report_path}")


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
        stage_clean(config)
    if args.stage in ("chunk", "all"):
        stage_chunk(config)
    if args.stage in ("embed", "all"):
        print("[embed] pendiente — se implementa en Fase 2 (embeddings e índice)")


if __name__ == "__main__":
    main()
