"""Validación de calidad y normalización territorial (Tarea 2, Fase 2).

Uso:
    python validate_data.py

Lee data/processed/processes.jsonl (salida de acquire_data.py --stage combine)
y produce:
    data/processed/processes_validated.jsonl   -- una fila por proceso, deduplicado
    data/outputs/duplicates_removed.csv         -- auditoría de qué se descartó y por qué
    data/outputs/unmatched_locations.csv        -- ubicaciones que no matchean un departamento
    data/outputs/data_quality_report.csv        -- por regla: cuántos flagged, acción tomada
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from src.validation import (
    build_quality_report,
    detect_encoding_issues,
    find_duplicate_groups,
    normalize_department,
    resolve_duplicates,
)


def load_config(path: Path = Path("config.yaml")) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()
    processed_dir = Path(config["paths"]["processed_dir"])
    outputs_dir = Path(config["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    in_path = processed_dir / "processes.jsonl"
    with in_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    print(f"[validate] {len(rows)} filas cargadas desde {in_path}")

    # 1) Registros repetidos por el mismo proceso real (mismo buyer+título+fecha, ocid distinto)
    duplicate_groups = find_duplicate_groups(rows)
    deduped_rows, removed_log = resolve_duplicates(rows, duplicate_groups)
    print(
        f"[validate] {len(duplicate_groups)} grupos duplicados detectados, "
        f"{len(removed_log)} filas descartadas (se conserva la más completa por grupo)"
    )

    if removed_log:
        with (outputs_dir / "duplicates_removed.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(removed_log[0].keys()))
            writer.writeheader()
            writer.writerows(removed_log)

    # 2) Inconsistencias de encoding/acentos (sobre el dataset ya deduplicado)
    encoding_flagged = set(
        detect_encoding_issues(
            deduped_rows, text_fields=["tender_title", "tender_description", "buyer_name"]
        )
    )

    # 3) Normalización territorial: buyer_department_raw -> uno de los 25 departamentos
    canonical_departments = config["territorial_normalization"]["departments"]
    unmatched_rows = []
    for row in deduped_rows:
        normalized = normalize_department(row.get("buyer_department_raw"), canonical_departments)
        row["buyer_department"] = normalized
        row["is_encoding_issue"] = row["ocid"] in encoding_flagged
        if normalized is None and row.get("buyer_department_raw"):
            unmatched_rows.append(
                {"ocid": row["ocid"], "buyer_department_raw": row["buyer_department_raw"]}
            )

    if unmatched_rows:
        unmatched_path = Path(config["territorial_normalization"]["unmatched_report_file"])
        unmatched_path.parent.mkdir(parents=True, exist_ok=True)
        with unmatched_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["ocid", "buyer_department_raw"])
            writer.writeheader()
            writer.writerows(unmatched_rows)
    print(f"[validate] {len(unmatched_rows)} procesos con ubicación no localizable a un departamento")

    # 4) Guardar dataset validado
    out_path = processed_dir / "processes_validated.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in deduped_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = list(deduped_rows[0].keys()) if deduped_rows else []
    with (processed_dir / "processes_validated.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deduped_rows)
    print(f"[validate] dataset validado ({len(deduped_rows)} filas) guardado en {out_path}")

    # 5) Reporte de calidad por regla
    quality_rules = build_quality_report(
        rows_before=rows,
        rows_after=deduped_rows,
        duplicate_removed_count=len(removed_log),
        unmatched_department_count=len(unmatched_rows),
        encoding_issue_count=len(encoding_flagged),
    )
    report_path = config["logging"]["quality_report_file"]
    with Path(report_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rule", "flagged_count", "action_taken"])
        writer.writeheader()
        for rule in quality_rules:
            writer.writerow(asdict(rule))
    print(f"[validate] reporte de calidad guardado en {report_path}")
    for rule in quality_rules:
        print(f"  - {rule.rule}: {rule.flagged_count} casos -> {rule.action_taken}")


if __name__ == "__main__":
    main()
