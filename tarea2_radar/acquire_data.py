"""Pipeline de adquisición de datos (Tarea 2, Fase 1).

Uso:
    python acquire_data.py --stage bulk     # descarga masiva de config.yaml -> months
    python acquire_data.py --stage recent   # API de actualizaciones recientes
    python acquire_data.py --stage all      # ambos + combina en un solo dataset

Ambos mecanismos son re-ejecutables sin duplicar descargas (ver src/acquisition.py).
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from src.acquisition import AcquisitionLogger, download_bulk_month, fetch_recent_updates
from src.flatten import flatten_compiled_release


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _iter_bulk_records(extract_dir: Path):
    """Los zips mensuales pueden traer uno o varios .json (recordPackage)."""
    for json_path in sorted(extract_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as f:
            package = json.load(f)
        yield from package.get("records", [])


def stage_bulk(config: dict, logger: AcquisitionLogger) -> list[dict]:
    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    seen_ocids: set[str] = set()
    for month_str in config["data_source"]["months"]:
        year, month = month_str.split("-")
        extract_dir = download_bulk_month(year, month, config, logger)
        n_before = len(rows)
        for record in _iter_bulk_records(extract_dir):
            compiled = record.get("compiledRelease") or {}
            ocid = compiled.get("ocid")
            if not ocid or ocid in seen_ocids:
                continue
            seen_ocids.add(ocid)
            rows.append(flatten_compiled_release(compiled, row_source=f"bulk:{month_str}"))
        print(f"[bulk] {month_str}: {len(rows) - n_before} procesos nuevos (total acumulado: {len(rows)})")

    out_path = processed_dir / "processes_bulk.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[bulk] guardado en {out_path}")
    return rows


def stage_recent(config: dict, logger: AcquisitionLogger) -> list[dict]:
    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    window_days = config["data_source"]["api"]["recent_window_days"]
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=window_days)

    records = fetch_recent_updates(
        start_date.isoformat(), end_date.isoformat(), config, logger
    )
    rows = []
    seen_ocids: set[str] = set()
    for record in records:
        compiled = record.get("compiledRelease") or {}
        ocid = compiled.get("ocid")
        if not ocid or ocid in seen_ocids:
            continue
        seen_ocids.add(ocid)
        rows.append(flatten_compiled_release(compiled, row_source="api:recent"))

    out_path = processed_dir / "processes_recent.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[recent] ventana {start_date} -> {end_date}: {len(rows)} procesos "
        f"(guardado en {out_path})"
    )
    return rows


def stage_combine(config: dict) -> None:
    processed_dir = Path(config["paths"]["processed_dir"])
    all_rows: dict[str, dict] = {}

    for name in ("processes_bulk.jsonl", "processes_recent.jsonl"):
        path = processed_dir / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                # Si el mismo ocid aparece en ambos, se prefiere la versión "recent"
                # (más fresca) sobre la del bulk histórico.
                existing = all_rows.get(row["ocid"])
                if existing is None or row["row_source"].startswith("api"):
                    all_rows[row["ocid"]] = row

    out_path = processed_dir / "processes.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in all_rows.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = list(next(iter(all_rows.values())).keys()) if all_rows else []
    with (processed_dir / "processes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows.values())

    print(f"[combine] {len(all_rows)} procesos únicos (una fila por ocid) -> {out_path}")


def stage_report(config: dict) -> None:
    """Agrega logs/acquisition.log en un resumen de tiempo/requests/tamaños."""
    log_path = Path(config["logging"]["acquisition_log_file"])
    outputs_dir = Path(config["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        print("[report] no hay log de adquisición todavía.")
        return

    with log_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    summary: dict[str, dict] = {}
    for row in rows:
        action = row["action"]
        s = summary.setdefault(
            action, {"requests": 0, "bytes_downloaded": 0, "duration_seconds": 0.0}
        )
        s["requests"] += 1
        s["bytes_downloaded"] += int(row["bytes_downloaded"])
        s["duration_seconds"] += float(row["duration_seconds"])

    report_path = outputs_dir / "acquisition_summary.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["action", "requests", "bytes_downloaded", "duration_seconds"])
        for action, s in summary.items():
            writer.writerow([action, s["requests"], s["bytes_downloaded"], round(s["duration_seconds"], 3)])
    print(f"[report] resumen de adquisición guardado en {report_path}")
    for action, s in summary.items():
        print(
            f"  {action}: {s['requests']} requests, "
            f"{s['bytes_downloaded'] / 1_000_000:.2f} MB, {s['duration_seconds']:.1f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Adquisición de datos OECE (Tarea 2)")
    parser.add_argument(
        "--stage", choices=["bulk", "recent", "combine", "report", "all"], default="all"
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    logger = AcquisitionLogger(Path(config["logging"]["acquisition_log_file"]))

    if args.stage in ("bulk", "all"):
        stage_bulk(config, logger)
    if args.stage in ("recent", "all"):
        stage_recent(config, logger)
    if args.stage in ("combine", "all"):
        stage_combine(config)
    if args.stage in ("report", "all"):
        stage_report(config)


if __name__ == "__main__":
    main()
