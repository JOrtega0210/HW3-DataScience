"""Indicador de riesgo: adjudicaciones monopostor (Tarea 2, Fase 5).

Uso:
    python compute_risk_indicator.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from src.risk_indicator import compute_department_share, compute_supplier_share


def load_config(path: Path = Path("config.yaml")) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()
    processed_dir = Path(config["paths"]["processed_dir"])
    outputs_dir = Path(config["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    in_path = processed_dir / "processes_validated.jsonl"
    with in_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    min_processes = config["risk_indicator"]["min_processes_per_entity"]
    top_n = config["risk_indicator"]["top_n_suppliers"]

    dept_share = compute_department_share(rows)
    with (outputs_dir / "risk_monopostor_by_department.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dept_share[0].keys()))
        writer.writeheader()
        writer.writerows(dept_share)
    print(f"[risk] {len(dept_share)} departamentos -> risk_monopostor_by_department.csv")
    for row in dept_share[:5]:
        print(f"  {row['buyer_department']}: {row['single_bidder_share']:.1%} ({row['single_bidder_processes']}/{row['total_awarded_processes']})")

    supplier_share = compute_supplier_share(rows, min_processes=min_processes, top_n=top_n)
    with (outputs_dir / "risk_monopostor_top_suppliers.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(supplier_share[0].keys()))
        writer.writeheader()
        writer.writerows(supplier_share)
    print(f"[risk] top {len(supplier_share)} proveedores (min {min_processes} procesos) -> risk_monopostor_top_suppliers.csv")
    for row in supplier_share[:5]:
        print(f"  {row['supplier_name']}: {row['single_bidder_share']:.1%} ({row['single_bidder_processes']}/{row['total_awarded_processes']})")


if __name__ == "__main__":
    main()
