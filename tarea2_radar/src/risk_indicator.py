"""Indicador de riesgo: adjudicaciones monopostor (Tarea 2, Fase 5).

Share de procesos adjudicados con exactamente 1 postor, por departamento y por
proveedor. Un solo postor no es evidencia de irregularidad por sí sola (puede
ser un mercado con poca oferta, o una contratación muy especializada) — es una
señal estadística para priorizar revisión, nunca una acusación.

Sin dependencias de UI: solo stdlib.
"""

from __future__ import annotations

from collections import defaultdict

COMPANY_MARKERS = (
    "S.A.C", "S.A.", "S.R.L", "E.I.R.L", "S.A.A", "S.C.R.L",
    "CONSORCIO", "COOPERATIVA", "EMPRESA", "ASOCIACION", "ASOCIACIÓN",
    "FUNDACION", "FUNDACIÓN", "ONG", "MUNICIPALIDAD", "UNIVERSIDAD",
)


def is_probable_company(name: str) -> bool:
    """Heurística simple: si el nombre no tiene ningún marcador de persona
    jurídica, se asume persona natural (nombre y apellidos) y se anonimiza en
    el Top 10 — nunca se publica el nombre de una persona natural individual."""
    upper = name.upper()
    return any(marker in upper for marker in COMPANY_MARKERS)


def compute_department_share(rows: list[dict]) -> list[dict]:
    """Share de monopostor por departamento, solo sobre procesos con al menos
    una adjudicación (num_awards > 0)."""
    awarded = [r for r in rows if r.get("num_awards", 0) > 0 and r.get("buyer_department")]
    totals: dict[str, int] = defaultdict(int)
    single_bidder: dict[str, int] = defaultdict(int)
    for r in awarded:
        dept = r["buyer_department"]
        totals[dept] += 1
        if r.get("is_single_bidder_awarded"):
            single_bidder[dept] += 1

    result = []
    for dept, total in totals.items():
        result.append(
            {
                "buyer_department": dept,
                "total_awarded_processes": total,
                "single_bidder_processes": single_bidder.get(dept, 0),
                "single_bidder_share": round(single_bidder.get(dept, 0) / total, 4),
            }
        )
    result.sort(key=lambda x: -x["single_bidder_share"])
    return result


def compute_supplier_share(
    rows: list[dict], min_processes: int, top_n: int
) -> list[dict]:
    """Share de monopostor por proveedor adjudicado, filtrando proveedores con
    menos de `min_processes` procesos adjudicados en total (evita reportar un
    100% de share basado en un solo proceso, estadísticamente poco confiable).
    Los nombres que no calzan con un patrón de persona jurídica se anonimizan."""
    awarded = [r for r in rows if r.get("num_awards", 0) > 0 and r.get("award_supplier_names")]

    totals: dict[str, int] = defaultdict(int)
    single_bidder: dict[str, int] = defaultdict(int)
    for r in awarded:
        suppliers = [s.strip() for s in r["award_supplier_names"].split(";") if s.strip()]
        for supplier in suppliers:
            totals[supplier] += 1
            if r.get("is_single_bidder_awarded"):
                single_bidder[supplier] += 1

    rows_out = []
    for supplier, total in totals.items():
        if total < min_processes:
            continue
        rows_out.append(
            {
                "supplier_name": supplier,
                "is_company": is_probable_company(supplier),
                "total_awarded_processes": total,
                "single_bidder_processes": single_bidder.get(supplier, 0),
                "single_bidder_share": round(single_bidder.get(supplier, 0) / total, 4),
            }
        )

    rows_out.sort(key=lambda x: (-x["single_bidder_share"], -x["total_awarded_processes"]))
    top_rows = rows_out[:top_n]

    anon_counter = 0
    for row in top_rows:
        if not row["is_company"]:
            anon_counter += 1
            row["supplier_name"] = f"(persona natural #{anon_counter} — nombre no publicado)"
    return top_rows
