"""Aplanado de compiledRelease (OCDS) a una fila por proceso de contratación
(Tarea 2, Fase 1).

Un "record" del recordPackage OCDS ya es la vista compilada/fusionada de todas las
"releases" (eventos: planificación, convocatoria, adjudicación, ...) de un mismo
`ocid` — por eso partimos de `record.compiledRelease` en vez de reconstruirlo a
mano desde `record.releases`.

Sin dependencias de UI: solo stdlib.
"""

from __future__ import annotations

from typing import Any


def _find_buyer_department(compiled_release: dict) -> str | None:
    parties = compiled_release.get("parties") or []
    for p in parties:
        if "buyer" in (p.get("roles") or []):
            address = p.get("address") or {}
            return address.get("department")
    # fallback: primera parte con address.department disponible
    for p in parties:
        address = p.get("address") or {}
        if address.get("department"):
            return address["department"]
    return None


def flatten_compiled_release(compiled_release: dict, row_source: str) -> dict[str, Any]:
    tender = compiled_release.get("tender") or {}
    buyer = compiled_release.get("buyer") or {}
    value = tender.get("value") or {}
    awards = compiled_release.get("awards") or []
    data_segmentation = compiled_release.get("dataSegmentation") or {}

    supplier_names = []
    for award in awards:
        for supplier in award.get("suppliers") or []:
            name = supplier.get("name")
            if name and name not in supplier_names:
                supplier_names.append(name)

    number_of_tenderers = tender.get("numberOfTenderers")
    has_award = len(awards) > 0
    is_single_bidder_awarded = has_award and number_of_tenderers == 1

    return {
        "ocid": compiled_release.get("ocid"),
        "buyer_id": buyer.get("id"),
        "buyer_name": buyer.get("name"),
        "buyer_department_raw": _find_buyer_department(compiled_release),
        "tender_title": tender.get("title"),
        "tender_description": tender.get("description"),
        "main_procurement_category": tender.get("mainProcurementCategory"),
        "procurement_method": tender.get("procurementMethod"),
        "procurement_method_details": tender.get("procurementMethodDetails"),
        "amount": value.get("amount"),
        "currency": value.get("currency"),
        "date_published": tender.get("datePublished"),
        "number_of_tenderers": number_of_tenderers,
        "num_awards": len(awards),
        "award_supplier_names": "; ".join(supplier_names) if supplier_names else None,
        "is_single_bidder_awarded": is_single_bidder_awarded,
        "data_segmentation_id": data_segmentation.get("id"),
        "row_source": row_source,
    }
