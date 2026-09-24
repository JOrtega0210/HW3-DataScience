"""Validación de calidad y normalización territorial (Tarea 2, Fase 2).

Detecta (sin corregir a ciegas): registros repetidos por el mismo proceso real,
procesos sin monto o monto cero, procesos sin descripción, campos de ubicación
que no calzan con los 25 departamentos oficiales, e inconsistencias de encoding.

Sin dependencias de UI: solo stdlib + unidecode.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from unidecode import unidecode

MOJIBAKE_MARKERS = ("�", "Ã©", "Ã±", "Ã³", "Ã¡", "Ã­", "Ã\xba")


@dataclass
class QualityRule:
    rule: str
    flagged_count: int
    action_taken: str


def normalize_department(raw: str | None, canonical_departments: list[str]) -> str | None:
    """Empareja `raw` (tal como viene de buyer.address.department) contra los 25
    departamentos oficiales, sin distinguir mayúsculas/minúsculas ni tildes.
    Devuelve el nombre canónico (con tilde, como en config.yaml) o None si no
    hay match (queda para el reporte de "procesos no localizables")."""
    if not raw or not raw.strip():
        return None
    key = unidecode(raw).strip().upper()
    lookup = {unidecode(d).strip().upper(): d for d in canonical_departments}
    return lookup.get(key)


def detect_encoding_issues(rows: list[dict], text_fields: list[str]) -> list[str]:
    """Devuelve los ocids cuyos campos de texto muestran señales de mojibake
    (doble-encoding UTF-8/Latin-1 o caracteres de reemplazo)."""
    flagged = []
    for row in rows:
        combined = " ".join(str(row.get(f) or "") for f in text_fields)
        if any(marker in combined for marker in MOJIBAKE_MARKERS):
            flagged.append(row["ocid"])
    return flagged


def find_duplicate_groups(rows: list[dict]) -> dict[tuple, list[str]]:
    """Agrupa por (buyer_id, tender_title, date_published): mismo comprador,
    misma nomenclatura y misma fecha de convocatoria pero con ocids distintos
    es evidencia fuerte de que el sistema de origen (SEACE V3) republicó/corrigió
    el mismo proceso real bajo un id interno nuevo."""
    groups: dict[tuple, list[str]] = defaultdict(list)
    for row in rows:
        key = (row.get("buyer_id"), row.get("tender_title"), row.get("date_published"))
        groups[key].append(row["ocid"])
    return {k: v for k, v in groups.items() if len(set(v)) > 1}


def _completeness_score(row: dict) -> tuple:
    """Mayor score = registro más completo/reciente dentro de un grupo duplicado.
    Incluye el ocid como último criterio (desempate) para que el resultado sea
    determinístico entre corridas: `set()`/hashing de strings en Python no
    garantiza el mismo orden de iteración en dos procesos distintos, así que
    sin este desempate explícito el "sobreviviente" de un grupo duplicado podía
    cambiar de una corrida a otra sin que cambiaran los datos."""
    return (
        row.get("number_of_tenderers") is not None,
        row.get("num_awards", 0) > 0,
        row.get("award_supplier_names") is not None,
        row["ocid"],
    )


def resolve_duplicates(
    rows: list[dict], duplicate_groups: dict[tuple, list[str]]
) -> tuple[list[dict], list[dict]]:
    """Para cada grupo duplicado, conserva el registro más completo y descarta
    el resto del dataset principal (pero los deja trazados en `removed_log` con
    referencia al ocid que se conservó, para auditoría — nunca se borran en
    silencio)."""
    by_ocid = {r["ocid"]: r for r in rows}
    ocids_to_drop: set[str] = set()
    removed_log: list[dict] = []

    for key, ocids in duplicate_groups.items():
        candidates = [by_ocid[o] for o in dict.fromkeys(ocids) if o in by_ocid]
        if len(candidates) < 2:
            continue
        candidates.sort(key=_completeness_score, reverse=True)
        kept = candidates[0]
        for dropped in candidates[1:]:
            ocids_to_drop.add(dropped["ocid"])
            removed_log.append(
                {
                    "dropped_ocid": dropped["ocid"],
                    "kept_ocid": kept["ocid"],
                    "buyer_id": key[0],
                    "tender_title": key[1],
                    "date_published": key[2],
                }
            )

    deduped_rows = [r for r in rows if r["ocid"] not in ocids_to_drop]
    return deduped_rows, removed_log


def build_quality_report(
    rows_before: list[dict],
    rows_after: list[dict],
    duplicate_removed_count: int,
    unmatched_department_count: int,
    encoding_issue_count: int,
) -> list[QualityRule]:
    n = len(rows_before)
    no_amount = sum(1 for r in rows_before if r.get("amount") is None)
    zero_amount = sum(1 for r in rows_before if r.get("amount") == 0)
    no_description = sum(
        1 for r in rows_before if not (r.get("tender_description") or "").strip()
    )

    return [
        QualityRule(
            rule="registros_repetidos_mismo_proceso",
            flagged_count=duplicate_removed_count,
            action_taken=(
                f"se conservó el registro más completo por grupo y se descartó el resto "
                f"({duplicate_removed_count} de {n} filas); detalle en duplicates_removed.csv"
            ),
        ),
        QualityRule(
            rule="proceso_sin_monto",
            flagged_count=no_amount,
            action_taken="ninguna (0 casos); se deja el campo null si volviera a ocurrir" if no_amount == 0 else "se conserva la fila, amount=null; se excluye de agregaciones monetarias en el dashboard",
        ),
        QualityRule(
            rule="proceso_monto_cero",
            flagged_count=zero_amount,
            action_taken="se conserva la fila (monto 0 es válido en catálogos/convenios); se documenta para no inflar promedios sin advertencia",
        ),
        QualityRule(
            rule="proceso_sin_descripcion",
            flagged_count=no_description,
            action_taken="ninguna (0 casos); si ocurriera, se excluiría del índice RAG híbrido (Fase 3) por falta de texto para embeber",
        ),
        QualityRule(
            rule="ubicacion_no_es_departamento",
            flagged_count=unmatched_department_count,
            action_taken="se deja buyer_department=null y se reporta en unmatched_locations.csv (no se descarta la fila)",
        ),
        QualityRule(
            rule="inconsistencia_encoding_acentos",
            flagged_count=encoding_issue_count,
            action_taken="ninguna (0 casos detectados en tender_title/tender_description/buyer_name)" if encoding_issue_count == 0 else "se marca is_encoding_issue=true para revisión manual, no se intenta corregir automáticamente",
        ),
    ]
