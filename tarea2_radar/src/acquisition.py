"""Adquisición de datos del portal OECE (Tarea 2, Fase 1).

Dos mecanismos, ambos re-ejecutables sin duplicar descargas:

1. Descargas masivas mensuales (histórico): GET /api/v1/file/{source}/{type}/{year}/{month}
   -> zip con un recordPackage OCDS (records[].compiledRelease). Sin autenticación.
2. API de actualizaciones recientes: GET /api/v1/records?startDate=...&endDate=...
   con paginación tradicional (from/size), throttling y reintentos.

Sin dependencias de UI: solo requests/zipfile/json + stdlib.
"""

from __future__ import annotations

import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass
class AcquisitionLogEntry:
    timestamp: str
    action: str
    url: str
    status_code: int | None
    bytes_downloaded: int
    duration_seconds: float
    note: str = ""


class AcquisitionLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.write_text(
                "timestamp,action,url,status_code,bytes_downloaded,duration_seconds,note\n",
                encoding="utf-8",
            )

    def log(self, entry: AcquisitionLogEntry) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            note = entry.note.replace(",", ";")
            f.write(
                f"{entry.timestamp},{entry.action},{entry.url},{entry.status_code},"
                f"{entry.bytes_downloaded},{entry.duration_seconds:.3f},{note}\n"
            )


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def download_bulk_month(
    year: str,
    month: str,
    config: dict,
    logger: AcquisitionLogger,
) -> Path:
    """Descarga (si no existe ya) el zip mensual y lo extrae. Idempotente: si el
    JSON ya fue extraído, no vuelve a descargar ni a pegarle a la red."""
    source = config["data_source"]["source_id"]
    file_type = config["data_source"]["file_type"]
    url = config["data_source"]["bulk_file_url_template"].format(
        source=source, type=file_type, year=year, month=month
    )

    raw_dir = Path(config["paths"]["raw_dir"])
    zip_path = raw_dir / f"{year}-{month}_{source}_{file_type}.zip"
    extract_dir = raw_dir / "extracted" / f"{year}-{month}"

    if extract_dir.exists() and any(extract_dir.iterdir()):
        logger.log(
            AcquisitionLogEntry(
                _now_iso(), "download_bulk_month", url, None, 0, 0.0,
                note=f"ya existe en {extract_dir}, se omite descarga",
            )
        )
        return extract_dir

    raw_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    resp = requests.get(url, timeout=config["data_source"].get("timeout_seconds", 60))
    resp.raise_for_status()
    zip_path.write_bytes(resp.content)
    duration = time.time() - t0

    logger.log(
        AcquisitionLogEntry(
            _now_iso(), "download_bulk_month", url, resp.status_code,
            len(resp.content), duration,
        )
    )

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    return extract_dir


def fetch_recent_updates(
    start_date: str,
    end_date: str,
    config: dict,
    logger: AcquisitionLogger,
) -> list[dict]:
    """Trae records recientes vía la API, con throttling y reintentos.

    Sigue `links.next` tal como lo devuelve la API en vez de reconstruir el offset
    manualmente: se verificó con curl que el parámetro `size` no se respeta (la API
    siempre pagina de 20 en 20 sin importar el tamaño pedido), así que cortar el
    loop con `len(records) < size` pedido trunca resultados en silencio. Cada
    página se cachea en disco (keyed por número de página) para poder reanudar una
    corrida interrumpida sin repetir requests ya hechos.
    """
    api_cfg = config["data_source"]["api"]
    base_url = api_cfg["base_url"]
    endpoint = api_cfg["records_endpoint"]
    size = api_cfg["max_page_size"]
    throttle = api_cfg["throttle_seconds"]
    max_retries = api_cfg["max_retries"]
    timeout = api_cfg["timeout_seconds"]

    cache_dir = Path(api_cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    next_url = f"{base_url}{endpoint}"
    next_params = {
        "startDate": start_date,
        "endDate": end_date,
        "sourceId": config["data_source"]["source_id"],
        "from": 0,
        "size": size,
    }
    page_num = 0

    while next_url:
        cache_key = f"{start_date}_{end_date}_page{page_num}.json"
        cache_path = cache_dir / cache_key

        if cache_path.exists():
            page = json.loads(cache_path.read_text(encoding="utf-8"))
            logger.log(
                AcquisitionLogEntry(
                    _now_iso(), "fetch_recent_updates_cached", cache_key, None, 0, 0.0,
                    note="leído de caché, sin request a la API",
                )
            )
        else:
            page = None
            for attempt in range(1, max_retries + 1):
                t0 = time.time()
                try:
                    resp = requests.get(next_url, params=next_params, timeout=timeout)
                    resp.raise_for_status()
                    page = resp.json()
                    duration = time.time() - t0
                    logger.log(
                        AcquisitionLogEntry(
                            _now_iso(), "fetch_recent_updates", resp.url, resp.status_code,
                            len(resp.content), duration, note=f"pagina {page_num}, intento {attempt}",
                        )
                    )
                    break
                except (requests.RequestException, ValueError) as exc:
                    duration = time.time() - t0
                    logger.log(
                        AcquisitionLogEntry(
                            _now_iso(), "fetch_recent_updates_error", next_url, None, 0,
                            duration, note=f"pagina {page_num}, intento {attempt}/{max_retries}: {exc}",
                        )
                    )
                    if attempt == max_retries:
                        raise
                    time.sleep(throttle * attempt)  # backoff simple

            cache_path.write_text(json.dumps(page, ensure_ascii=False), encoding="utf-8")
            time.sleep(throttle)

        records = page.get("records", [])
        if not records:
            break
        all_records.extend(records)
        page_num += 1

        next_link = (page.get("links") or {}).get("next")
        if not next_link:
            break
        next_url = next_link
        next_params = None  # los params ya vienen embebidos en la URL de "next"

    return all_records
