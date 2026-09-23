"""Chunking de texto limpio con metadata estable (Tarea 1, Fase 2).

Usa el tokenizer real del modelo de embeddings (no una aproximación por caracteres)
para que el tamaño de chunk configurado en tokens sea exacto, y para poder verificar
el límite de tokens del modelo (`max_seq_length`) antes de indexar.

Sin dependencias de UI: solo `transformers` (tokenizer) + stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoTokenizer

_tokenizer_cache: dict[str, "PreTrainedTokenizerFast"] = {}


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    version: str
    page_number: int
    config_name: str
    token_count: int
    char_start: int
    char_end: int
    text: str


def get_tokenizer(model_name: str):
    """Carga (y cachea en memoria) el tokenizer real del modelo de embeddings.

    No requiere torch/tensorflow: solo se usa la tokenización, no la red neuronal.
    """
    if model_name not in _tokenizer_cache:
        tok = AutoTokenizer.from_pretrained(model_name)
        if not tok.is_fast:
            raise RuntimeError(
                f"El tokenizer de {model_name} no es 'fast' (no soporta offset_mapping), "
                "necesario para mapear cada chunk a caracteres exactos del texto original."
            )
        # Se tokeniza la página completa antes de partirla en ventanas: el límite real
        # de tokens del modelo (max_tokens en config.yaml) se aplica manualmente al
        # armar cada chunk, así que se silencia el warning genérico del tokenizer.
        tok.model_max_length = 10**9
        _tokenizer_cache[model_name] = tok
    return _tokenizer_cache[model_name]


def chunk_page_text(
    text: str,
    tokenizer,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> list[tuple[int, int, int]]:
    """Divide el texto de una página en ventanas de tokens con overlap.

    Devuelve (char_start, char_end, token_count) por ventana, usando el
    offset_mapping del tokenizer para recortar el texto ORIGINAL (evita artefactos
    de un decode() de vuelta desde subwords, ej. con el marcador '▁' de sentencepiece).
    """
    if overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens debe ser menor que chunk_size_tokens")

    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]
    n = len(offsets)
    if n == 0:
        return []

    step = chunk_size_tokens - overlap_tokens
    spans: list[tuple[int, int, int]] = []
    start = 0
    while start < n:
        end = min(start + chunk_size_tokens, n)
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        spans.append((char_start, char_end, end - start))
        if end == n:
            break
        start += step
    return spans


def chunk_document_pages(
    doc_id: str,
    version: str,
    pages: list[dict],
    tokenizer,
    config_name: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """`pages`: lista de dicts con page_number, clean_text, is_cover_page
    (formato de data/processed/clean/<doc_id>.jsonl)."""
    chunks: list[Chunk] = []
    for page in pages:
        if page.get("is_cover_page"):
            continue
        text = page["clean_text"]
        spans = chunk_page_text(text, tokenizer, chunk_size_tokens, overlap_tokens)
        for idx, (char_start, char_end, token_count) in enumerate(spans):
            chunk_id = f"{doc_id}:{config_name}:p{page['page_number']:03d}:c{idx:03d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    version=version,
                    page_number=page["page_number"],
                    config_name=config_name,
                    token_count=token_count,
                    char_start=char_start,
                    char_end=char_end,
                    text=text[char_start:char_end],
                )
            )
    return chunks
