"""
chunker.py
==========
Token-aware text chunking using tiktoken.

Parameters (from the research specification):
  - chunk_size:  600 tokens  (balances context richness vs. LLM window cost)
  - overlap:     60 tokens   (middle of the 40-100 range; prevents "Lost in the Middle"
                              by ensuring cross-boundary entities appear in both chunks)

The chunker operates on raw text strings and returns (chunk_text, start_token, end_token)
triples so that provenance back to the original document is always preserved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

import tiktoken

ENCODING_NAME = "cl100k_base"   # used by GPT-4 / text-embedding-3-*
DEFAULT_CHUNK_SIZE = 600        # tokens per chunk
DEFAULT_OVERLAP = 60            # token overlap between consecutive chunks


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    start_token: int    # inclusive, relative to the full document token list
    end_token: int      # exclusive
    chunk_index: int    # 0-based position in the chunk sequence


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    encoding_name: str = ENCODING_NAME,
) -> list[Chunk]:
    """
    Split *text* into overlapping token windows.

    Returns an empty list for empty / whitespace-only input.
    The last chunk may be shorter than chunk_size.
    """
    if not text or not text.strip():
        return []

    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)

    if len(tokens) == 0:
        return []

    stride = chunk_size - overlap
    if stride <= 0:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

    chunks: list[Chunk] = []
    idx = 0
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(
            Chunk(
                text=enc.decode(chunk_tokens),
                start_token=start,
                end_token=end,
                chunk_index=idx,
            )
        )
        idx += 1
        if end == len(tokens):
            break
        start += stride

    return chunks


def iter_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    encoding_name: str = ENCODING_NAME,
) -> Generator[Chunk, None, None]:
    """Lazy generator variant — use for very long documents to avoid large lists."""
    if not text or not text.strip():
        return

    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)

    if not tokens:
        return

    stride = chunk_size - overlap
    if stride <= 0:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

    idx = 0
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        yield Chunk(
            text=enc.decode(tokens[start:end]),
            start_token=start,
            end_token=end,
            chunk_index=idx,
        )
        idx += 1
        if end == len(tokens):
            break
        start += stride


def count_tokens(text: str, encoding_name: str = ENCODING_NAME) -> int:
    """Return the number of tokens in *text* without chunking."""
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text or ""))
