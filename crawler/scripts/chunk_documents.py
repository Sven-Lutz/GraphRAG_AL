"""
chunk_documents.py
==================
Stage 1 of the GraphRAG pipeline.

Maps the relational SQLite schema to a graph-ready `chunks` table:

  Relational                →  Graph
  ─────────────────────────────────────────────────────
  documents_raw (node)      →  Document node
  segments     (rows)       →  text source for chunking
  chunks       (new table)  →  TextUnit / Chunk node
  document_links            →  LINKS_TO relationship

Each high-scoring segment is tokenised with tiktoken and split into
600-token windows with 60-token overlap (40-100 range per spec).
Short segments that already fit in one chunk are stored as-is.

The `chunks` table is the canonical input for:
  - generate_embeddings.py  (Stage 2: vector index)
  - extract_graph_ollama.py (Stage 2: entity extraction)

Usage:
    python -m crawler.scripts.chunk_documents
    python -m crawler.scripts.chunk_documents --min-score 10 --limit 50000
"""
from __future__ import annotations

import argparse
import hashlib
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from crawler.core.segmentation.chunker import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    chunk_text,
    count_tokens,
)

DB_PATH = Path("crawler/data/db/crawl.sqlite")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def setup_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA busy_timeout=60000;
        PRAGMA foreign_keys=ON;

        -- TextUnit / Chunk node (maps 1-to-many from segment)
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id          TEXT PRIMARY KEY,
            document_id       TEXT NOT NULL,
            municipality_id   TEXT NOT NULL,
            segment_id        TEXT,           -- source segment (nullable: direct-from-doc mode)
            chunk_index       INTEGER NOT NULL,
            chunk_hash        TEXT NOT NULL,
            text              TEXT NOT NULL,
            start_token       INTEGER NOT NULL,
            end_token         INTEGER NOT NULL,
            token_count       INTEGER NOT NULL,
            heading_context   TEXT,           -- inherited from source segment
            page_ref          TEXT,
            impact_score      INTEGER,
            embedding         BLOB,           -- float32 array, populated by generate_embeddings.py
            embedded_at       TEXT,
            created_at        TEXT NOT NULL,
            FOREIGN KEY(document_id) REFERENCES documents_raw(document_id) ON DELETE CASCADE
        );

        CREATE UNIQUE INDEX IF NOT EXISTS uq_chunks_hash
            ON chunks(document_id, chunk_hash);

        CREATE INDEX IF NOT EXISTS idx_chunks_doc
            ON chunks(document_id);

        CREATE INDEX IF NOT EXISTS idx_chunks_muni
            ON chunks(municipality_id);

        CREATE INDEX IF NOT EXISTS idx_chunks_score
            ON chunks(impact_score DESC);

        CREATE INDEX IF NOT EXISTS idx_chunks_no_embedding
            ON chunks(embedding) WHERE embedding IS NULL;
        """
    )


# ---------------------------------------------------------------------------
# Chunking logic
# ---------------------------------------------------------------------------

def process_segments(
    conn: sqlite3.Connection,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    min_score: int = 10,
    min_len: int = 80,
    limit: int = 100_000,
) -> tuple[int, int]:
    """
    Read segments from DB, chunk them, insert into `chunks` table.
    Returns (segments_processed, chunks_created).
    """
    query = """
        SELECT
            s.segment_id,
            s.document_id,
            d.municipality_id,
            s.text,
            s.heading_context,
            s.page_ref,
            COALESCE(s.impact_score, 0) AS impact_score
        FROM segments s
        JOIN documents_raw d ON d.document_id = s.document_id
        WHERE length(s.text) >= ?
          AND COALESCE(s.is_negative, 0) = 0
          AND COALESCE(s.impact_score, 0) >= ?
          AND s.segment_id NOT IN (
              SELECT DISTINCT segment_id FROM chunks WHERE segment_id IS NOT NULL
          )
        ORDER BY impact_score DESC
        LIMIT ?
    """
    rows = conn.execute(query, (min_len, min_score, limit)).fetchall()

    now = datetime.now(timezone.utc).isoformat()
    insert_sql = """
        INSERT OR IGNORE INTO chunks (
            chunk_id, document_id, municipality_id, segment_id,
            chunk_index, chunk_hash, text, start_token, end_token,
            token_count, heading_context, page_ref, impact_score, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    segs_done = 0
    chunks_created = 0

    for seg_id, doc_id, muni_id, text, heading_ctx, page_ref, score in rows:
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        batch = []
        for chunk in chunks:
            chunk_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
            batch.append((
                str(uuid.uuid4()),
                doc_id,
                muni_id,
                seg_id,
                chunk.chunk_index,
                chunk_hash,
                chunk.text,
                chunk.start_token,
                chunk.end_token,
                chunk.end_token - chunk.start_token,
                heading_ctx,
                page_ref,
                int(score),
                now,
            ))
        if batch:
            conn.executemany(insert_sql, batch)
            chunks_created += len(batch)
        segs_done += 1

    return segs_done, chunks_created


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    min_score: int = 10,
    limit: int = 100_000,
) -> None:
    conn = sqlite3.connect(str(DB_PATH), timeout=60.0, isolation_level=None)
    setup_db(conn)

    print(f"⚙️  Chunking segments  (size={chunk_size} tok, overlap={overlap} tok, min_score={min_score})")

    with conn:
        segs, chunks = process_segments(
            conn,
            chunk_size=chunk_size,
            overlap=overlap,
            min_score=min_score,
            limit=limit,
        )

    print(f"✅ {segs} Segmente verarbeitet → {chunks} Chunks erstellt")
    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Segment → Chunk Mapping (Stage 1 GraphRAG)")
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument("--overlap",    type=int, default=DEFAULT_OVERLAP)
    ap.add_argument("--min-score",  type=int, default=10)
    ap.add_argument("--limit",      type=int, default=100_000)
    args = ap.parse_args()
    main(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        min_score=args.min_score,
        limit=args.limit,
    )
