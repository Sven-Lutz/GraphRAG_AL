"""
generate_embeddings.py
======================
Stage 2a of the GraphRAG pipeline.

Computes vector embeddings for every chunk using a local SentenceTransformers
model (privacy-compliant, no API calls) and stores them in two places:

  1. SQLite  – chunks.embedding  (raw float32 bytes, for local FAISS search)
  2. Neo4j   – Chunk.embedding   (as float list property) + vector index
               so that semantic similarity queries run natively inside Neo4j

Recommended model: paraphrase-multilingual-MiniLM-L12-v2
  - 384 dimensions, multilingual (covers German)
  - Fits comfortably in 4 GB VRAM / 8 GB RAM
  - max_seq_length = 256 tokens (chunks ≤600 tok are truncated; still good)

For better quality (if GPU available): intfloat/multilingual-e5-large (1024-dim)

Usage:
    python -m crawler.scripts.generate_embeddings
    python -m crawler.scripts.generate_embeddings --model intfloat/multilingual-e5-large --batch 64
    python -m crawler.scripts.generate_embeddings --neo4j   # also push to Neo4j
"""
from __future__ import annotations

import argparse
import sqlite3
import struct
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from crawler.core.config import (
    DB_PATH, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    EMBED_MODEL, EMBED_BATCH_SIZE,
)

DEFAULT_MODEL = EMBED_MODEL
DEFAULT_BATCH  = EMBED_BATCH_SIZE


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def vec_to_blob(vec: np.ndarray) -> bytes:
    """Pack float32 numpy array to raw bytes for SQLite BLOB storage."""
    return struct.pack(f"{len(vec)}f", *vec.astype(np.float32).tolist())


def blob_to_vec(blob: bytes) -> np.ndarray:
    """Unpack SQLite BLOB back to float32 numpy array."""
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

def load_model(model_name: str):
    """Load SentenceTransformer model with automatic device selection."""
    import torch
    from sentence_transformers import SentenceTransformer

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"   🔧 Modell: {model_name}  |  Device: {device}")
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = 256
    return model


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

def get_unembed_chunks(
    conn: sqlite3.Connection, limit: int = 10_000
) -> list[tuple[str, str]]:
    """Return (chunk_id, text) for chunks without an embedding yet."""
    rows = conn.execute(
        """
        SELECT chunk_id, text FROM chunks
        WHERE embedding IS NULL
        ORDER BY impact_score DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [(str(r[0]), str(r[1])) for r in rows]


def store_embeddings_sqlite(
    conn: sqlite3.Connection,
    chunk_ids: list[str],
    embeddings: np.ndarray,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (vec_to_blob(embeddings[i]), now, chunk_ids[i])
        for i in range(len(chunk_ids))
    ]
    conn.executemany(
        "UPDATE chunks SET embedding=?, embedded_at=? WHERE chunk_id=?",
        rows,
    )


# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------

def push_to_neo4j(
    conn: sqlite3.Connection,
    model_name: str,
    dim: int,
) -> None:
    """
    Export chunks with embeddings to Neo4j as Chunk nodes with a vector index.
    Requires Neo4j 5.11+ (native vector index support).
    """
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        # Create vector index (idempotent)
        session.run(
            f"""
            CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
        )
        # Also ensure unique constraint
        session.run(
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE"
        )

        # Fetch all embedded chunks from SQLite
        rows = conn.execute(
            """
            SELECT c.chunk_id, c.document_id, c.municipality_id,
                   c.text, c.heading_context, c.impact_score,
                   c.chunk_index, c.token_count, c.embedding
            FROM chunks c
            WHERE c.embedding IS NOT NULL
            """
        ).fetchall()

        n = 0
        batch_size = 100
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            params_list = []
            for row in batch:
                chunk_id, doc_id, muni_id, text, heading_ctx, score, \
                    chunk_idx, tok_count, blob = row
                vec = blob_to_vec(blob).tolist()
                params_list.append({
                    "chunk_id":       str(chunk_id),
                    "doc_id":         str(doc_id),
                    "muni_id":        str(muni_id),
                    "text":           str(text)[:4000],
                    "heading_context": str(heading_ctx) if heading_ctx else None,
                    "impact_score":   int(score) if score else 0,
                    "chunk_index":    int(chunk_idx),
                    "token_count":    int(tok_count),
                    "embedding":      vec,
                })

            session.run(
                """
                UNWIND $rows AS r
                MERGE (c:Chunk {chunk_id: r.chunk_id})
                SET c.text            = r.text,
                    c.heading_context = r.heading_context,
                    c.impact_score    = r.impact_score,
                    c.chunk_index     = r.chunk_index,
                    c.token_count     = r.token_count,
                    c.embedding       = r.embedding
                WITH c, r
                MATCH (d:Document {id: r.doc_id})
                MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                {"rows": params_list},
            )
            n += len(batch)
            print(f"   [neo4j] {n}/{len(rows)} Chunks exportiert…")

    driver.close()
    print(f"✅ Neo4j: {n} Chunks mit Vektor-Index exportiert (dim={dim})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH,
    limit: int = 50_000,
    push_neo4j: bool = False,
) -> None:
    model = load_model(model_name)
    conn  = sqlite3.connect(str(DB_PATH), timeout=60.0, isolation_level=None)

    total_embedded = 0

    while True:
        rows = get_unembed_chunks(conn, limit=min(limit, batch_size * 10))
        if not rows:
            break

        chunk_ids = [r[0] for r in rows]
        texts     = [r[1] for r in rows]

        print(f"⚙️  Embedding {len(texts)} Chunks…")
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        with conn:
            store_embeddings_sqlite(conn, chunk_ids, embeddings)

        total_embedded += len(chunk_ids)
        print(f"   ✅ {total_embedded} Chunks embedded (SQLite)")

        if total_embedded >= limit:
            break

    dim = model.get_sentence_embedding_dimension()
    print(f"\n🏁 Embedding abgeschlossen. Dimensionen: {dim}  |  Gesamt: {total_embedded}")

    if push_neo4j:
        print("\n📤 Exportiere nach Neo4j…")
        push_to_neo4j(conn, model_name, dim)

    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SentenceTransformer Embedding (Stage 2a)")
    ap.add_argument("--model",    default=DEFAULT_MODEL, help="SentenceTransformers Modell-Name")
    ap.add_argument("--batch",    type=int, default=DEFAULT_BATCH)
    ap.add_argument("--limit",    type=int, default=50_000, help="Max. Chunks pro Lauf")
    ap.add_argument("--neo4j",    action="store_true",     help="Chunks + Vektoren nach Neo4j exportieren")
    args = ap.parse_args()
    main(model_name=args.model, batch_size=args.batch, limit=args.limit, push_neo4j=args.neo4j)
