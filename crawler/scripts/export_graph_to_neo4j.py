from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from neo4j import GraphDatabase


from crawler.core.config import DB_PATH, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


# ---------- normalization / keys ----------
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\wäöüÄÖÜß\- ]+", re.UNICODE)

def norm_name(s: str) -> str:
    s = (s or "").strip()
    s = _WS.sub(" ", s)
    return s

def entity_key(name: str, etype: str, category: str) -> str:
    # stable merge key: normalized name + coarse type/category
    n = norm_name(name).lower()
    n = _PUNCT.sub("", n)
    n = _WS.sub(" ", n).strip()
    return f"{etype}::{category}::{n}"


# ---------- Neo4j schema ----------
SCHEMA_CYPHER = [
    "CREATE CONSTRAINT municipality_id IF NOT EXISTS FOR (m:Municipality) REQUIRE m.id IS UNIQUE",
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT segment_rowid IF NOT EXISTS FOR (s:Segment) REQUIRE s.rowid IS UNIQUE",
    "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    "CREATE INDEX entity_category IF NOT EXISTS FOR (e:Entity) ON (e.category)",
    "CREATE INDEX entity_status IF NOT EXISTS FOR (e:Entity) ON (e.status)",
    "CREATE INDEX segment_heading IF NOT EXISTS FOR (s:Segment) ON (s.heading_context)",
]

ALLOWED_REL_TYPES = {
    "FÖRDERT",
    "BAUT",
    "BESCHLIESST",
    "IMPLEMENTIERT",
    "PLANT",
    "GEHÖRT_ZU",
    "KOOPERIERT_MIT",
    "FINANZIERT_DURCH",
    "BEZIEHT_SICH_AUF",
}

def safe_rel_type(t: str) -> str:
    t = (t or "").strip().upper()
    # Neo4j relationship types cannot contain spaces; keep your German tokens
    t = t.replace(" ", "_")
    return t if t in ALLOWED_REL_TYPES else "BEZIEHT_SICH_AUF"


# ---------- read from SQLite ----------
@dataclass(frozen=True)
class TripletRow:
    municipality_id: str
    document_id: str
    segment_rowid: int
    graph_json: str

def iter_graph_triplets(conn: sqlite3.Connection, limit: Optional[int] = None) -> Iterable[TripletRow]:
    sql = """
    SELECT municipality_id, document_id, segment_rowid, graph_json
    FROM graph_triplets
    ORDER BY extracted_at ASC
    """
    if limit is not None:
        sql += " LIMIT ?"
        cur = conn.execute(sql, (int(limit),))
    else:
        cur = conn.execute(sql)
    for r in cur.fetchall():
        yield TripletRow(
            municipality_id=str(r[0]),
            document_id=str(r[1]),
            segment_rowid=int(r[2]),
            graph_json=str(r[3]),
        )

def get_segment_meta(
    conn: sqlite3.Connection, segment_rowid: int
) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """Returns (impact_score, text_snippet, heading_context) for a segment."""
    cur = conn.execute(
        "SELECT COALESCE(impact_score,0), text, heading_context FROM segments WHERE rowid=? LIMIT 1",
        (int(segment_rowid),),
    )
    row = cur.fetchone()
    if not row:
        return None, None, None
    score = int(row[0]) if row[0] is not None else None
    text = str(row[1]) if row[1] is not None else None
    heading_ctx = str(row[2]) if row[2] is not None else None
    return score, text, heading_ctx


# ---------- Neo4j upsert queries ----------
UPSERT_BASE = """
MERGE (m:Municipality {id: $muni_id})
MERGE (d:Document {id: $doc_id})
MERGE (s:Segment {rowid: $seg_rowid})
SET s.impact_score = $impact_score,
    s.text = $seg_text,
    s.heading_context = $heading_context
MERGE (m)-[:HAS_DOCUMENT]->(d)
MERGE (d)-[:HAS_SEGMENT]->(s)
"""

UPSERT_ENTITY = """
MERGE (e:Entity {key: $key})
ON CREATE SET
  e.name = $name,
  e.type = $type,
  e.category = $category,
  e.status = $status,
  e.metrics = $metrics
ON MATCH SET
  e.name = coalesce(e.name, $name),
  e.type = coalesce(e.type, $type),
  e.category = coalesce(e.category, $category),
  e.status = coalesce(e.status, $status),
  e.metrics = CASE
    WHEN e.metrics IS NULL THEN $metrics
    ELSE apoc.map.merge(e.metrics, $metrics)
  END
RETURN e.key AS key
"""

# NOTE: requires APOC for apoc.map.merge. If you don't have APOC, remove metrics merge logic.
# If you want "no APOC" version, tell me and I’ll swap it.

MENTION_EDGE = """
MATCH (s:Segment {rowid: $seg_rowid})
MATCH (e:Entity {key: $e_key})
MERGE (s)-[r:MENTIONS]->(e)
SET r.weight = coalesce(r.weight, 0) + 1
"""

def rel_edge_query(rel_type: str) -> str:
    # dynamic relationship type
    return f"""
MATCH (src:Entity {{key: $src_key}})
MATCH (dst:Entity {{key: $dst_key}})
MATCH (s:Segment {{rowid: $seg_rowid}})
MERGE (src)-[r:{rel_type}]->(dst)
ON CREATE SET r.evidence = $evidence, r.segment_rowid = $seg_rowid
ON MATCH SET r.evidence = coalesce(r.evidence, $evidence)
MERGE (s)-[:EVIDENCE_FOR]->(r)
"""


def _load_canonical_map(conn: sqlite3.Connection) -> Dict[str, str]:
    """Load entity_canonical table (from resolve_entities.py) → {original_name → canonical_name}."""
    try:
        cur = conn.execute("SELECT original_name, canonical_name FROM entity_canonical")
        return {str(r[0]): str(r[1]) for r in cur.fetchall()}
    except Exception:
        return {}


def main(limit: Optional[int] = None):
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found: {DB_PATH}")

    # SQLite
    conn = sqlite3.connect(str(DB_PATH), timeout=60.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=60000;")

    # Load entity resolution map (if available)
    canonical_map = _load_canonical_map(conn)
    if canonical_map:
        print(f"[neo4j] Loaded {len(canonical_map)} entity canonical mappings")

    # Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        # Schema
        for stmt in SCHEMA_CYPHER:
            session.run(stmt)

        # Optional: ensure APOC exists if you keep the metrics merge (otherwise remove UPSERT_ENTITY usage of apoc)
        # session.run("RETURN apoc.version()")

        n_rows = 0
        n_entities = 0
        n_rels = 0

        for row in iter_graph_triplets(conn, limit=limit):
            n_rows += 1

            impact_score, seg_text, heading_ctx = get_segment_meta(conn, row.segment_rowid)

            # base nodes/edges
            session.run(
                UPSERT_BASE,
                muni_id=row.municipality_id,
                doc_id=row.document_id,
                seg_rowid=int(row.segment_rowid),
                impact_score=int(impact_score) if impact_score is not None else 0,
                seg_text=seg_text[:2000] if seg_text else None,  # cap to keep graph lean
                heading_context=heading_ctx,
            )

            try:
                kg = json.loads(row.graph_json)
            except Exception:
                continue

            entities = kg.get("entities", []) or []
            rels = kg.get("relationships", []) or []

            # upsert entities + mention edges (apply entity resolution if available)
            key_by_name: Dict[str, str] = {}
            for e in entities:
                raw_name = norm_name(e.get("name", ""))
                name = canonical_map.get(raw_name, raw_name)  # resolve to canonical
                etype = e.get("type", "Dokument")
                cat = e.get("category", "Sonstiges")
                status = e.get("status", "Unbekannt")
                metrics = e.get("metrics", {}) or {}

                k = entity_key(name, etype, cat)
                key_by_name[name] = k

                session.run(
                    UPSERT_ENTITY,
                    key=k,
                    name=name,
                    type=etype,
                    category=cat,
                    status=status,
                    metrics=metrics,
                )
                session.run(MENTION_EDGE, seg_rowid=int(row.segment_rowid), e_key=k)
                n_entities += 1

            # relationships
            for r in rels:
                src_name = norm_name(r.get("source_entity", ""))
                dst_name = norm_name(r.get("target_entity", ""))
                rt = safe_rel_type(r.get("relation_type", "BEZIEHT_SICH_AUF"))
                evidence = norm_name(r.get("evidence", ""))[:500]

                # map by normalized names present in entity list; if missing, create minimal nodes
                src_key = key_by_name.get(src_name) or entity_key(src_name, "Dokument", "Sonstiges")
                dst_key = key_by_name.get(dst_name) or entity_key(dst_name, "Dokument", "Sonstiges")

                # ensure nodes exist (minimal)
                session.run(
                    UPSERT_ENTITY,
                    key=src_key,
                    name=src_name,
                    type="Dokument",
                    category="Sonstiges",
                    status="Unbekannt",
                    metrics={},
                )
                session.run(
                    UPSERT_ENTITY,
                    key=dst_key,
                    name=dst_name,
                    type="Dokument",
                    category="Sonstiges",
                    status="Unbekannt",
                    metrics={},
                )

                session.run(
                    rel_edge_query(rt),
                    src_key=src_key,
                    dst_key=dst_key,
                    seg_rowid=int(row.segment_rowid),
                    evidence=evidence,
                )
                n_rels += 1

            if n_rows % 50 == 0:
                print(f"[neo4j] processed {n_rows} triplets...")

        print(f"✅ Done. triplets={n_rows}, entities_upserts~={n_entities}, rel_upserts~={n_rels}")

        # Export document link graph (LINKS_TO edges)
        n_links = 0
        try:
            cur = conn.execute(
                """
                SELECT dl.source_document_id, dl.target_document_id, dl.anchor_text
                FROM document_links dl
                WHERE dl.target_document_id IS NOT NULL
                """
            )
            for src_doc, tgt_doc, anchor in cur.fetchall():
                session.run(
                    """
                    MATCH (src:Document {id: $src_id})
                    MATCH (tgt:Document {id: $tgt_id})
                    MERGE (src)-[r:LINKS_TO]->(tgt)
                    SET r.anchor_text = $anchor
                    """,
                    src_id=str(src_doc),
                    tgt_id=str(tgt_doc),
                    anchor=(anchor or "")[:300],
                )
                n_links += 1
            print(f"✅ Document links exported: {n_links} LINKS_TO edges")
        except Exception as e:
            print(f"⚠️  Document links export skipped: {e}")

    driver.close()
    conn.close()


if __name__ == "__main__":
    # set a small limit first, then remove
    main(limit=500)