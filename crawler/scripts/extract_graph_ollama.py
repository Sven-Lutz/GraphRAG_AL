"""
extract_graph_ollama.py
=======================
Stage 2b of the GraphRAG pipeline — privacy-compliant, local-LLM variant.

Replaces the OpenAI-based extract_graph.py with a fully local pipeline:

  Ollama  →  local LLM (e.g. llama3.1:8b, mistral, qwen2.5)
  Pydantic → strict JSON schema enforcement to prevent import errors
  Gleaning → self-reflection pass that asks the model if any entities
             were missed ("Did you overlook anything?") — significantly
             increases extraction recall on dense municipal documents.

Graph output is written to the same `graph_triplets` table so the
existing Neo4j exporter (export_graph_to_neo4j.py) works unchanged.

MODELS tested (German-language performance, descending):
  - qwen2.5:14b          best quality, needs 10+ GB VRAM
  - llama3.1:8b          good quality, 6 GB VRAM
  - mistral:7b           decent, 5 GB VRAM
  - gemma2:9b            strong for structured output

Usage:
    python -m crawler.scripts.extract_graph_ollama
    python -m crawler.scripts.extract_graph_ollama --model qwen2.5:14b --gleaning 2
    python -m crawler.scripts.extract_graph_ollama --chunk-mode  # use chunks table
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from crawler.core.config import DB_PATH, OLLAMA_URL, OLLAMA_MODEL as _DEFAULT_MODEL

MODEL_NAME = _DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Pydantic schema (same as extract_graph.py for compatibility)
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    name: str
    type: Literal[
        "Akteur", "Infrastruktur", "Förderprogramm",
        "Maßnahme", "Konzept/Ziel", "Dokument",
    ]
    category: Literal["Mobilität", "Wärme", "Strom", "Finanzen", "Governance", "Sonstiges"]
    status: Literal["Geplant", "In Umsetzung", "Abgeschlossen", "Existierend", "Unbekannt"]
    metrics: dict[str, str] = Field(default_factory=dict)


class Relationship(BaseModel):
    source_entity: str
    relation_type: Literal[
        "FÖRDERT", "BAUT", "BESCHLIESST", "IMPLEMENTIERT",
        "PLANT", "GEHÖRT_ZU", "KOOPERIERT_MIT", "FINANZIERT_DURCH", "BEZIEHT_SICH_AUF",
    ]
    target_entity: str
    evidence: str


class KnowledgeGraph(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_EXTRACTION = """\
Du bist ein Experte für kommunale Klimaschutzpolitik in Bayern.
Extrahiere alle Entitäten und Beziehungen aus dem gegebenen Text.

WICHTIG: Antworte NUR mit einem JSON-Objekt im folgenden Format:
{
  "entities": [
    {"name": "...", "type": "Akteur|Infrastruktur|Förderprogramm|Maßnahme|Konzept/Ziel|Dokument",
     "category": "Mobilität|Wärme|Strom|Finanzen|Governance|Sonstiges",
     "status": "Geplant|In Umsetzung|Abgeschlossen|Existierend|Unbekannt",
     "metrics": {"year": "...", "amount": "...", "capacity": "..."}}
  ],
  "relationships": [
    {"source_entity": "...",
     "relation_type": "FÖRDERT|BAUT|BESCHLIESST|IMPLEMENTIERT|PLANT|GEHÖRT_ZU|KOOPERIERT_MIT|FINANZIERT_DURCH|BEZIEHT_SICH_AUF",
     "target_entity": "...",
     "evidence": "exakter Originaltext (max. 200 Zeichen)"}
  ]
}
Nur belegte Fakten. Kein Halluzinieren. Kein erklärender Text außerhalb des JSON.
"""

SYSTEM_GLEANING = """\
Du hast soeben Entitäten und Beziehungen aus einem Text extrahiert.
Überprüfe deine Extraktion kritisch:

Wurden Akteure (Behörden, Firmen, Vereine), Maßnahmen (Projekte, Investitionen),
Infrastrukturen (Anlagen, Netze) oder Förderprogramme (KfW, BAFA, NKI, EFRE) übersehen?

Falls ja, antworte mit einem JSON-Objekt mit NUR den NEUEN (bisher fehlenden) Entitäten
und Beziehungen im gleichen Schema wie zuvor.
Falls alle Entitäten vollständig sind, antworte mit: {"entities": [], "relationships": []}
"""


def _user_extraction_prompt(text: str, heading_context: Optional[str], url: Optional[str]) -> str:
    parts = []
    if heading_context:
        parts.append(f"ABSCHNITT: {heading_context}")
    if url:
        parts.append(f"QUELLE: {url}")
    parts.append(f"TEXT:\n{text}")
    return "\n\n".join(parts)


def _user_gleaning_prompt(text: str, first_extraction: KnowledgeGraph) -> str:
    entity_names = [e.name for e in first_extraction.entities]
    return (
        f"ORIGINALTEXT:\n{text[:800]}\n\n"
        f"BEREITS EXTRAHIERTE ENTITÄTEN: {json.dumps(entity_names, ensure_ascii=False)}\n\n"
        "Hast du relevante Entitäten oder Beziehungen übersehen?"
    )


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

def _call_ollama(model: str, system: str, user: str) -> str:
    """Call Ollama REST API; returns raw model response text."""
    import urllib.request

    payload = json.dumps({
        "model":    model,
        "messages": [
            {"role": "system",  "content": system},
            {"role": "user",    "content": user},
        ],
        "stream":   False,
        "options":  {"temperature": 0.0, "num_predict": 2048},
        "format":   "json",   # Ollama structured output mode
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"]


def _parse_kg(raw: str) -> KnowledgeGraph:
    """Parse JSON string → KnowledgeGraph, tolerating minor model quirks."""
    # Strip markdown code fences if model wraps output
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return KnowledgeGraph.model_validate_json(raw)
    except (ValidationError, json.JSONDecodeError):
        # Fallback: try to extract the JSON object manually
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            return KnowledgeGraph.model_validate_json(raw[start:end])
        return KnowledgeGraph()


def extract_with_gleaning(
    model: str,
    text: str,
    heading_context: Optional[str],
    url: Optional[str],
    gleaning_rounds: int = 1,
) -> KnowledgeGraph:
    """
    Two-pass extraction:
      1. Initial structured extraction
      2. Up to `gleaning_rounds` self-reflection passes that append missed entities
    """
    # --- Round 1: initial extraction ---
    user_prompt = _user_extraction_prompt(text, heading_context, url)
    raw = _call_ollama(model, SYSTEM_EXTRACTION, user_prompt)
    kg  = _parse_kg(raw)

    # --- Gleaning rounds: self-reflection ---
    for _ in range(gleaning_rounds):
        if not kg.entities:
            break  # nothing to reflect on
        glean_prompt = _user_gleaning_prompt(text, kg)
        raw2 = _call_ollama(model, SYSTEM_GLEANING, glean_prompt)
        kg2  = _parse_kg(raw2)

        # Merge new findings, deduplicating by name
        existing_names = {e.name.lower() for e in kg.entities}
        for e in kg2.entities:
            if e.name.lower() not in existing_names:
                kg.entities.append(e)
                existing_names.add(e.name.lower())
        kg.relationships.extend(kg2.relationships)

    return kg


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def setup_db(cur: sqlite3.Cursor) -> None:
    cur.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA busy_timeout=60000;

        CREATE TABLE IF NOT EXISTS graph_triplets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            municipality_id   TEXT NOT NULL,
            document_id       TEXT NOT NULL,
            segment_rowid     INTEGER NOT NULL,
            segment_hash      TEXT NOT NULL,
            model_name        TEXT NOT NULL,
            graph_json        TEXT NOT NULL,
            entity_count      INTEGER NOT NULL DEFAULT 0,
            relationship_count INTEGER NOT NULL DEFAULT 0,
            extracted_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE UNIQUE INDEX IF NOT EXISTS uq_graph_triplets_seg
            ON graph_triplets(segment_rowid);
        CREATE INDEX IF NOT EXISTS idx_graph_triplets_muni
            ON graph_triplets(municipality_id);
        """
    )


def get_chunks_for_extraction(
    cur: sqlite3.Cursor,
    limit: int = 200,
    min_score: int = 15,
    min_len: int = 160,
    per_doc: int = 5,
    use_chunks_table: bool = False,
) -> list[dict]:
    """Return rows to extract.  Can pull from either `segments` or `chunks`."""
    if use_chunks_table:
        query = """
        WITH ranked AS (
          SELECT
            c.rowid         AS rowid,
            c.chunk_index   AS order_index,
            c.municipality_id,
            c.document_id,
            d.url_canonical,
            c.text,
            c.heading_context,
            COALESCE(c.impact_score, 0) AS impact_score,
            ROW_NUMBER() OVER (
              PARTITION BY c.document_id
              ORDER BY COALESCE(c.impact_score, 0) DESC
            ) AS rn
          FROM chunks c
          JOIN documents_raw d ON d.document_id = c.document_id
          WHERE length(c.text) >= ?
            AND COALESCE(c.impact_score, 0) >= ?
            AND c.rowid NOT IN (SELECT segment_rowid FROM graph_triplets)
        )
        SELECT rowid, order_index, municipality_id, document_id, url_canonical,
               text, heading_context, impact_score
        FROM ranked WHERE rn <= ?
        ORDER BY impact_score DESC LIMIT ?
        """
    else:
        query = """
        WITH ranked AS (
          SELECT
            s.rowid         AS rowid,
            s.order_index,
            d.municipality_id,
            d.document_id,
            d.url_canonical,
            s.text,
            s.heading_context,
            COALESCE(s.impact_score, 0) AS impact_score,
            ROW_NUMBER() OVER (
              PARTITION BY s.document_id
              ORDER BY COALESCE(s.impact_score, 0) DESC
            ) AS rn
          FROM segments s
          JOIN documents_raw d ON d.document_id = s.document_id
          WHERE length(s.text) >= ?
            AND COALESCE(s.is_negative, 0) = 0
            AND COALESCE(s.impact_score, 0) >= ?
            AND s.rowid NOT IN (SELECT segment_rowid FROM graph_triplets)
        )
        SELECT rowid, order_index, municipality_id, document_id, url_canonical,
               text, heading_context, impact_score
        FROM ranked WHERE rn <= ?
        ORDER BY impact_score DESC LIMIT ?
        """
    cur.execute(query, (min_len, min_score, per_doc, limit))
    cols = ["rowid", "order_index", "municipality_id", "document_id",
            "url_canonical", "text", "heading_context", "impact_score"]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    model: str = MODEL_NAME,
    gleaning_rounds: int = 1,
    limit: int = 200,
    min_score: int = 15,
    use_chunks_table: bool = False,
    sleep_between: float = 0.0,
) -> None:
    conn = sqlite3.connect(str(DB_PATH), timeout=60.0, isolation_level=None)
    cur  = conn.cursor()
    setup_db(cur)

    # Build canonical name map from resolve_entities output (if available)
    canonical_map: dict[str, str] = {}
    try:
        for raw_name, canon_name in cur.execute(
            "SELECT original_name, canonical_name FROM entity_canonical"
        ).fetchall():
            canonical_map[raw_name.lower()] = canon_name
    except Exception:
        pass  # table doesn't exist yet — run resolve_entities.py first

    rows = get_chunks_for_extraction(
        cur, limit=limit, min_score=min_score, use_chunks_table=use_chunks_table
    )
    if not rows:
        print("✅ Keine neuen Texte für Extraktion.")
        conn.close()
        return

    mode = "chunks" if use_chunks_table else "segments"
    print(f"🚀 Extrahiere Graph via Ollama ({model}) für {len(rows)} {mode} | gleaning={gleaning_rounds}\n")

    for row in rows:
        rowid    = int(row["rowid"])
        text     = row["text"]
        muni     = row["municipality_id"]
        doc      = row["document_id"]
        url      = row["url_canonical"]
        heading  = row["heading_context"]
        score    = row["impact_score"]

        print(f"⚙️  muni={muni}  rowid={rowid}  score={score}")
        if heading:
            print(f"   📂 {heading}")

        try:
            kg = extract_with_gleaning(
                model=model,
                text=text,
                heading_context=heading,
                url=url,
                gleaning_rounds=gleaning_rounds,
            )
            # Apply canonical name resolution before persisting
            if canonical_map:
                for ent in kg.entities:
                    ent.name = canonical_map.get(ent.name.lower(), ent.name)
                for rel in kg.relationships:
                    rel.source_entity = canonical_map.get(rel.source_entity.lower(), rel.source_entity)
                    rel.target_entity = canonical_map.get(rel.target_entity.lower(), rel.target_entity)
            graph_json = kg.model_dump_json(ensure_ascii=False)
            e, r = len(kg.entities), len(kg.relationships)
            seg_hash = hashlib.sha256((text or "").encode()).hexdigest()

            cur.execute(
                """
                INSERT OR IGNORE INTO graph_triplets (
                    municipality_id, document_id, segment_rowid, segment_hash,
                    model_name, graph_json, entity_count, relationship_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (muni, doc, rowid, seg_hash, model, graph_json, e, r),
            )
            print(f"   🟢 entities={e}  relationships={r}")

        except Exception as ex:
            print(f"   🔴 Fehler: {ex}")

        if sleep_between > 0:
            time.sleep(sleep_between)

    conn.close()
    print("\n🏁 Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ollama Graph Extraktion mit Gleaning (Stage 2b)")
    ap.add_argument("--model",       default=MODEL_NAME)
    ap.add_argument("--gleaning",    type=int,   default=1,   help="Anzahl Gleaning-Runden")
    ap.add_argument("--limit",       type=int,   default=200)
    ap.add_argument("--min-score",   type=int,   default=15)
    ap.add_argument("--chunk-mode",  action="store_true",     help="Aus chunks-Tabelle statt segments lesen")
    ap.add_argument("--sleep",       type=float, default=0.0, help="Pause zwischen Anfragen (Sekunden)")
    args = ap.parse_args()
    main(
        model=args.model,
        gleaning_rounds=args.gleaning,
        limit=args.limit,
        min_score=args.min_score,
        use_chunks_table=args.chunk_mode,
        sleep_between=args.sleep,
    )
