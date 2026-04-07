"""
summarize_documents.py
======================
Generates LLM-based per-document summaries and stores them in the
`document_summaries` table.  These summaries form the **community layer**
of the GraphRAG pipeline: they let a retriever answer questions at the
document level without scanning thousands of raw segments, and they provide
the coarse-grained context needed for multi-hop reasoning across municipalities.

Usage:
    python -m crawler.scripts.summarize_documents          # process all unsummarised docs
    python -m crawler.scripts.summarize_documents --limit 100
    python -m crawler.scripts.summarize_documents --min-score 15 --per-doc 8
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field
from openai import OpenAI

client = OpenAI()
MODEL_NAME = "gpt-4o-mini"
DB_PATH = Path("crawler/data/db/crawl.sqlite")


# ---------------------------------------------------------------------------
# Pydantic schema for the structured summary
# ---------------------------------------------------------------------------

class DocumentSummary(BaseModel):
    summary: str = Field(
        description=(
            "3-5 Sätze: Kernaussagen des Dokuments zu Klimaschutz, Energie oder Nachhaltigkeit. "
            "Wenn nicht klimarelevant: 1 Satz mit Begründung."
        )
    )
    document_type: Literal[
        "Klimaschutzkonzept",
        "Energiebericht",
        "Wärmeplanung",
        "Beschluss",
        "Haushalt",
        "Bauleitplanung",
        "Förderbescheid",
        "Bericht",
        "Sonstiges",
    ]
    key_topics: List[str] = Field(
        default_factory=list,
        description="Bis zu 5 Themenstichwörter, z.B. ['Photovoltaik', 'Wärmepumpe', 'KfW-Förderung']",
    )
    key_entities: List[str] = Field(
        default_factory=list,
        description="Bis zu 5 wichtige Akteure/Programme, z.B. ['Stadtwerke', 'BAFA', 'NKI']",
    )
    year_range: Optional[str] = Field(
        default=None,
        description="Zeitraum des Dokuments, z.B. '2023-2030' oder '2024'",
    )
    action_count: int = Field(
        default=0,
        description="Geschätzte Anzahl konkreter Maßnahmen/Projekte im Dokument",
    )


SYSTEM_MSG = """\
Du bist Experte für kommunale Klimaschutzpolitik in Bayern.
Fasse die wichtigsten Inhalte eines kommunalen Dokuments strukturiert zusammen.
Beziehe dich ausschließlich auf die gegebenen Textsegmente.
Sei präzise und faktenbasiert.
"""


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def setup_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA busy_timeout=60000;

        CREATE TABLE IF NOT EXISTS document_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            municipality_id TEXT NOT NULL,
            url_canonical TEXT,
            model_name TEXT NOT NULL,
            summary_text TEXT NOT NULL,
            document_type TEXT,
            key_topics TEXT,
            key_entities TEXT,
            year_range TEXT,
            action_count INTEGER DEFAULT 0,
            summarized_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS uq_doc_summaries
            ON document_summaries(document_id);
        CREATE INDEX IF NOT EXISTS idx_doc_summaries_muni
            ON document_summaries(municipality_id);
        """
    )


def get_unsummarised_docs(
    conn: sqlite3.Connection,
    limit: int = 200,
    min_score: int = 15,
    per_doc: int = 6,
    min_len: int = 150,
) -> list[dict]:
    """
    Return documents that have high-impact segments but no summary yet.
    For each document, aggregate the top-N segments into a single text block.
    """
    query = """
    WITH ranked AS (
        SELECT
            s.text,
            s.heading_context,
            s.order_index,
            d.document_id,
            d.municipality_id,
            d.url_canonical,
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
          AND d.document_id NOT IN (SELECT document_id FROM document_summaries)
    )
    SELECT document_id, municipality_id, url_canonical, text, heading_context, impact_score
    FROM ranked
    WHERE rn <= ?
    ORDER BY document_id, impact_score DESC
    LIMIT ?;
    """
    cur = conn.execute(query, (min_len, min_score, per_doc, limit * per_doc))
    rows = cur.fetchall()

    # Group by document
    docs: dict[str, dict] = {}
    for doc_id, muni_id, url, text, heading_ctx, score in rows:
        if doc_id not in docs:
            docs[doc_id] = {
                "document_id": doc_id,
                "municipality_id": muni_id,
                "url_canonical": url,
                "segments": [],
            }
        docs[doc_id]["segments"].append({
            "text": text,
            "heading_context": heading_ctx,
            "score": score,
        })

    result = list(docs.values())
    return result[:limit]


def build_user_prompt(doc: dict) -> str:
    lines: list[str] = []
    if doc.get("url_canonical"):
        lines.append(f"QUELLE: {doc['url_canonical']}\n")
    for seg in doc["segments"]:
        if seg.get("heading_context"):
            lines.append(f"[{seg['heading_context']}]")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines).strip()


def summarise(user_prompt: str) -> DocumentSummary:
    resp = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_prompt},
        ],
        response_format=DocumentSummary,
        temperature=0.0,
    )
    return resp.choices[0].message.parsed


def store_summary(conn: sqlite3.Connection, doc: dict, summary: DocumentSummary) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT OR REPLACE INTO document_summaries (
            document_id, municipality_id, url_canonical, model_name,
            summary_text, document_type, key_topics, key_entities,
            year_range, action_count, summarized_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            doc["document_id"],
            doc["municipality_id"],
            doc.get("url_canonical"),
            MODEL_NAME,
            summary.summary,
            summary.document_type,
            json.dumps(summary.key_topics, ensure_ascii=False),
            json.dumps(summary.key_entities, ensure_ascii=False),
            summary.year_range,
            summary.action_count,
            now,
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(limit: int = 200, min_score: int = 15, per_doc: int = 6) -> None:
    conn = sqlite3.connect(str(DB_PATH), timeout=60.0, isolation_level=None)
    setup_db(conn)

    docs = get_unsummarised_docs(conn, limit=limit, min_score=min_score, per_doc=per_doc)
    if not docs:
        print("✅ Alle Dokumente bereits zusammengefasst.")
        conn.close()
        return

    print(f"🚀 Erstelle Zusammenfassungen für {len(docs)} Dokumente\n")

    ok = 0
    for doc in docs:
        doc_id = doc["document_id"]
        muni_id = doc["municipality_id"]
        n_segs = len(doc["segments"])
        print(f"⚙️  muni={muni_id}  doc={doc_id[:8]}…  segments={n_segs}")
        try:
            user_prompt = build_user_prompt(doc)
            summary = summarise(user_prompt)
            store_summary(conn, doc, summary)
            print(
                f"   🟢 [{summary.document_type}]  topics={summary.key_topics[:3]}  "
                f"actions={summary.action_count}"
            )
            ok += 1
        except Exception as ex:
            print(f"   🔴 Fehler: {ex}")

        time.sleep(0.3)

    conn.close()
    print(f"\n🏁 Done. {ok}/{len(docs)} Dokumente zusammengefasst.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Erstelle LLM-Zusammenfassungen pro Dokument.")
    parser.add_argument("--limit", type=int, default=200, help="Max. Anzahl Dokumente")
    parser.add_argument("--min-score", type=int, default=15, help="Minimaler Impact-Score")
    parser.add_argument("--per-doc", type=int, default=6, help="Top-N Segmente pro Dokument")
    args = parser.parse_args()
    main(limit=args.limit, min_score=args.min_score, per_doc=args.per_doc)
