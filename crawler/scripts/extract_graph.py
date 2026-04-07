from __future__ import annotations

import sqlite3
import time
import hashlib
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field
from openai import OpenAI

client = OpenAI()
MODEL_NAME = "gpt-4o-mini"
DB_PATH = Path("crawler/data/db/crawl.sqlite")

# Neighbours to fetch for context window around the target segment
CONTEXT_WINDOW = 1


class Entity(BaseModel):
    name: str
    type: Literal[
        "Akteur",          # Person, Organisation, Behörde
        "Infrastruktur",   # Anlage, Netz, Gebäude
        "Förderprogramm",  # KfW, BAFA, NKI, EFRE …
        "Maßnahme",        # Konkrete Aktion / Projekt
        "Konzept/Ziel",    # Strategie, Plan, Zieldefinition
        "Dokument",        # Beschluss, Konzept, Antrag
    ]
    category: Literal["Mobilität", "Wärme", "Strom", "Finanzen", "Governance", "Sonstiges"]
    status: Literal["Geplant", "In Umsetzung", "Abgeschlossen", "Existierend", "Unbekannt"]
    metrics: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Quantitative Kennzahlen: z.B. {'year': '2025', 'amount': '500.000 €', "
            "'capacity': '100 kWp', 'co2_reduction': '50 t/a', 'deadline': '2030'}"
        ),
    )


class Relationship(BaseModel):
    source_entity: str
    relation_type: Literal[
        "FÖRDERT",           # A fördert B (finanziell)
        "BAUT",              # A baut / realisiert B
        "BESCHLIESST",       # A beschließt B (Gremium → Maßnahme)
        "IMPLEMENTIERT",     # A setzt B um (laufend/abgeschlossen)
        "PLANT",             # A plant B (noch nicht umgesetzt)
        "GEHÖRT_ZU",         # B ist Teil von A
        "KOOPERIERT_MIT",    # A arbeitet mit B zusammen
        "FINANZIERT_DURCH",  # A wird finanziert durch B
        "BEZIEHT_SICH_AUF",  # generische Referenz
    ]
    target_entity: str
    evidence: str = Field(description="Exakter Originaltext-Ausschnitt (max. 200 Zeichen)")


class KnowledgeGraph(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)


SYSTEM_MSG = """\
Du bist Experte für kommunale Klimaschutzpolitik in Bayern.
Extrahiere strukturierte Knowledge-Graph-Daten aus kommunalen Klimaschutztexten.

Regeln:
- Nur Fakten, die direkt im Text belegt sind – keine Annahmen oder Schlussfolgerungen.
- evidence: exakter Originaltext-Ausschnitt (max. 200 Zeichen).
- Zeitangaben (Jahre, Fristen) als metrics erfassen: {"year": "2025", "deadline": "2030"}.
- Geldbeträge als metrics: {"amount": "500.000 €", "foerderquote": "80%"}.
- Kapazitäten als metrics: {"capacity": "150 kWp", "co2_reduction": "45 t/a"}.
- Bei keinen relevanten Fakten: entities=[] relationships=[].
"""


def setup_db(cur: sqlite3.Cursor) -> None:
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA busy_timeout=60000;")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS graph_triplets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            municipality_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            segment_rowid INTEGER NOT NULL,
            segment_hash TEXT NOT NULL,
            model_name TEXT NOT NULL,
            graph_json TEXT NOT NULL,
            entity_count INTEGER NOT NULL DEFAULT 0,
            relationship_count INTEGER NOT NULL DEFAULT 0,
            extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_graph_triplets_seg ON graph_triplets(segment_rowid)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_graph_triplets_muni ON graph_triplets(municipality_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_graph_triplets_doc ON graph_triplets(document_id)"
    )


def get_segments(
    cur: sqlite3.Cursor,
    limit: int = 500,
    min_len: int = 160,
    min_score: int = 20,
    per_doc: int = 5,
):
    """
    Return the top-scoring unprocessed segments.
    Also fetches heading_context and the document URL for enriched prompting.
    """
    query = """
    WITH ranked AS (
      SELECT
        s.rowid            AS rowid,
        s.order_index      AS order_index,
        d.municipality_id  AS municipality_id,
        d.document_id      AS document_id,
        d.url_canonical    AS url_canonical,
        s.text             AS text,
        s.heading_context  AS heading_context,
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
    FROM ranked
    WHERE rn <= ?
    ORDER BY impact_score DESC
    LIMIT ?;
    """
    cur.execute(query, (min_len, min_score, per_doc, limit))
    return cur.fetchall()


def get_neighbor_texts(
    cur: sqlite3.Cursor,
    document_id: str,
    order_index: int,
    window: int = CONTEXT_WINDOW,
) -> tuple[str, str]:
    """Fetch the text of the preceding and following segments for context."""
    cur.execute(
        """
        SELECT order_index, text FROM segments
        WHERE document_id = ?
          AND order_index BETWEEN ? AND ?
          AND COALESCE(is_negative, 0) = 0
        ORDER BY order_index
        """,
        (document_id, order_index - window, order_index + window),
    )
    rows = cur.fetchall()
    prev_parts = [r[1] for r in rows if r[0] < order_index]
    next_parts = [r[1] for r in rows if r[0] > order_index]
    return " ".join(prev_parts), " ".join(next_parts)


def seg_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def build_user_prompt(
    text: str,
    heading_context: Optional[str],
    url: Optional[str],
    prev_text: str,
    next_text: str,
) -> str:
    parts: list[str] = []
    if heading_context:
        parts.append(f"ABSCHNITT: {heading_context}")
    if url:
        parts.append(f"QUELLE: {url}")
    if prev_text:
        parts.append(f"[VORHERIGER KONTEXT]\n{prev_text[:400]}")
    parts.append(f"[ZIELTEXT]\n{text}")
    if next_text:
        parts.append(f"[NACHFOLGENDER KONTEXT]\n{next_text[:400]}")
    return "\n\n".join(parts)


def extract(user_prompt: str) -> KnowledgeGraph:
    resp = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_prompt},
        ],
        response_format=KnowledgeGraph,
        temperature=0.0,
    )
    return resp.choices[0].message.parsed


def main():
    conn = sqlite3.connect(str(DB_PATH), timeout=60.0, isolation_level=None)
    cur = conn.cursor()
    setup_db(cur)

    segs = get_segments(cur, limit=500, min_len=160, min_score=20, per_doc=5)
    if not segs:
        print("✅ Keine neuen Segmente (scored) gefunden.")
        conn.close()
        return

    print(f"🚀 Extrahiere Graph für {len(segs)} Segmente\n")

    for rowid, order_index, muni, doc, url, text, heading_ctx, score in segs:
        print(f"⚙️  muni={muni} doc={doc[:8]}… rowid={rowid} score={score}")
        if heading_ctx:
            print(f"   📂 {heading_ctx}")
        try:
            prev_text, next_text = get_neighbor_texts(cur, doc, order_index)
            user_prompt = build_user_prompt(text, heading_ctx, url, prev_text, next_text)

            kg = extract(user_prompt)
            graph_json = kg.model_dump_json(ensure_ascii=False)
            e, r = len(kg.entities), len(kg.relationships)
            print(f"   🟢 entities={e} relationships={r}")

            cur.execute(
                """
                INSERT OR IGNORE INTO graph_triplets (
                    municipality_id, document_id, segment_rowid, segment_hash,
                    model_name, graph_json, entity_count, relationship_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (str(muni), str(doc), int(rowid), seg_hash(text), MODEL_NAME, graph_json, int(e), int(r)),
            )

        except Exception as ex:
            print(f"   🔴 Fehler: {ex}")

        time.sleep(0.3)

    conn.close()
    print("\n🏁 Done.")


if __name__ == "__main__":
    main()
