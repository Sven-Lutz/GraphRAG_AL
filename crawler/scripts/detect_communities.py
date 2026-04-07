"""
detect_communities.py
=====================
Stage 3 of the GraphRAG pipeline.

1. Runs Neo4j GDS Leiden algorithm on the Entity–Relationship graph to
   identify dense thematic clusters (communities).
2. For each community, assembles the top-scoring entity names, their
   relationship context, and supporting chunk snippets.
3. Sends this context to an LLM to generate a structured Community Report
   (title, summary, importance score, key claims, representative entities).
4. Stores reports in Neo4j (CommunityReport node) and SQLite for offline use.

Community Reports are the backbone of GraphRAG *Global Search*:
  Query → Map community reports → Reduce → Holistic answer

Prerequisites:
  - Neo4j 5.x with Graph Data Science (GDS) plugin >= 2.6
  - Populated Entity + Relationship nodes (run export_graph_to_neo4j.py first)
  - Ollama running locally (or set OPENAI_API_KEY for cloud fallback)

Usage:
    python -m crawler.scripts.detect_communities
    python -m crawler.scripts.detect_communities --algorithm louvain --min-size 3
    python -m crawler.scripts.detect_communities --llm openai  # use cloud LLM
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from crawler.core.config import (
    DB_PATH, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    OLLAMA_URL, OLLAMA_MODEL, GDS_GRAPH_NAME,
)

OPENAI_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Pydantic schema for community reports
# ---------------------------------------------------------------------------

class KeyClaim(BaseModel):
    claim: str
    importance: int = Field(ge=1, le=10)
    evidence:   str


class CommunityReport(BaseModel):
    title:       str = Field(description="Prägnanter Titel (max. 10 Wörter)")
    summary:     str = Field(description="3-5 Sätze: Kernaussagen der Community")
    importance:  int = Field(ge=1, le=10, description="Wichtigkeit für das Forschungsprojekt (1-10)")
    domain:      Literal["Klimaschutz", "Energie", "Mobilität", "Finanzen", "Governance", "Gemischt"]
    key_claims:  list[KeyClaim] = Field(default_factory=list, max_length=5)
    key_entities: list[str]    = Field(default_factory=list, max_length=10)


SYSTEM_REPORT = """\
Du bist Experte für kommunale Klimaschutzpolitik in Bayern.
Analysiere die gegebene Gruppe von Entitäten und Beziehungen aus einem Kommunal-Datensatz.
Erstelle einen strukturierten Community-Report, der die Kernaussagen und Zusammenhänge zusammenfasst.
Antworte ausschließlich mit einem JSON-Objekt – kein erklärender Text.
"""


# ---------------------------------------------------------------------------
# Neo4j GDS: project graph + run Leiden / Louvain
# ---------------------------------------------------------------------------

def project_gds_graph(session, algorithm: str) -> None:
    """Project Entity nodes + all relationship types into a GDS in-memory graph."""
    # Drop if already exists
    try:
        session.run(f"CALL gds.graph.drop('{GDS_GRAPH_NAME}', false)")
    except Exception:
        pass

    session.run(
        f"""
        CALL gds.graph.project(
            '{GDS_GRAPH_NAME}',
            'Entity',
            {{
                FÖRDERT:          {{orientation: 'UNDIRECTED'}},
                BAUT:             {{orientation: 'UNDIRECTED'}},
                BESCHLIESST:      {{orientation: 'UNDIRECTED'}},
                IMPLEMENTIERT:    {{orientation: 'UNDIRECTED'}},
                PLANT:            {{orientation: 'UNDIRECTED'}},
                GEHÖRT_ZU:        {{orientation: 'UNDIRECTED'}},
                KOOPERIERT_MIT:   {{orientation: 'UNDIRECTED'}},
                FINANZIERT_DURCH: {{orientation: 'UNDIRECTED'}},
                BEZIEHT_SICH_AUF: {{orientation: 'UNDIRECTED'}}
            }}
        )
        """
    )


def run_community_detection(
    session,
    algorithm: str = "leiden",
    min_community_size: int = 3,
) -> list[dict]:
    """
    Run Leiden (or Louvain) and write communityId back to Entity nodes.
    Returns list of {communityId, size}.
    """
    algo_call = (
        f"gds.leiden.write('{GDS_GRAPH_NAME}', {{writeProperty: 'communityId', gamma: 1.0}})"
        if algorithm == "leiden"
        else f"gds.louvain.write('{GDS_GRAPH_NAME}', {{writeProperty: 'communityId'}})"
    )
    session.run(f"CALL {algo_call}")

    result = session.run(
        """
        MATCH (e:Entity)
        WHERE e.communityId IS NOT NULL
        WITH e.communityId AS cid, count(e) AS sz
        WHERE sz >= $min_size
        RETURN cid, sz
        ORDER BY sz DESC
        """,
        {"min_size": min_community_size},
    )
    return [{"communityId": r["cid"], "size": r["sz"]} for r in result]


def get_community_context(session, community_id: int, max_entities: int = 20) -> dict:
    """
    Fetch entity names, relationship snippets, and chunk text for one community.
    """
    entities = session.run(
        """
        MATCH (e:Entity {communityId: $cid})
        RETURN e.name AS name, e.type AS type, e.category AS cat,
               e.status AS status, e.metrics AS metrics
        ORDER BY e.name
        LIMIT $max_e
        """,
        {"cid": community_id, "max_e": max_entities},
    ).data()

    relationships = session.run(
        """
        MATCH (src:Entity {communityId: $cid})-[r]->(dst:Entity {communityId: $cid})
        RETURN type(r) AS rel, src.name AS src, dst.name AS dst,
               r.evidence AS evidence
        LIMIT 30
        """,
        {"cid": community_id},
    ).data()

    # Pull a few representative chunk snippets
    chunks = session.run(
        """
        MATCH (e:Entity {communityId: $cid})<-[:MENTIONS]-(s:Segment)
        RETURN s.text AS text, s.impact_score AS score
        ORDER BY score DESC
        LIMIT 5
        """,
        {"cid": community_id},
    ).data()

    return {
        "community_id": community_id,
        "entities": entities,
        "relationships": relationships,
        "chunk_snippets": [c["text"][:400] for c in chunks if c["text"]],
    }


def build_report_prompt(ctx: dict) -> str:
    lines = [f"COMMUNITY ID: {ctx['community_id']}", ""]
    lines.append("ENTITÄTEN:")
    for e in ctx["entities"]:
        metrics = e.get("metrics") or {}
        m_str = ", ".join(f"{k}={v}" for k, v in metrics.items()) if metrics else ""
        lines.append(
            f"  - {e['name']} [{e['type']}/{e['cat']}] Status={e['status']}"
            + (f"  {m_str}" if m_str else "")
        )
    lines.append("")
    lines.append("BEZIEHUNGEN:")
    for r in ctx["relationships"][:20]:
        lines.append(f"  {r['src']} --[{r['rel']}]--> {r['dst']}")
        if r.get("evidence"):
            lines.append(f"    Beleg: \"{r['evidence'][:120]}\"")
    if ctx["chunk_snippets"]:
        lines.append("")
        lines.append("REPRÄSENTATIVE TEXTAUSZÜGE:")
        for snippet in ctx["chunk_snippets"]:
            lines.append(f"  «{snippet}»")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call (Ollama or OpenAI)
# ---------------------------------------------------------------------------

def call_llm(prompt: str, backend: str) -> CommunityReport:
    if backend == "ollama":
        import urllib.request
        payload = json.dumps({
            "model":    OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_REPORT},
                {"role": "user",   "content": prompt},
            ],
            "stream":  False,
            "options": {"temperature": 0.1, "num_predict": 1024},
            "format":  "json",
        }).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = json.loads(resp.read())["message"]["content"]
    else:
        from openai import OpenAI
        client = OpenAI()
        resp = client.beta.chat.completions.parse(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_REPORT},
                {"role": "user",   "content": prompt},
            ],
            response_format=CommunityReport,
            temperature=0.1,
        )
        return resp.choices[0].message.parsed

    # Parse Ollama response
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return CommunityReport.model_validate_json(raw)


# ---------------------------------------------------------------------------
# Store community report
# ---------------------------------------------------------------------------

def store_report_neo4j(session, community_id: int, report: CommunityReport) -> None:
    session.run(
        """
        MERGE (cr:CommunityReport {community_id: $cid})
        SET cr.title        = $title,
            cr.summary      = $summary,
            cr.importance   = $importance,
            cr.domain       = $domain,
            cr.key_entities = $key_entities,
            cr.key_claims   = $key_claims
        WITH cr
        MATCH (e:Entity {communityId: $cid})
        MERGE (e)-[:BELONGS_TO_COMMUNITY]->(cr)
        """,
        {
            "cid":         community_id,
            "title":       report.title,
            "summary":     report.summary,
            "importance":  report.importance,
            "domain":      report.domain,
            "key_entities": report.key_entities,
            "key_claims":  [c.model_dump() for c in report.key_claims],
        },
    )


def store_report_sqlite(conn: sqlite3.Connection, community_id: int, report: CommunityReport) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS community_reports (
            community_id   INTEGER PRIMARY KEY,
            title          TEXT NOT NULL,
            summary        TEXT NOT NULL,
            importance     INTEGER,
            domain         TEXT,
            key_entities   TEXT,
            key_claims     TEXT,
            created_at     TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO community_reports
            (community_id, title, summary, importance, domain, key_entities, key_claims)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            community_id,
            report.title,
            report.summary,
            report.importance,
            report.domain,
            json.dumps(report.key_entities, ensure_ascii=False),
            json.dumps([c.model_dump() for c in report.key_claims], ensure_ascii=False),
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    algorithm: str = "leiden",
    min_community_size: int = 3,
    llm_backend: str = "ollama",
    max_communities: int = 500,
    sleep_between: float = 0.3,
) -> None:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    conn   = sqlite3.connect(str(DB_PATH), timeout=60.0, isolation_level=None)

    with driver.session() as session:
        print(f"📐 Projiziere Entity-Graph in GDS…")
        project_gds_graph(session, algorithm)

        print(f"🔍 Führe {algorithm.capitalize()} Community Detection durch…")
        communities = run_community_detection(session, algorithm, min_community_size)
        print(f"   {len(communities)} Communities gefunden (min_size={min_community_size})")

        ok = 0
        for comm in communities[:max_communities]:
            cid  = int(comm["communityId"])
            size = int(comm["size"])
            print(f"\n⚙️  Community {cid} (size={size})")
            try:
                ctx    = get_community_context(session, cid)
                prompt = build_report_prompt(ctx)
                report = call_llm(prompt, llm_backend)

                store_report_neo4j(session, cid, report)
                store_report_sqlite(conn, cid, report)

                print(f"   🟢 [{report.domain}] {report.title}  importance={report.importance}")
                ok += 1
            except Exception as ex:
                print(f"   🔴 Fehler: {ex}")

            time.sleep(sleep_between)

        # Drop projected graph to free GDS memory
        try:
            session.run(f"CALL gds.graph.drop('{GDS_GRAPH_NAME}', false)")
        except Exception:
            pass

    driver.close()
    conn.close()
    print(f"\n🏁 Fertig: {ok}/{min(len(communities), max_communities)} Community Reports erstellt.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Community Detection + LLM Reports (Stage 3)")
    ap.add_argument("--algorithm",  choices=["leiden", "louvain"], default="leiden")
    ap.add_argument("--min-size",   type=int,   default=3,       help="Minimale Community-Größe")
    ap.add_argument("--llm",        choices=["ollama", "openai"], default="ollama")
    ap.add_argument("--max",        type=int,   default=500,     help="Max. Communities für Reports")
    ap.add_argument("--sleep",      type=float, default=0.3)
    args = ap.parse_args()
    main(
        algorithm=args.algorithm,
        min_community_size=args.min_size,
        llm_backend=args.llm,
        max_communities=args.max,
        sleep_between=args.sleep,
    )
