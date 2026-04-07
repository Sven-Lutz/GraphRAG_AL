"""
query_agent.py
==============
Stage 4 of the GraphRAG pipeline — Multi-Agent Retriever Router.

Architecture
────────────
                         User Query
                              │
                    ┌─────────▼──────────┐
                    │   Retriever Router  │  (LLM decides which tool)
                    └──┬────┬────┬───────┘
                       │    │    │
              ┌────────▼┐  ┌▼───┴─────┐  ┌▼──────────┐
              │  Local  │  │  Global  │  │ Text2Cypher│
              │  Search │  │  Search  │  │  (exact)   │
              │(vector +│  │(Map-Red. │  │            │
              │  graph) │  │communities│  └───────────┘
              └────┬────┘  └────┬─────┘
                   │            │
                   └──────┬─────┘
                          │
               ┌──────────▼──────────┐
               │   Answer Critic     │  (is the answer complete?)
               │ + Query Updater     │  (if not → new search loop)
               └──────────┬──────────┘
                          │
                     Final Answer

Tools
─────
1. local_search    – vector similarity → top-K chunks → expand entity neighborhood
2. global_search   – LazyGraphRAG map-reduce over community reports
3. text2cypher     – natural language → Cypher → exact aggregations
4. answer_critic   – validates completeness; returns new queries if needed

Usage:
    python -m crawler.scripts.query_agent "Welche Gemeinden haben Wärmepumpen gefördert?"
    python -m crawler.scripts.query_agent --interactive
    python -m crawler.scripts.query_agent --llm openai "Was kostet die Energiewende in Bayern?"
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import struct
from pathlib import Path
from typing import Any, Optional

import numpy as np

from crawler.core.config import (
    DB_PATH, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    OLLAMA_URL, OLLAMA_MODEL, MAX_CRITIC_LOOPS,
    LOCAL_SEARCH_TOP_K, GLOBAL_SEARCH_MAX_COMMUNITIES,
)

OPENAI_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _call_ollama(system: str, user: str, json_mode: bool = False) -> str:
    import urllib.request
    payload = json.dumps({
        "model":    OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream":  False,
        "options": {"temperature": 0.0, "num_predict": 2048},
        **({"format": "json"} if json_mode else {}),
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["message"]["content"]


def _call_openai(system: str, user: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


def llm(system: str, user: str, backend: str, json_mode: bool = False) -> str:
    if backend == "ollama":
        return _call_ollama(system, user, json_mode=json_mode)
    return _call_openai(system, user)


# ---------------------------------------------------------------------------
# Tool 1: Local Search
# ---------------------------------------------------------------------------

SYSTEM_LOCAL = """\
Du bist ein Experte für kommunale Klimaschutzpolitik in Bayern.
Beantworte die Frage präzise und detailliert anhand der gegebenen Kontexte.
Zitiere konkrete Fakten, Zahlen und Quellen wo möglich.
"""

def _embed_query(query: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return model.encode([query], normalize_embeddings=True)[0].astype(np.float32)


def _blob_to_vec(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)


def local_search(
    query: str,
    backend: str,
    top_k: int = 8,
    hop: int = 1,
) -> str:
    """
    1. Embed query → cosine similarity over chunks table
    2. Retrieve top-K chunks (vector search in SQLite)
    3. Expand: pull entities mentioned in those chunks + their neighbors
    4. Synthesise answer
    """
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0, isolation_level=None)

    # Vector search (brute-force cosine in Python)
    q_vec = _embed_query(query)
    rows = conn.execute(
        "SELECT chunk_id, text, heading_context, impact_score, embedding FROM chunks "
        "WHERE embedding IS NOT NULL ORDER BY impact_score DESC LIMIT 2000"
    ).fetchall()

    if not rows:
        conn.close()
        return "[LocalSearch] Keine eingebetteten Chunks gefunden. Bitte generate_embeddings.py ausführen."

    chunk_ids, texts, headings, scores, blobs = zip(*rows)
    vecs = np.stack([_blob_to_vec(b) for b in blobs])
    sims = vecs @ q_vec
    top_idx = np.argsort(sims)[::-1][:top_k]

    context_parts = []
    for i in top_idx:
        h = headings[i] or ""
        header = f"[{h}]" if h else ""
        context_parts.append(f"{header}\n{texts[i]}")

    # Expand with entity neighborhood from Neo4j
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        chunk_id_list = [chunk_ids[i] for i in top_idx]
        with driver.session() as session:
            result = session.run(
                """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE c.chunk_id IN $ids
                WITH e LIMIT 15
                MATCH (e)-[r]-(e2:Entity)
                RETURN e.name AS src, type(r) AS rel, e2.name AS dst,
                       r.evidence AS ev
                LIMIT 40
                """,
                {"ids": chunk_id_list},
            )
            rels = result.data()
            if rels:
                rel_lines = [f"  {r['src']} --[{r['rel']}]--> {r['dst']}" for r in rels]
                context_parts.append("ENTITÄTS-NETZWERK:\n" + "\n".join(rel_lines))
        driver.close()
    except Exception:
        pass  # Neo4j optional; chunks alone still useful

    conn.close()

    context = "\n\n---\n\n".join(context_parts)
    user_prompt = f"FRAGE: {query}\n\nKONTEXTE:\n{context}"
    return llm(SYSTEM_LOCAL, user_prompt, backend)


# ---------------------------------------------------------------------------
# Tool 2: Global Search (LazyGraphRAG Map-Reduce)
# ---------------------------------------------------------------------------

SYSTEM_MAP = """\
Du bist ein Analyst für kommunale Klimaschutzpolitik.
Beantworte die Frage soweit möglich anhand dieses Community-Reports.
Wenn der Report nicht relevant ist, antworte mit: NICHT_RELEVANT
"""

SYSTEM_REDUCE = """\
Du bist ein Experte für kommunale Klimaschutzpolitik in Bayern.
Fasse die gegebenen Teilantworten zu einer kohärenten, umfassenden Gesamtantwort zusammen.
Priorisiere Antworten mit hohem Wichtigkeits-Score. Nenne konkrete Fakten und Beispiele.
"""

def global_search(query: str, backend: str, max_communities: int = 50) -> str:
    """
    LazyGraphRAG Map-Reduce:
    1. Pre-filter community reports by keyword overlap (Lazy: skip irrelevant ones)
    2. Map: ask LLM to answer from each relevant report
    3. Reduce: synthesise all partial answers into one
    """
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0, isolation_level=None)

    try:
        reports = conn.execute(
            """
            SELECT community_id, title, summary, importance, key_entities
            FROM community_reports
            ORDER BY importance DESC
            LIMIT ?
            """,
            (max_communities,),
        ).fetchall()
    except Exception:
        conn.close()
        return "[GlobalSearch] Keine Community Reports. Bitte detect_communities.py ausführen."

    conn.close()

    if not reports:
        return "[GlobalSearch] Keine Community Reports gefunden."

    # LazyGraphRAG: pre-filter by keyword overlap to skip irrelevant communities
    query_words = set(re.findall(r"\w{4,}", query.lower()))
    relevant = []
    for cid, title, summary, importance, key_entities_json in reports:
        report_text = f"{title} {summary} {key_entities_json or ''}".lower()
        report_words = set(re.findall(r"\w{4,}", report_text))
        overlap = len(query_words & report_words)
        if overlap > 0:
            relevant.append((cid, title, summary, importance, overlap))

    relevant.sort(key=lambda x: (-x[3], -x[4]))  # sort by importance then overlap

    if not relevant:
        relevant = [(r[0], r[1], r[2], r[3], 0) for r in reports[:10]]

    # Augment with document summaries that overlap with query keywords
    conn2 = sqlite3.connect(str(DB_PATH), timeout=30.0, isolation_level=None)
    doc_summaries: list[str] = []
    try:
        rows = conn2.execute(
            "SELECT summary, document_type, key_topics FROM document_summaries LIMIT 30"
        ).fetchall()
        for row_summary, doc_type, topics_json in rows:
            doc_text = f"{row_summary} {topics_json or ''}".lower()
            doc_words = set(re.findall(r"\w{4,}", doc_text))
            if len(query_words & doc_words) > 0:
                doc_summaries.append(f"[{doc_type}] {row_summary}")
    except Exception:
        pass
    conn2.close()

    # Map phase
    partial_answers = []
    for cid, title, summary, importance, _ in relevant[:20]:
        extra = ""
        if doc_summaries:
            extra = "\n\nRELEVANTE DOKUMENTEN-ZUSAMMENFASSUNGEN:\n" + "\n".join(doc_summaries[:5])
        report_ctx = f"COMMUNITY: {title} (Wichtigkeit: {importance}/10)\n\n{summary}{extra}"
        answer = llm(SYSTEM_MAP, f"FRAGE: {query}\n\n{report_ctx}", backend)
        if "NICHT_RELEVANT" not in answer:
            partial_answers.append((importance, answer))

    if not partial_answers:
        return "[GlobalSearch] Keine relevanten Community Reports für diese Frage gefunden."

    partial_answers.sort(key=lambda x: -x[0])
    combined = "\n\n---\n\n".join(f"[Wichtigkeit: {imp}]\n{ans}" for imp, ans in partial_answers[:10])

    # Reduce phase
    return llm(SYSTEM_REDUCE, f"FRAGE: {query}\n\nTEILANTWORTEN:\n{combined}", backend)


# ---------------------------------------------------------------------------
# Tool 3: Text2Cypher
# ---------------------------------------------------------------------------

SCHEMA_DESCRIPTION = """\
Neo4j Schema:
Nodes: Municipality(id), Document(id), Segment(rowid, impact_score, text, heading_context),
       Chunk(chunk_id, text, impact_score, embedding), Entity(key, name, type, category, status, metrics),
       CommunityReport(community_id, title, summary, importance, domain)
Relationships: Municipality-[:HAS_DOCUMENT]->Document, Document-[:HAS_SEGMENT]->Segment,
               Document-[:HAS_CHUNK]->Chunk, Segment-[:MENTIONS]->Entity,
               Chunk-[:MENTIONS]->Entity, Entity-[rel_type]->Entity (FÖRDERT, BAUT, BESCHLIESST,
               IMPLEMENTIERT, PLANT, GEHÖRT_ZU, KOOPERIERT_MIT, FINANZIERT_DURCH, BEZIEHT_SICH_AUF),
               Entity-[:BELONGS_TO_COMMUNITY]->CommunityReport, Document-[:LINKS_TO]->Document
"""

SYSTEM_CYPHER = f"""\
Du bist ein Neo4j Cypher-Experte für kommunale Klimadaten.
Übersetze die Nutzerfrage in eine valide Cypher-Abfrage.
Antworte NUR mit der Cypher-Abfrage – ohne Erklärung, ohne Markdown-Fences.

{SCHEMA_DESCRIPTION}

Beispiele:
- Frage: "Wie viele Gemeinden haben Klimaschutzkonzepte?"
  Cypher: MATCH (d:Document)-[:HAS_SEGMENT]->(s:Segment) WHERE s.heading_context CONTAINS 'Klimaschutz' RETURN count(DISTINCT d.id) AS anzahl

- Frage: "Welche Entitäten vom Typ Förderprogramm wurden am häufigsten erwähnt?"
  Cypher: MATCH (e:Entity {{type: 'Förderprogramm'}})<-[:MENTIONS]-(s:Segment) RETURN e.name, count(s) AS mentions ORDER BY mentions DESC LIMIT 10
"""

def text2cypher(query: str, backend: str) -> str:
    """Translate natural language to Cypher; execute; format result."""
    cypher = llm(SYSTEM_CYPHER, query, backend).strip()
    # Strip any accidental markdown
    if cypher.startswith("```"):
        cypher = cypher.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run(cypher)
            rows = result.data()
        driver.close()
    except Exception as e:
        return f"[Text2Cypher] Cypher-Fehler: {e}\nQuery: {cypher}"

    if not rows:
        return f"[Text2Cypher] Keine Ergebnisse.\nCypher: {cypher}"

    # Format result as markdown table
    if rows:
        keys = list(rows[0].keys())
        header = " | ".join(keys)
        sep    = " | ".join(["---"] * len(keys))
        body   = "\n".join(" | ".join(str(r.get(k, "")) for k in keys) for r in rows[:50])
        return f"```\n{header}\n{sep}\n{body}\n```\n\n_Cypher_: `{cypher}`"
    return ""


# ---------------------------------------------------------------------------
# Retriever Router
# ---------------------------------------------------------------------------

SYSTEM_ROUTER = """\
Du bist ein Retrieval-Router für ein GraphRAG-System über kommunale Klimadaten.
Analysiere die Frage und wähle das beste Werkzeug:

- local_search:  Spezifische Detail-Fragen zu konkreten Gemeinden, Projekten, Maßnahmen
- global_search: Abstrakte Übersichts-Fragen ("Was sind die wichtigsten Trends?", "Welche Gemeinden...")
- text2cypher:   Exakte Zählungen, Aggregationen, Auflistungen ("Wie viele...", "Welche haben...")

Antworte NUR mit einem JSON-Objekt: {"tool": "local_search"|"global_search"|"text2cypher"}
"""

def route(query: str, backend: str) -> str:
    raw = llm(SYSTEM_ROUTER, query, backend, json_mode=True)
    try:
        data = json.loads(raw)
        return data.get("tool", "local_search")
    except Exception:
        return "local_search"


# ---------------------------------------------------------------------------
# Answer Critic + Continuous Query Updating
# ---------------------------------------------------------------------------

SYSTEM_CRITIC = """\
Du prüfst, ob eine Antwort eine Frage vollständig und korrekt beantwortet.

Bewerte die Antwort:
1. Ist sie vollständig? (alle Aspekte der Frage beantwortet)
2. Enthält sie konkrete Fakten? (keine reinen Allgemeinaussagen)
3. Ist sie kohärent?

Antworte mit JSON:
{
  "is_complete": true/false,
  "confidence": 0.0-1.0,
  "missing_aspects": ["..."],   // was fehlt noch?
  "follow_up_query": "..."      // neue Suchanfrage falls unvollständig, sonst null
}
"""

def critic_check(question: str, answer: str, backend: str) -> dict:
    prompt = f"FRAGE: {question}\n\nANTWORT: {answer[:2000]}"
    raw = llm(SYSTEM_CRITIC, prompt, backend, json_mode=True)
    try:
        return json.loads(raw)
    except Exception:
        return {"is_complete": True, "confidence": 0.5, "missing_aspects": [], "follow_up_query": None}


# ---------------------------------------------------------------------------
# Main query loop
# ---------------------------------------------------------------------------

def answer(query: str, backend: str = "ollama", verbose: bool = False) -> str:
    current_query = query
    all_answers   = []

    for loop in range(MAX_CRITIC_LOOPS):
        # Route
        tool = route(current_query, backend)
        if verbose:
            print(f"[loop {loop+1}] Tool: {tool}")

        # Execute
        if tool == "local_search":
            result = local_search(current_query, backend)
        elif tool == "global_search":
            result = global_search(current_query, backend)
        else:
            result = text2cypher(current_query, backend)

        all_answers.append(result)

        # Critic
        review = critic_check(query, result, backend)
        if verbose:
            print(f"[critic] complete={review.get('is_complete')}  "
                  f"confidence={review.get('confidence', 0):.2f}")

        if review.get("is_complete", True) or review.get("confidence", 0) >= 0.8:
            break

        follow_up = review.get("follow_up_query")
        if not follow_up:
            break
        current_query = follow_up
        if verbose:
            print(f"[critic] Follow-up: {follow_up}")

    # If multiple rounds, synthesise
    if len(all_answers) > 1:
        combined = "\n\n---\n\n".join(all_answers)
        final = llm(
            "Fasse diese Teilantworten zu einer kohärenten Gesamtantwort zusammen.",
            f"URSPRUNGSFRAGE: {query}\n\nTEILANTWORTEN:\n{combined}",
            backend,
        )
        return final
    return all_answers[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="GraphRAG Multi-Agent Query System (Stage 4)")
    ap.add_argument("query", nargs="?",  default=None, help="Suchanfrage")
    ap.add_argument("--llm", choices=["ollama", "openai"], default="ollama")
    ap.add_argument("--interactive", "-i", action="store_true")
    ap.add_argument("--verbose",     "-v", action="store_true")
    args = ap.parse_args()

    if args.interactive:
        print("GraphRAG Query Agent (Strg+C zum Beenden)\n")
        while True:
            try:
                q = input("Frage: ").strip()
                if not q:
                    continue
                print("\n" + answer(q, backend=args.llm, verbose=args.verbose) + "\n")
            except KeyboardInterrupt:
                break
    elif args.query:
        print(answer(args.query, backend=args.llm, verbose=args.verbose))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
