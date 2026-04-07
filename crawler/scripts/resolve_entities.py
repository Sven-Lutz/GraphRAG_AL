"""
resolve_entities.py
===================
Post-processing entity resolution for the KlimaCrawler knowledge graph.

Reads all entities from graph_triplets, normalises names, detects duplicates,
and writes a canonical mapping to the `entity_canonical` table.  The Neo4j
export can then merge variant nodes into one canonical entity, dramatically
reducing graph fragmentation.

Usage:
    python -m crawler.scripts.resolve_entities
    python -m crawler.scripts.resolve_entities --min-freq 2 --similarity 0.85
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

DB_PATH = Path("crawler/data/db/crawl.sqlite")

# ---------------------------------------------------------------------------
# Normalisation rules
# ---------------------------------------------------------------------------

# Legal suffixes commonly appended to org names
_LEGAL_SUFFIXES = re.compile(
    r"\s*\b(gmbh|ag|e\.?\s?v\.?|mbh|ohg|kg|co\.?\s?kg|ggmbh|gbr|se|kör|aör)\s*$",
    re.IGNORECASE,
)
# Common German articles and prepositions
_ARTICLES = re.compile(
    r"\b(der|die|das|des|dem|den|ein|eine|einer|eines|einem|einen"
    r"|und|oder|für|von|zum|zur|im|am|auf|aus|bei|in|mit|nach|zu)\b",
    re.IGNORECASE,
)
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\wäöüÄÖÜß\- ]+", re.UNICODE)


def normalise_entity_name(name: str) -> str:
    """Aggressive normalisation for duplicate detection."""
    s = (name or "").strip()
    s = _LEGAL_SUFFIXES.sub("", s)
    s = _PUNCT.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    s = _WS.sub(" ", s).strip().lower()
    return s


def similarity(a: str, b: str) -> float:
    """Quick string similarity via SequenceMatcher."""
    return SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------

def setup_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA busy_timeout=60000;

        CREATE TABLE IF NOT EXISTS entity_canonical (
            original_name    TEXT NOT NULL,
            entity_type      TEXT NOT NULL,
            category         TEXT NOT NULL,
            canonical_name   TEXT NOT NULL,
            normalised_key   TEXT NOT NULL,
            frequency        INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (original_name, entity_type, category)
        );
        CREATE INDEX IF NOT EXISTS idx_ec_canonical
            ON entity_canonical(canonical_name);
        CREATE INDEX IF NOT EXISTS idx_ec_key
            ON entity_canonical(normalised_key);
        """
    )


# ---------------------------------------------------------------------------
# Entity extraction from graph_triplets JSON
# ---------------------------------------------------------------------------

@dataclass
class RawEntity:
    name: str
    entity_type: str
    category: str


def load_all_entities(conn: sqlite3.Connection) -> list[RawEntity]:
    cur = conn.execute("SELECT graph_json FROM graph_triplets WHERE entity_count > 0")
    entities: list[RawEntity] = []
    for (graph_json,) in cur.fetchall():
        try:
            kg = json.loads(graph_json)
        except Exception:
            continue
        for e in kg.get("entities", []):
            name = (e.get("name") or "").strip()
            if not name or len(name) < 2:
                continue
            entities.append(RawEntity(
                name=name,
                entity_type=e.get("type", "Dokument"),
                category=e.get("category", "Sonstiges"),
            ))
    return entities


# ---------------------------------------------------------------------------
# Resolution logic
# ---------------------------------------------------------------------------

def resolve(
    entities: list[RawEntity],
    sim_threshold: float = 0.85,
) -> dict[tuple[str, str, str], str]:
    """
    Returns a mapping: (original_name, entity_type, category) → canonical_name.

    Strategy:
    1. Exact-match on normalised key → group.
    2. Within each (entity_type, category), fuzzy-match remaining names
       above sim_threshold → merge into the most frequent variant.
    """
    # Step 1: Group by normalised key + type + category
    groups: dict[str, list[str]] = defaultdict(list)
    for ent in entities:
        key = f"{ent.entity_type}::{ent.category}::{normalise_entity_name(ent.name)}"
        groups[key].append(ent.name)

    # For each group, pick the most frequent (ties: longest) as canonical
    key_to_canonical: dict[str, str] = {}
    for key, names in groups.items():
        from collections import Counter
        freq = Counter(names)
        canonical = max(freq.keys(), key=lambda n: (freq[n], len(n)))
        key_to_canonical[key] = canonical

    # Build initial mapping
    mapping: dict[tuple[str, str, str], str] = {}
    for ent in entities:
        key = f"{ent.entity_type}::{ent.category}::{normalise_entity_name(ent.name)}"
        mapping[(ent.name, ent.entity_type, ent.category)] = key_to_canonical[key]

    # Step 2: Fuzzy merge within same (type, category)
    # Group canonical names by (type, category)
    type_cat_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key, canonical in key_to_canonical.items():
        parts = key.split("::", 2)
        if len(parts) == 3:
            type_cat_groups[(parts[0], parts[1])].append(canonical)

    merge_map: dict[str, str] = {}  # old_canonical → new_canonical
    for (etype, cat), canonicals in type_cat_groups.items():
        unique = list(set(canonicals))
        unique.sort(key=lambda n: (-len(n), n))  # longer names first (usually more specific)
        merged: set[int] = set()

        for i in range(len(unique)):
            if i in merged:
                continue
            for j in range(i + 1, len(unique)):
                if j in merged:
                    continue
                norm_i = normalise_entity_name(unique[i])
                norm_j = normalise_entity_name(unique[j])
                if not norm_i or not norm_j:
                    continue
                # Check if one is substring of the other, or high similarity
                is_substring = norm_i in norm_j or norm_j in norm_i
                is_similar = similarity(norm_i, norm_j) >= sim_threshold
                if is_substring or is_similar:
                    # Merge j into i (i is longer/more specific)
                    merge_map[unique[j]] = unique[i]
                    merged.add(j)

    # Apply fuzzy merges
    for key in mapping:
        canonical = mapping[key]
        if canonical in merge_map:
            mapping[key] = merge_map[canonical]

    return mapping


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(sim_threshold: float = 0.85, min_freq: int = 1) -> None:
    conn = sqlite3.connect(str(DB_PATH), timeout=60.0, isolation_level=None)
    setup_db(conn)

    print("📖 Lade Entitäten aus graph_triplets…")
    entities = load_all_entities(conn)
    if not entities:
        print("✅ Keine Entitäten gefunden.")
        conn.close()
        return
    print(f"   {len(entities)} Rohentitäten geladen.")

    print("🔗 Starte Entity Resolution…")
    mapping = resolve(entities, sim_threshold=sim_threshold)

    # Count unique original → canonical pairs
    unique_pairs = set(mapping.items())
    n_originals = len(set(k[0] for k in mapping))
    n_canonicals = len(set(mapping.values()))
    n_merged = n_originals - n_canonicals

    print(f"   {n_originals} originale Namen → {n_canonicals} kanonische Entitäten")
    print(f"   {n_merged} Duplikate aufgelöst")

    # Compute frequency per canonical
    from collections import Counter
    canonical_freq = Counter(mapping.values())

    # Store to DB
    rows = []
    for (orig_name, etype, cat), canonical in mapping.items():
        norm_key = normalise_entity_name(orig_name)
        freq = canonical_freq.get(canonical, 1)
        if freq >= min_freq:
            rows.append((orig_name, etype, cat, canonical, norm_key, freq))

    # Deduplicate rows by primary key
    seen_pk = set()
    deduped = []
    for row in rows:
        pk = (row[0], row[1], row[2])
        if pk not in seen_pk:
            seen_pk.add(pk)
            deduped.append(row)

    conn.execute("DELETE FROM entity_canonical")
    conn.executemany(
        """
        INSERT OR REPLACE INTO entity_canonical
            (original_name, entity_type, category, canonical_name, normalised_key, frequency)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        deduped,
    )
    print(f"💾 {len(deduped)} Einträge in entity_canonical geschrieben.")

    # Show top merges
    merges = [(o, c) for (o, et, cat), c in mapping.items() if o != c]
    if merges:
        print(f"\n🔀 Beispiel-Zusammenführungen (Top 20):")
        for orig, canon in sorted(set(merges), key=lambda x: x[0])[:20]:
            print(f"   '{orig}' → '{canon}'")

    conn.close()
    print("\n🏁 Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entity Resolution für den KlimaCrawler Knowledge Graph.")
    parser.add_argument("--similarity", type=float, default=0.85, help="Fuzzy-Schwelle (0.0-1.0)")
    parser.add_argument("--min-freq", type=int, default=1, help="Min. Häufigkeit für Aufnahme")
    args = parser.parse_args()
    main(sim_threshold=args.similarity, min_freq=args.min_freq)
