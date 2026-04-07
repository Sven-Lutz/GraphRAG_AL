"""
config.py
=========
Single source of truth for all runtime settings.

Every script imports from here. Override via environment variables or by
editing this file directly — no more hunting for hardcoded URLs.

Environment variable precedence:
    KLIMA_NEO4J_URI        overrides NEO4J_URI
    KLIMA_NEO4J_USER       overrides NEO4J_USER
    KLIMA_NEO4J_PASSWORD   overrides NEO4J_PASSWORD
    KLIMA_OLLAMA_URL       overrides OLLAMA_URL
    KLIMA_OLLAMA_MODEL     overrides OLLAMA_MODEL
    KLIMA_DB_PATH          overrides DB_PATH
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DB_PATH: Path = Path(os.environ.get("KLIMA_DB_PATH", "crawler/data/db/crawl.sqlite"))

# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------
NEO4J_URI:      str = os.environ.get("KLIMA_NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER:     str = os.environ.get("KLIMA_NEO4J_USER",     "neo4j")
NEO4J_PASSWORD: str = os.environ.get("KLIMA_NEO4J_PASSWORD", "password")

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
OLLAMA_URL:   str = os.environ.get("KLIMA_OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("KLIMA_OLLAMA_MODEL", "llama3.1:8b")

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE:    int = 600   # tokens per chunk (cl100k_base)
CHUNK_OVERLAP: int = 60    # token overlap between consecutive chunks

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBED_MODEL:      str = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_BATCH_SIZE: int = 64

# ---------------------------------------------------------------------------
# Graph extraction
# ---------------------------------------------------------------------------
EXTRACT_MIN_SCORE:  float = 15.0
EXTRACT_CONTEXT_WINDOW: int = 1   # neighbour segments included in prompt

# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------
COMMUNITY_MIN_SIZE: int = 3
COMMUNITY_ALGORITHM: str = "leiden"   # or "louvain"
GDS_GRAPH_NAME: str = "klimaEntities"

# ---------------------------------------------------------------------------
# Query agent
# ---------------------------------------------------------------------------
MAX_CRITIC_LOOPS: int = 3
LOCAL_SEARCH_TOP_K: int = 8
GLOBAL_SEARCH_MAX_COMMUNITIES: int = 50
