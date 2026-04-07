# crawler/core/storage.py
from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

DB_PATH = Path("crawler/data/db/crawl.sqlite")
RAW_DIR = Path("crawler/data/raw")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


@dataclass(frozen=True)
class RawWriteResult:
    doc_id: str
    raw_hash: str
    raw_path: str


class Storage:
    def __init__(self, db_path: Path = DB_PATH, raw_dir: Path = RAW_DIR) -> None:
        self.db_path = Path(db_path)
        self.raw_dir = Path(raw_dir)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(
            str(self.db_path),
            timeout=60.0,        # wait longer on locks
            isolation_level=None # AUTOCOMMIT (minimizes lock duration)
        )
        self.conn.row_factory = sqlite3.Row

        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA busy_timeout=60000;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        self.conn.execute("PRAGMA cache_size=-64000;")

        self._init_schema()

        self._sql_insert_doc = """
            INSERT INTO documents_raw (
                document_id, municipality_id, url_canonical, url_final, fetched_at,
                status_code, content_type, raw_hash, raw_path, raw_ext, content_length
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self._sql_link_muni_doc = """
            INSERT OR IGNORE INTO municipality_documents (municipality_id, document_id)
            VALUES (?, ?)
        """

        # NOTE: extended with impact_score/hit_count/is_negative/heading_context (nullable)
        self._sql_insert_segment = """
            INSERT OR IGNORE INTO segments (
                segment_id, document_id, order_index, segment_type, text, segment_hash, page_ref,
                impact_score, hit_count, is_negative, heading_context
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        self._sql_mark_visited = """
            INSERT INTO visited (url_canonical, last_fetch_at, last_status_code, last_error)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(url_canonical) DO UPDATE SET
                last_fetch_at=excluded.last_fetch_at,
                last_status_code=excluded.last_status_code,
                last_error=excluded.last_error
        """

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _table_columns(self, table: str) -> Set[str]:
        try:
            cur = self.conn.execute(f"PRAGMA table_info({table})")
            return {str(r[1]) for r in cur.fetchall()}
        except Exception:
            return set()

    def _ensure_column(self, table: str, col: str, ddl_type: str) -> None:
        cols = self._table_columns(table)
        if col in cols:
            return
        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl_type}")

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents_raw (
                document_id     TEXT PRIMARY KEY,
                municipality_id TEXT,
                url_canonical   TEXT,
                url_final       TEXT,
                fetched_at      TEXT,
                status_code     INTEGER,
                content_type    TEXT,
                raw_hash        TEXT,
                raw_path        TEXT,
                raw_ext         TEXT,
                content_length  INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_documents_raw_url_canonical
            ON documents_raw(url_canonical);

            CREATE TABLE IF NOT EXISTS segments (
                segment_id   TEXT PRIMARY KEY,
                document_id  TEXT NOT NULL,
                order_index  INTEGER,
                segment_type TEXT,
                text         TEXT,
                segment_hash TEXT,
                page_ref     TEXT,
                FOREIGN KEY(document_id) REFERENCES documents_raw(document_id) ON DELETE CASCADE
            );

            CREATE UNIQUE INDEX IF NOT EXISTS uq_segment_dedup
            ON segments(document_id, segment_hash);

            CREATE TABLE IF NOT EXISTS visited (
                url_canonical    TEXT PRIMARY KEY,
                last_fetch_at    TEXT,
                last_status_code INTEGER,
                last_error       TEXT
            );

            CREATE TABLE IF NOT EXISTS municipality_documents (
                municipality_id TEXT NOT NULL,
                document_id     TEXT NOT NULL,
                PRIMARY KEY (municipality_id, document_id),
                FOREIGN KEY(document_id) REFERENCES documents_raw(document_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_muni_docs_doc
            ON municipality_documents(document_id);

            -- Distributed seed work queue
            CREATE TABLE IF NOT EXISTS seed_jobs (
              municipality_id TEXT PRIMARY KEY,
              seed_url        TEXT NOT NULL,

              status          TEXT NOT NULL DEFAULT 'pending',  -- pending|claimed|done|failed
              claimed_by      TEXT,
              claimed_at      TEXT,
              heartbeat_at    TEXT,
              done_at         TEXT,
              last_error      TEXT,

              attempt_count   INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_seed_jobs_status
            ON seed_jobs(status);

            CREATE INDEX IF NOT EXISTS idx_seed_jobs_claimed_at
            ON seed_jobs(claimed_at);
            """
        )

        # ensure extra columns on documents_raw (legacy safety)
        try:
            self._ensure_column("documents_raw", "raw_ext", "TEXT")
            self._ensure_column("documents_raw", "content_length", "INTEGER")
            self._ensure_column("documents_raw", "meta_json", "TEXT")
        except Exception:
            pass

        # --- segments enrichment columns (for topic modeling / relevance filtering) ---
        try:
            self._ensure_column("segments", "impact_score", "INTEGER")
            self._ensure_column("segments", "hit_count", "INTEGER")
            self._ensure_column("segments", "is_negative", "INTEGER")
            self._ensure_column("segments", "heading_context", "TEXT")
        except Exception:
            pass

        # --- indices for fast top-segment retrieval ---
        try:
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_segments_doc_score ON segments(document_id, impact_score DESC)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_segments_score ON segments(impact_score DESC)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_segments_negative ON segments(is_negative)"
            )
        except Exception:
            pass

        # --- graph_triplets: index for fast per-municipality graph queries ---
        try:
            self.conn.execute(
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
                    extracted_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self.conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_graph_triplets_seg ON graph_triplets(segment_rowid)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_graph_triplets_muni ON graph_triplets(municipality_id)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_graph_triplets_doc ON graph_triplets(document_id)"
            )
        except Exception:
            pass

        # --- document_links: cross-document link graph for GraphRAG edges ---
        try:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_links (
                    source_document_id TEXT NOT NULL,
                    target_url_canonical TEXT NOT NULL,
                    target_document_id TEXT,
                    anchor_text TEXT,
                    PRIMARY KEY (source_document_id, target_url_canonical),
                    FOREIGN KEY(source_document_id) REFERENCES documents_raw(document_id) ON DELETE CASCADE
                )
                """
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_doclinks_target ON document_links(target_url_canonical)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_doclinks_target_doc ON document_links(target_document_id)"
            )
        except Exception:
            pass

        # --- document_summaries: LLM-generated per-document summary for GraphRAG ---
        try:
            self.conn.execute(
                """
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
                )
                """
            )
            self.conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_doc_summaries ON document_summaries(document_id)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_doc_summaries_muni ON document_summaries(municipality_id)"
            )
        except Exception:
            pass

        self.conn.commit()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self.conn:
            yield

    def _extract_body(self, fetch_result: Any) -> bytes:
        body = getattr(fetch_result, "body", None)
        if body is None:
            body = getattr(fetch_result, "content", None)
        if body is None:
            return b""
        if isinstance(body, (bytes, bytearray, memoryview)):
            return bytes(body)
        try:
            return bytes(body)
        except Exception:
            return b""

    def _extract_headers(self, fetch_result: Any) -> Dict[str, str]:
        h = getattr(fetch_result, "headers", None)
        if isinstance(h, dict):
            return {str(k): str(v) for k, v in h.items()}
        try:
            if h is not None:
                return {str(k): str(v) for k, v in dict(h).items()}
        except Exception:
            pass
        return {}

    @staticmethod
    def _guess_ext(content_type: str, url_final: str) -> str:
        ct = (content_type or "").lower()
        url = (url_final or "").lower()

        if "application/pdf" in ct or url.endswith(".pdf"):
            return ".pdf"
        if "text/html" in ct or "application/xhtml" in ct or url.endswith((".html", ".htm")):
            return ".html"
        return ".bin"

    def store_raw(self, municipality_id: str, url_canonical: str, fetch_result: Any) -> str:
        doc_id = str(uuid.uuid4())

        body = self._extract_body(fetch_result)
        raw_hash = hashlib.sha256(body).hexdigest()
        raw_path = self.raw_dir / raw_hash

        if not raw_path.exists():
            tmp_path = self.raw_dir / f".{raw_hash}.{uuid.uuid4().hex}.tmp"
            tmp_path.write_bytes(body)
            os.replace(tmp_path, raw_path)

        headers = self._extract_headers(fetch_result)

        url_final = (
            str(getattr(fetch_result, "url_final", "") or "")
            or str(getattr(fetch_result, "url", "") or "")
        )
        status_code = int(getattr(fetch_result, "status_code", 0) or 0)
        content_type = (
            str(getattr(fetch_result, "content_type", "") or "")
            or str(headers.get("Content-Type") or "")
        )

        raw_ext = self._guess_ext(content_type, url_final)
        content_length = len(body)

        self.conn.execute(
            self._sql_insert_doc,
            (
                doc_id,
                municipality_id,
                url_canonical,
                url_final,
                _utc_now_iso(),
                status_code,
                content_type,
                raw_hash,
                str(raw_path),
                raw_ext,
                int(content_length),
            ),
        )
        self.conn.execute(self._sql_link_muni_doc, (municipality_id, doc_id))
        return doc_id

    def update_document_meta(self, document_id: str, meta: dict) -> None:
        """Store HTML-extracted metadata (description, date, author, ...) on the document."""
        if not meta:
            return
        self.conn.execute(
            "UPDATE documents_raw SET meta_json=? WHERE document_id=?",
            (json.dumps(meta, ensure_ascii=False), document_id),
        )

    def store_outlinks(self, source_doc_id: str, links: Iterable[tuple[str, str]]) -> int:
        """Persist outlinks for the document link graph (GraphRAG LINKS_TO edges)."""
        rows = []
        for url, anchor in links:
            url = (url or "").strip()
            if not url:
                continue
            rows.append((source_doc_id, url, (anchor or "")[:500]))
        if not rows:
            return 0
        before = self.conn.total_changes
        self.conn.executemany(
            """
            INSERT OR IGNORE INTO document_links (source_document_id, target_url_canonical, anchor_text)
            VALUES (?, ?, ?)
            """,
            rows,
        )
        after = self.conn.total_changes
        return max(0, after - before)

    def resolve_document_links(self) -> int:
        """
        Populate target_document_id in document_links by joining on documents_raw.url_canonical.
        Call this once after a crawl batch completes so that LINKS_TO edges can be
        exported to Neo4j.  Safe to call multiple times (only updates NULLs).
        """
        cur = self.conn.execute(
            """
            UPDATE document_links
            SET target_document_id = (
                SELECT document_id
                FROM documents_raw
                WHERE url_canonical = document_links.target_url_canonical
                LIMIT 1
            )
            WHERE target_document_id IS NULL
            """
        )
        return int(cur.rowcount or 0)

    def link_document_to_municipality(self, municipality_id: str, document_id: str) -> None:
        self.conn.execute(self._sql_link_muni_doc, (municipality_id, document_id))

    def store_segments(self, document_id: str, segments: Iterable[Any]) -> int:
        """
        Stores segments and (optionally) segment relevance features:
          - impact_score (INTEGER)
          - hit_count (INTEGER)
          - is_negative (INTEGER 0/1)
        If those attributes are absent on Segment objects, NULL is stored.
        """
        rows: List[
            Tuple[
                str, str, int, str, str, str, Optional[str],
                Optional[int], Optional[int], Optional[int]
            ]
        ] = []
        sha256 = hashlib.sha256

        for seg in segments:
            text = str(getattr(seg, "text", "") or "")
            norm = text.strip()
            if not norm:
                continue

            seg_hash = sha256(norm.encode("utf-8")).hexdigest()

            impact_score = getattr(seg, "impact_score", None)
            hit_count = getattr(seg, "hit_count", None)
            is_negative = getattr(seg, "is_negative", None)

            rows.append(
                (
                    str(uuid.uuid4()),
                    document_id,
                    int(getattr(seg, "order_index", 0) or 0),
                    str(getattr(seg, "segment_type", "") or ""),
                    text,
                    seg_hash,
                    getattr(seg, "page_ref", None),
                    int(impact_score) if impact_score is not None else None,
                    int(hit_count) if hit_count is not None else None,
                    int(is_negative) if is_negative is not None else None,
                    getattr(seg, "heading_context", None),
                )
            )

        if not rows:
            return 0

        before = self.conn.total_changes
        self.conn.executemany(self._sql_insert_segment, rows)
        after = self.conn.total_changes
        return max(0, after - before)

    def mark_visited(self, url_canonical: str, status: int, error: Optional[str] = None) -> None:
        self.conn.execute(self._sql_mark_visited, (url_canonical, _utc_now_iso(), int(status), error))

    def is_visited(self, url_canonical: str) -> bool:
        cur = self.conn.execute("SELECT 1 FROM visited WHERE url_canonical=? LIMIT 1", (url_canonical,))
        return cur.fetchone() is not None

    def get_document_id_by_canonical_url(self, url_canonical: str) -> Optional[str]:
        cur = self.conn.execute(
            "SELECT document_id FROM documents_raw WHERE url_canonical=? LIMIT 1",
            (url_canonical,),
        )
        row = cur.fetchone()
        return str(row["document_id"]) if row else None

    def upsert_seed_jobs(self, seeds: Iterable[Tuple[str, str]]) -> int:
        rows = [(str(m), str(u)) for (m, u) in seeds]
        if not rows:
            return 0
        before = self.conn.total_changes
        with self.conn:
            self.conn.executemany(
                """
                INSERT INTO seed_jobs (municipality_id, seed_url, status, attempt_count)
                VALUES (?, ?, 'pending', 0)
                ON CONFLICT(municipality_id) DO UPDATE SET
                    seed_url=excluded.seed_url
                """,
                rows,
            )
        after = self.conn.total_changes
        return max(0, after - before)

    def claim_next_seed_jobs(
        self,
        worker_id: Optional[str] = None,
        limit: int = 1,
    ) -> List[Tuple[str, str]]:
        wid = worker_id or default_worker_id()
        now = _utc_now_iso()
        limit = max(1, int(limit))

        with self.conn:
            self.conn.execute(
                """
                UPDATE seed_jobs
                SET status='claimed',
                    claimed_by=?,
                    claimed_at=?,
                    heartbeat_at=?,
                    attempt_count=attempt_count+1
                WHERE municipality_id IN (
                    SELECT municipality_id
                    FROM seed_jobs
                    WHERE status='pending'
                    ORDER BY municipality_id
                    LIMIT ?
                )
                """,
                (wid, now, now, limit),
            )

            cur = self.conn.execute(
                """
                SELECT municipality_id, seed_url
                FROM seed_jobs
                WHERE status='claimed' AND claimed_by=?
                ORDER BY claimed_at DESC
                LIMIT ?
                """,
                (wid, limit),
            )
            return [(str(r["municipality_id"]), str(r["seed_url"])) for r in cur.fetchall()]

    def heartbeat_seed_jobs(self, worker_id: Optional[str] = None) -> int:
        wid = worker_id or default_worker_id()
        now = _utc_now_iso()
        with self.conn:
            cur = self.conn.execute(
                "UPDATE seed_jobs SET heartbeat_at=? WHERE status='claimed' AND claimed_by=?",
                (now, wid),
            )
        return int(cur.rowcount or 0)

    def finish_seed_job(self, municipality_id: str, ok: bool, error: Optional[str] = None) -> None:
        now = _utc_now_iso()
        status = "done" if ok else "failed"
        with self.conn:
            self.conn.execute(
                """
                UPDATE seed_jobs
                SET status=?, done_at=?, last_error=?
                WHERE municipality_id=?
                """,
                (status, now, error, str(municipality_id)),
            )

    def is_visited_with_error(self, url: str) -> bool:
        """Prüft, ob eine URL bereits besucht wurde, aber einen Fehler (z.B. 404) erzeugt hat."""
        query = "SELECT 1 FROM visited WHERE url_canonical = ? AND last_status_code != 200 LIMIT 1"
        res = self.conn.execute(query, (url,)).fetchone()
        return res is not None

    def store_segments_scored(
        self,
        document_id: str,
        segments: Iterable[Any],
        scorer,  # callable(text)->(score,hits,neg)
    ) -> int:
        """
        Stores segments but computes impact_score/hit_count/is_negative without
        mutating Segment objects (needed because Segment is frozen+slots).
        """
        rows: List[
            Tuple[
                str, str, int, str, str, str, Optional[str],
                Optional[int], Optional[int], Optional[int]
            ]
        ] = []
        sha256 = hashlib.sha256

        for seg in segments:
            text = str(getattr(seg, "text", "") or "")
            norm = text.strip()
            if not norm:
                continue

            seg_hash = sha256(norm.encode("utf-8")).hexdigest()

            score, hits, neg = scorer(text)

            rows.append(
                (
                    str(uuid.uuid4()),
                    document_id,
                    int(getattr(seg, "order_index", 0) or 0),
                    str(getattr(seg, "segment_type", "") or ""),
                    text,
                    seg_hash,
                    getattr(seg, "page_ref", None),
                    int(score),
                    int(hits),
                    int(neg),
                    getattr(seg, "heading_context", None),
                )
            )

        if not rows:
            return 0

        before = self.conn.total_changes
        self.conn.executemany(self._sql_insert_segment, rows)
        after = self.conn.total_changes
        return max(0, after - before)