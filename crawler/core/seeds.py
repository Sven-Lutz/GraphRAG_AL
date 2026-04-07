from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

DEFAULT_MUNI_SQLITE_PATH = Path("crawler/data/db/municipalities.sqlite")
DEFAULT_CSV_PATH = Path("crawler/data/seeds/municipalities.csv")
DEFAULT_CRAWL_DB_PATH = Path("crawler/data/db/crawl.sqlite")

def _is_valid_url(url: str) -> bool:
    try:
        p = urlparse((url or "").strip())
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False

def _norm_domain(domain: str) -> str:
    d = (domain or "").strip().lower()
    if not d:
        return ""
    d = d.split(":", 1)[0].rstrip(".")
    if "://" in d:
        try:
            d = urlparse(d).netloc.lower()
            d = d.split(":", 1)[0].rstrip(".")
        except Exception:
            return ""
    return d

def _derive_domain_from_url(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return _norm_domain(host)
    except Exception:
        return ""

def _parse_allowed_domains(raw: Optional[str], homepage_url: str) -> Set[str]:
    domains: Set[str] = set()

    if raw:
        for d in str(raw).split("|"):
            dn = _norm_domain(d)
            if dn:
                domains.add(dn)

    if not domains:
        dn = _derive_domain_from_url(homepage_url)
        if dn:
            domains.add(dn)

    return domains

def load_seeds_from_sqlite(
    db_path: Path = DEFAULT_MUNI_SQLITE_PATH,
    limit: Optional[int] = None,
    start: int = 0,
) -> Tuple[List[Tuple[str, str]], Dict[str, Set[str]]]:
    if not db_path.exists():
        raise FileNotFoundError(f"municipalities.sqlite not found: {db_path}")

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) 
    
    query = """
        SELECT ags, homepage_url, allowed_domains
        FROM municipalities
        WHERE homepage_url IS NOT NULL AND homepage_url != ''
        ORDER BY ags
    """
    
    params = []
    if limit is not None or start > 0:
        query += " LIMIT ? OFFSET ?"
        safe_limit = limit if limit is not None else 99999999
        params.extend([safe_limit, max(0, start)])

    try:
        cur = con.execute(query, params)
        
        seeds_all: List[Tuple[str, str]] = []
        allowed: Dict[str, Set[str]] = {}

        for ags, homepage_url, allowed_domains in cur: # Lazy Iteration statt fetchall()
            muni_id = str(ags).strip()
            homepage = str(homepage_url).strip()

            if not muni_id or not _is_valid_url(homepage):
                continue

            seeds_all.append((muni_id, homepage))
            allowed[muni_id] = _parse_allowed_domains(
                str(allowed_domains) if allowed_domains is not None else None,
                homepage,
            )
            
        return seeds_all, allowed
    finally:
        con.close()


def load_seeds_from_csv(
    csv_path: Path = DEFAULT_CSV_PATH,
    limit: Optional[int] = None,
    start: int = 0,
    end: Optional[int] = None,
) -> Tuple[List[Tuple[str, str]], Dict[str, Set[str]]]:
    """
    CSV columns expected:
      - ags
      - homepage_url
      - allowed_domains (optional, pipe-separated)
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"municipalities.csv not found: {csv_path}")

    seeds_all: List[Tuple[str, str]] = []
    allowed: Dict[str, Set[str]] = {}

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            muni_id = str(row.get("ags", "")).strip()
            homepage = str(row.get("homepage_url", "")).strip()

            if not muni_id or not _is_valid_url(homepage):
                continue

            seeds_all.append((muni_id, homepage))
            allowed[muni_id] = _parse_allowed_domains(row.get("allowed_domains"), homepage)

    if start < 0:
        start = 0
    if end is None:
        sliced = seeds_all[start:]
    else:
        sliced = seeds_all[start:end]

    if limit is not None:
        sliced = sliced[: int(limit)]

    allowed_sliced = {m: allowed[m] for (m, _) in sliced}
    return sliced, allowed_sliced

def ensure_seed_jobs_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
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

def upsert_seed_jobs(
    seeds: Iterable[Tuple[str, str]],
    crawl_db_path: Path = DEFAULT_CRAWL_DB_PATH,
) -> int:
    """
    Inserts/updates seed_jobs in crawl.sqlite.
    Creates the seed_jobs schema if missing.
    Returns number of rows affected (best-effort via total_changes delta).
    """
    rows = [(str(m), str(u)) for (m, u) in seeds]
    if not rows:
        return 0

    print("upsert_seed_jobs -> crawl_db_path =", crawl_db_path)
    
    con = sqlite3.connect(
        str(crawl_db_path), 
        timeout=60.0, 
        isolation_level=None
    )
    
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA busy_timeout=60000;")
        
        ensure_seed_jobs_schema(con)  
        before = con.total_changes
        
        con.execute("BEGIN TRANSACTION;") 
        try:
            con.executemany(
                """
                INSERT INTO seed_jobs (municipality_id, seed_url, status, attempt_count)
                VALUES (?, ?, 'pending', 0)
                ON CONFLICT(municipality_id) DO UPDATE SET
                    seed_url=excluded.seed_url,
                    status=CASE
                        WHEN seed_jobs.status IN ('pending','failed') THEN 'pending'
                        ELSE seed_jobs.status
                    END
                """,
                rows,
            )
            con.execute("COMMIT;")
        except Exception:
            con.execute("ROLLBACK;")
            raise
            
        after = con.total_changes
        return max(0, after - before)
    finally:
        con.close()