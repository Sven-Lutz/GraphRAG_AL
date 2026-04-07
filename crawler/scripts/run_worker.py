# crawler/scripts/run_worker.py
from __future__ import annotations

import argparse
import sqlite3
import threading
import time
import logging
import sys
from datetime import timedelta

from crawler.core.engine import Engine, EngineLimits
from crawler.core.seed_jobs import (
    DEFAULT_CRAWL_DB_PATH,
    claim_next_job,
    default_worker_id,
    heartbeat_job,
    mark_done,
    mark_failed,
)
from crawler.core.seeds import load_seeds_from_sqlite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("worker.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

HEARTBEAT_EVERY_SECONDS = 30.0

KLIMA_KEYWORDS = {
    "positive": [
        "klima", "klimaschutz", "energie", "waermeplanung", "wärmeplanung", 
        "co2", "thg", "solar", "photovoltaik", "pv", "wind", 
        "förder", "foerder", "zuschuss", "mittel", "haushalt", "finanz", 
        "investition", "nki", "kfw", "eu-mittel", "efre", "beschluss", "vorlage"
    ],
    "negative": [
        "impressum", "datenschutz", "barrierefreiheit", "leichte-sprache", 
        "login", "karriere", "stellenangebot", "jubiläum", "veranstaltung",
        "tourismus", "museum", "sport"
    ]
}

KLIMA_LIMITS = EngineLimits(
    max_depth=12,
    max_pages_per_muni=25000,
    max_file_size_mb=200  # Auf 200 MB angehoben für große Pläne
)

# Additional seed paths from crawl.yaml — enqueued alongside each homepage
# to discover relevant climate/energy pages that may not be well-linked
SEED_ADDITIONAL_PATHS = [
    "/klima", "/klimaschutz", "/energie", "/waermeplanung",
    "/umwelt", "/mobilitaet", "/verkehr", "/bauen",
    "/foerderung", "/foerderprogramme", "/haushalt",
    "/ratsinfo", "/bekanntmachungen",
]

def _heartbeat_loop(db_path: str, municipality_id: str, worker_id: str, stop: threading.Event) -> None:
    con = sqlite3.connect(db_path, timeout=60.0, isolation_level=None)
    try:
        con.execute("PRAGMA busy_timeout=60000;")
        con.execute("PRAGMA journal_mode=WAL;")
        while not stop.wait(HEARTBEAT_EVERY_SECONDS):
            try:
                heartbeat_job(con, municipality_id, worker_id)
            except sqlite3.OperationalError:
                pass
            except Exception:
                pass
    finally:
        con.close()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_CRAWL_DB_PATH))
    ap.add_argument("--stale-after", type=int, default=15 * 60)
    ap.add_argument("--limit", type=int, default=None, help="Anzahl der zu bearbeitenden Kommunen.")
    args = ap.parse_args()

    worker_id = default_worker_id()
    _, allowed = load_seeds_from_sqlite()

    engine = Engine(
        keywords=KLIMA_KEYWORDS,
        limits=KLIMA_LIMITS,
        allowed_domains_by_muni=allowed,
        worker_id=worker_id,
        seed_additional_paths=SEED_ADDITIONAL_PATHS,
    )

    con = sqlite3.connect(args.db, timeout=60.0, isolation_level=None)
    con.execute("PRAGMA busy_timeout=60000;")
    con.execute("PRAGMA journal_mode=WAL;")

    jobs_processed = 0
    start_time = time.time()
    
    logger.info(f"🚀 Starte Industrial-Worker {worker_id} (High-Recall Mode)")
    if args.limit:
        logger.info(f"🎯 Ziel-Limit: {args.limit} Kommunen")

    try:
        while True:
            if args.limit and jobs_processed >= args.limit:
                logger.info(f"✅ Limit von {args.limit} erreicht. Worker beendet.")
                break

            job = claim_next_job(con, worker_id=worker_id, stale_after_seconds=args.stale_after)

            if job is None:
                logger.info("💤 Keine 'pending' Jobs gefunden. Beende.")
                break

            stop = threading.Event()
            hb = threading.Thread(
                target=_heartbeat_loop,
                args=(args.db, job.municipality_id, worker_id, stop),
                daemon=True,
            )
            hb.start()

            muni_start_time = time.time()
            try:
                logger.info(f"▶️ Bearbeite Gemeinde {job.municipality_id} ({job.seed_url})")
                
                engine.run([(job.municipality_id, job.seed_url)])

                # Resolve target_document_id in document_links so Neo4j LINKS_TO
                # edges can be exported (otherwise target_document_id stays NULL)
                resolved = engine.storage.resolve_document_links()
                if resolved:
                    logger.info(f"🔗 {resolved} document links resolved for {job.municipality_id}")

                mark_done(con, job.municipality_id, worker_id)
                
                duration = timedelta(seconds=int(time.time() - muni_start_time))
                logger.info(f"✅ Gemeinde {job.municipality_id} fertiggestellt. Dauer: {duration}.")

            except KeyboardInterrupt:
                logger.warning(f"⚠️ Strg+C gedrückt! Beende Gemeinde {job.municipality_id} unvollständig.")
                mark_failed(con, job.municipality_id, worker_id, "Abgebrochen durch Benutzer (SIGINT)")
                stop.set()
                hb.join(timeout=2.0)
                sys.exit(0)

            except Exception as e:
                mark_failed(con, job.municipality_id, worker_id, str(e))
                logger.error(f"❌ Fehler bei {job.municipality_id}: {e}", exc_info=True)

            finally:
                stop.set()
                hb.join(timeout=2.0)
                jobs_processed += 1

    finally:
        con.close()
        total_time = timedelta(seconds=int(time.time() - start_time))
        logger.info(f"🏁 Fertig: {jobs_processed} Kommunen in {total_time} verarbeitet.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)