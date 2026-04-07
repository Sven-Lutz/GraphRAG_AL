from __future__ import annotations

import argparse
from pathlib import Path

from crawler.core.seeds import load_seeds_from_sqlite, upsert_seed_jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="crawler/data/db/crawl.sqlite")
    args = ap.parse_args()

    seeds, _ = load_seeds_from_sqlite()
    n = upsert_seed_jobs(seeds, crawl_db_path=Path(args.db))
    print(f"Upserted {n} seed_jobs into {args.db}")


if __name__ == "__main__":
    main()
