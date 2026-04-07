from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import httpx


WIKIPEDIA_API = "https://de.wikipedia.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

WIKI_PAGE = "Liste_der_Städte_und_Gemeinden_in_Bayern"
TARGET_SECTION = "Alle politisch selbständigen Städte und Gemeinden"

UA = "wi2026-registry/1.3 (academic research; mailto:bt716151@uni-bayreuth.de)"
HTTP_HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
}
SPARQL_HEADERS = {
    "Accept": "application/sparql-results+json",
    "Content-Type": "application/x-www-form-urlencoded",
    "User-Agent": UA,
}

RB_SET = {
    "Oberbayern",
    "Niederbayern",
    "Oberpfalz",
    "Oberfranken",
    "Mittelfranken",
    "Unterfranken",
    "Schwaben",
}

KREISFREI_HINTS = ("kreisfreie stadt",)
LANDKREIS_HINTS_PREFIX = ("landkreis ",)
LANDKREIS_HINTS_INFIX = ("kreisfreie stadt",)


@dataclass(frozen=True)
class WikiEntry:
    title: str
    url: str
    is_city: bool


@dataclass(frozen=True)
class PageMeta:
    title: str
    url: str
    pageid: int
    revid: int
    qid: str
    is_city: bool


@dataclass(frozen=True)
class MunicipalityRow:
    ags: str
    name: str
    is_kreisfrei: int
    bundesland: str
    regierungsbezirk: str
    landkreis: str
    population: str
    population_date: str
    homepage_url: str
    allowed_domains: str
    lat: str
    lon: str
    wikipedia_url: str
    wikidata_qid: str
    wikipedia_revision: int
    last_checked: str
    source: str


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def cache_get(cache_dir: Path, key: str) -> Optional[dict[str, Any]]:
    p = cache_dir / f"{sha1(key)}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def cache_put(cache_dir: Path, key: str, data: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{sha1(key)}.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_name(name: str) -> str:
    n = re.sub(r"\s+", " ", name).strip()
    n = n.replace("–", "-").replace("—", "-")
    return n


def as_str(x: Any) -> str:
    return str(x) if x is not None else ""


def normalize_ags(x: Any) -> str:
    """
    AGS must be an 8-digit string (TEXT). Never treat it as int.
    This prevents loss of leading zeros (e.g. Bayern '09......').
    """
    s = as_str(x).strip()
    if not s:
        return ""
    s = re.sub(r"\D+", "", s)
    if re.fullmatch(r"\d{7}", s):
        s = s.zfill(8)
    return s if re.fullmatch(r"\d{8}", s) else ""


def normalize_http_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    if u.startswith("//"):
        u = "https:" + u
    try:
        p = urlparse(u)
    except Exception:
        return ""
    if p.scheme not in ("http", "https"):
        return ""
    if not p.hostname:
        return ""
    return u


def allowed_domains_from_url(url: str) -> str:
    host = (urlparse(url).hostname or "").lower().strip(".")
    if not host:
        return ""
    s = {host}
    if host.startswith("www."):
        s.add(host[4:])
    else:
        s.add("www." + host)
    return "|".join(sorted(s))


def _sleep_backoff(attempt: int, cap_s: float = 60.0, base: float = 1.0) -> None:
    sleep = min(cap_s, base * (2**attempt))
    sleep = sleep * (0.5 + random.random())
    time.sleep(sleep)


def _respect_retry_after(headers: dict[str, str] | httpx.Headers) -> bool:
    ra = headers.get("Retry-After")
    if not ra:
        return False
    try:
        s = int(ra)
    except Exception:
        return False
    if s > 0:
        time.sleep(min(120, s))
        return True
    return False


def fetch_wikipedia_wikitext(client: httpx.Client, cache_dir: Path) -> str:
    key = f"mw_wikitext::{WIKI_PAGE}"
    cached = cache_get(cache_dir, key)
    if cached and isinstance(cached.get("wikitext"), str) and len(cached["wikitext"]) > 1000:
        return cached["wikitext"]

    params = {
        "action": "parse",
        "format": "json",
        "page": WIKI_PAGE,
        "prop": "wikitext",
        "redirects": 1,
    }

    last_err: Optional[Exception] = None
    for attempt in range(10):
        try:
            r = client.get(WIKIPEDIA_API, params=params)
            if r.status_code == 429 and _respect_retry_after(r.headers):
                continue
            r.raise_for_status()
            data = r.json()

            if "error" in data:
                msg = str((data.get("error") or {}).get("info") or data["error"])
                raise RuntimeError(f"MediaWiki parse error: {msg}")

            wt = ((data.get("parse") or {}).get("wikitext") or {}).get("*")
            if not isinstance(wt, str) or len(wt) < 1000:
                raise RuntimeError("Wikitext missing/too short.")

            cache_put(cache_dir, key, {"wikitext": wt, "fetched_at": time.time()})
            return wt
        except Exception as e:
            last_err = e
            _sleep_backoff(attempt)

    raise RuntimeError(f"Failed to fetch Wikipedia wikitext after retries: {last_err}")


def extract_section_wikitext(wt: str, heading: str) -> str:
    pat = rf"^\s*==\s*{re.escape(heading)}\s*==\s*$"
    m = re.search(pat, wt, flags=re.MULTILINE)
    if not m:
        raise RuntimeError(f"Section not found: {heading}")

    start = m.end()
    m2 = re.search(r"^\s*==[^=].*?==\s*$", wt[start:], flags=re.MULTILINE)
    end = start + (m2.start() if m2 else len(wt) - start)
    return wt[start:end]


def parse_entries_from_wikitext_section(wikitext: str) -> list[WikiEntry]:
    section = extract_section_wikitext(wikitext, TARGET_SECTION)

    raw_links = re.findall(r"\[\[([^\[\]]+?)\]\]", section)
    titles: list[str] = []
    for raw in raw_links:
        title = raw.split("|", 1)[0].strip()
        if not title:
            continue
        if ":" in title:
            continue
        if title.startswith("#"):
            continue
        titles.append(title)

    seen: set[str] = set()
    out: list[WikiEntry] = []
    for t in titles:
        tt = normalize_name(t)
        if not tt:
            continue
        tl = tt.casefold()
        if tl.startswith(("liste ", "verwaltung")):
            continue
        if tt in seen:
            continue
        seen.add(tt)

        url = "https://de.wikipedia.org/wiki/" + tt.replace(" ", "_")
        out.append(WikiEntry(title=tt, url=url, is_city=False))

    if len(out) < 1500:
        raise RuntimeError(f"Parsed suspiciously few entries from section: {len(out)}")

    return out


def mw_api_query_pages(client: httpx.Client, titles: list[str]) -> dict[str, Any]:
    params = {
        "action": "query",
        "format": "json",
        "prop": "pageprops|revisions|info",
        "ppprop": "wikibase_item",
        "rvprop": "ids",
        "inprop": "url",
        "redirects": 1,
        "titles": "|".join(titles),
        "maxlag": 5,
    }

    last_err: Optional[Exception] = None
    for attempt in range(12):
        try:
            r = client.get(WIKIPEDIA_API, params=params)
            if r.status_code == 429 and _respect_retry_after(r.headers):
                continue
            r.raise_for_status()
            data = r.json()

            if isinstance(data, dict) and "error" in data:
                msg = str((data.get("error") or {}).get("info") or data["error"])
                raise RuntimeError(f"MediaWiki API error: {msg}")

            return data
        except Exception as e:
            last_err = e
            _sleep_backoff(attempt)

    raise RuntimeError(f"MediaWiki API failed after retries: {last_err}")


def titles_to_pagemeta(
    client: httpx.Client,
    entries: list[WikiEntry],
    cache_dir: Path,
    batch_size: int,
    polite_sleep_s: float = 0.10,
) -> list[PageMeta]:
    entry_by_title_cf = {e.title.casefold(): e for e in entries}
    titles = [e.title for e in entries]

    out: list[PageMeta] = []
    for i in range(0, len(titles), batch_size):
        chunk = titles[i : i + batch_size]
        chunk_key = sha1("|".join(chunk))
        key = f"mw_pages::{chunk_key}::{len(chunk)}"

        data = cache_get(cache_dir, key)
        if data is None:
            data = mw_api_query_pages(client, chunk)
            cache_put(cache_dir, key, data)

        pages = (data.get("query", {}) or {}).get("pages", {}) or {}
        for _, p in pages.items():
            if not isinstance(p, dict) or "missing" in p:
                continue

            title = as_str(p.get("title")).strip()
            pageid = int(p.get("pageid", 0) or 0)
            fullurl = as_str(p.get("fullurl")).strip()

            revid = 0
            revs = p.get("revisions")
            if isinstance(revs, list) and revs:
                revid = int((revs[0] or {}).get("revid", 0) or 0)

            pp = p.get("pageprops") or {}
            qid = as_str(pp.get("wikibase_item")).strip()

            if not title or not pageid or not revid or not qid:
                continue

            e = entry_by_title_cf.get(title.casefold())
            is_city = bool(e.is_city) if e else False
            url = fullurl or (e.url if e else "")
            if not url:
                url = "https://de.wikipedia.org/wiki/" + title.replace(" ", "_")

            out.append(
                PageMeta(
                    title=title,
                    url=url,
                    pageid=pageid,
                    revid=revid,
                    qid=qid,
                    is_city=is_city,
                )
            )

        time.sleep(polite_sleep_s)

    if not out:
        raise RuntimeError("0 metas from MediaWiki API (no pages/QIDs).")

    best: dict[str, PageMeta] = {}
    for m in out:
        prev = best.get(m.qid)
        if prev is None or (m.is_city and not prev.is_city):
            best[m.qid] = m

    return sorted(best.values(), key=lambda x: x.title.lower())


_SPARQL_PREFIXES = """\
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""


def sparql_post_json(client: httpx.Client, query: str, retries: int) -> dict[str, Any]:
    last_err: Optional[Exception] = None

    for attempt in range(max(1, retries)):
        try:
            r = client.post(WIKIDATA_SPARQL, data={"query": query})

            if r.status_code in (429, 500, 502, 503, 504):
                body = (r.text or "")[:800].replace("\n", " ")
                last_err = RuntimeError(f"SPARQL HTTP {r.status_code}: {body}")

                if r.status_code == 429 and _respect_retry_after(r.headers):
                    continue

                _sleep_backoff(attempt, cap_s=120.0, base=1.0)
                continue

            if r.status_code >= 400:
                body = (r.text or "")[:2000].replace("\n", " ")
                raise RuntimeError(f"SPARQL HTTP {r.status_code}: {body}")

            try:
                return r.json()
            except Exception as e:
                body = (r.text or "")[:2000].replace("\n", " ")
                raise RuntimeError(f"SPARQL non-JSON response (HTTP {r.status_code}): {body}") from e

        except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            last_err = e
            _sleep_backoff(attempt, cap_s=120.0, base=1.0)
        except Exception as e:
            last_err = e
            _sleep_backoff(attempt, cap_s=120.0, base=1.0)

    raise RuntimeError(f"SPARQL failed after retries: {last_err}")


def _maybe_set_landkreis(rec: dict[str, Any], lk_label: str) -> None:
    if not lk_label:
        return
    lk_cf = lk_label.casefold().strip()
    if lk_cf.startswith(LANDKREIS_HINTS_PREFIX) or any(h in lk_cf for h in LANDKREIS_HINTS_INFIX):
        if rec.get("landkreis") is None:
            rec["landkreis"] = lk_label


def _maybe_set_regierungsbezirk(rec: dict[str, Any], rb_label: str) -> None:
    if not rb_label:
        return
    if rb_label in RB_SET and rec.get("regierungsbezirk") is None:
        rec["regierungsbezirk"] = rb_label


def enrich_qids_bulk(
    client: httpx.Client,
    qids: list[str],
    cache_dir: Path,
    chunk_size: int,
    polite_sleep_s: float,
    retries: int,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    for i in range(0, len(qids), chunk_size):
        chunk = qids[i : i + chunk_size]
        chunk_key = sha1("|".join(chunk))
        key = f"sparql_enrich::{chunk_key}::{len(chunk)}"

        data = cache_get(cache_dir, key)
        if data is None:
            values = " ".join(f"wd:{q}" for q in chunk)

            query = f"""{_SPARQL_PREFIXES}
SELECT DISTINCT ?item ?itemLabel ?ags ?website ?coord ?pop ?popTime ?adminLabel WHERE {{
  VALUES ?item {{ {values} }}

  OPTIONAL {{ ?item wdt:P439 ?ags . }}
  OPTIONAL {{ ?item wdt:P856 ?website . }}
  OPTIONAL {{ ?item wdt:P625 ?coord . }}

  OPTIONAL {{
    ?item p:P1082 ?popStmt .
    ?popStmt ps:P1082 ?pop .
    OPTIONAL {{ ?popStmt pq:P585 ?popTime . }}
  }}

  OPTIONAL {{
    ?item wdt:P131 ?admin .
    ?admin rdfs:label ?adminLabel .
    FILTER(lang(?adminLabel) = "de")
  }}

  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "de". }}
}}
"""
            data = sparql_post_json(client, query, retries=retries)
            cache_put(cache_dir, key, data)

        rows = (data.get("results", {}) or {}).get("bindings", []) or []
        for b in rows:
            item_uri = as_str((b.get("item") or {}).get("value")).strip()
            qid = item_uri.rsplit("/", 1)[-1] if item_uri else ""
            if not qid:
                continue

            rec = out.setdefault(
                qid,
                {
                    "label_de": None,
                    "ags": None,
                    "website": None,
                    "lat": None,
                    "lon": None,
                    "population": None,
                    "population_date": None,
                    "regierungsbezirk": None,
                    "landkreis": None,
                },
            )

            if rec["label_de"] is None and "itemLabel" in b:
                rec["label_de"] = normalize_name(as_str(b["itemLabel"]["value"]))

            if rec["ags"] is None and "ags" in b:
                ags_norm = normalize_ags((b.get("ags") or {}).get("value"))
                if ags_norm:
                    rec["ags"] = ags_norm

            if rec["website"] is None and "website" in b:
                rec["website"] = normalize_http_url(as_str(b["website"]["value"]))

            if "coord" in b and (rec["lat"] is None or rec["lon"] is None):
                wkt = as_str(b["coord"]["value"])
                m = re.search(r"Point\(([-0-9.]+)\s+([-0-9.]+)\)", wkt)
                if m:
                    rec["lon"] = float(m.group(1))
                    rec["lat"] = float(m.group(2))

            if "pop" in b:
                pop_raw = as_str(b["pop"]["value"])
                try:
                    pop_val = int(float(pop_raw))
                except Exception:
                    pop_val = None
                pop_time = as_str((b.get("popTime") or {}).get("value")).strip() or None
                if pop_val is not None:
                    if rec["population"] is None:
                        rec["population"] = pop_val
                        rec["population_date"] = pop_time
                    else:
                        if pop_time and (rec["population_date"] is None or pop_time > rec["population_date"]):
                            rec["population"] = pop_val
                            rec["population_date"] = pop_time

            admin_label = as_str((b.get("adminLabel") or {}).get("value")).strip()
            if admin_label:
                _maybe_set_regierungsbezirk(rec, admin_label)
                _maybe_set_landkreis(rec, admin_label)

        time.sleep(polite_sleep_s)

    return out


def enrich_rb_bulk(
    client: httpx.Client,
    qids: list[str],
    cache_dir: Path,
    chunk_size: int,
    polite_sleep_s: float,
    retries: int,
) -> dict[str, str]:
    out_rb: dict[str, str] = {}
    rb_values = " ".join(f"\"{rb}\"@de" for rb in sorted(RB_SET))

    for i in range(0, len(qids), chunk_size):
        chunk = qids[i : i + chunk_size]
        chunk_key = sha1("|".join(chunk))
        key = f"sparql_rb::{chunk_key}::{len(chunk)}"

        data = cache_get(cache_dir, key)
        if data is None:
            values = " ".join(f"wd:{q}" for q in chunk)
            query = f"""{_SPARQL_PREFIXES}
SELECT DISTINCT ?item ?rbLabel WHERE {{
  VALUES ?item {{ {values} }}
  ?item wdt:P131* ?x .
  ?x rdfs:label ?rbLabel .
  FILTER(lang(?rbLabel) = "de")
  VALUES ?rbLabel {{ {rb_values} }}
}}
"""
            data = sparql_post_json(client, query, retries=retries)
            cache_put(cache_dir, key, data)

        rows = (data.get("results", {}) or {}).get("bindings", []) or []
        for b in rows:
            item_uri = as_str((b.get("item") or {}).get("value")).strip()
            qid = item_uri.rsplit("/", 1)[-1] if item_uri else ""
            rb_label = as_str((b.get("rbLabel") or {}).get("value")).strip()
            if qid and rb_label and qid not in out_rb:
                out_rb[qid] = rb_label

        time.sleep(polite_sleep_s)

    return out_rb


CSV_FIELDS = [
    "ags",
    "name",
    "is_kreisfrei",
    "bundesland",
    "regierungsbezirk",
    "landkreis",
    "population",
    "population_date",
    "homepage_url",
    "allowed_domains",
    "lat",
    "lon",
    "wikipedia_url",
    "wikidata_qid",
    "wikipedia_revision",
    "last_checked",
    "source",
]


def write_municipalities_csv(path: Path, rows: list[MunicipalityRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            d = {k: getattr(r, k) for k in CSV_FIELDS}
            d["ags"] = normalize_ags(d.get("ags")) or ""
            d["allowed_domains"] = "|".join(sorted({x.strip().lower() for x in (d.get("allowed_domains") or "").split("|") if x.strip()}))
            w.writerow(d)
    tmp.replace(path)


def write_municipalities_sqlite(db_path: Path, rows: list[MunicipalityRow]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(db_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    con = sqlite3.connect(str(tmp))
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")

    cur.execute(
        """
        CREATE TABLE municipalities (
            ags TEXT PRIMARY KEY,
            name TEXT,
            is_kreisfrei INTEGER,
            bundesland TEXT,
            regierungsbezirk TEXT,
            landkreis TEXT,
            population TEXT,
            population_date TEXT,
            homepage_url TEXT,
            allowed_domains TEXT,
            lat TEXT,
            lon TEXT,
            wikipedia_url TEXT,
            wikidata_qid TEXT,
            wikipedia_revision INTEGER,
            last_checked TEXT,
            source TEXT
        )
        """
    )

    cur.executemany(
        """
        INSERT INTO municipalities VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        [
            (
                normalize_ags(r.ags),  # Hard guarantee in DB, too
                r.name,
                r.is_kreisfrei,
                r.bundesland,
                r.regierungsbezirk,
                r.landkreis,
                r.population,
                r.population_date,
                r.homepage_url,
                "|".join(sorted({x.strip().lower() for x in (r.allowed_domains or "").split("|") if x.strip()})),
                r.lat,
                r.lon,
                r.wikipedia_url,
                r.wikidata_qid,
                r.wikipedia_revision,
                r.last_checked,
                r.source,
            )
            for r in rows
        ],
    )

    cur.execute("CREATE INDEX idx_muni_rb ON municipalities(regierungsbezirk);")
    cur.execute("CREATE INDEX idx_muni_lk ON municipalities(landkreis);")

    con.commit()
    con.close()
    tmp.replace(db_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="crawler/data/cache/municipal_registry")
    ap.add_argument("--out_csv", default="crawler/data/seeds/municipalities.csv")
    ap.add_argument("--out_sqlite", default="crawler/data/db/municipalities.sqlite")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--mw_batch", type=int, default=50)
    ap.add_argument("--sparql_chunk", type=int, default=20)
    ap.add_argument("--sparql_sleep", type=float, default=0.8)
    ap.add_argument("--sparql_retries", type=int, default=20)
    ap.add_argument("--rb_chunk", type=int, default=80)

    args = ap.parse_args()
    cache_dir = Path(args.cache_dir)
    last_checked = date.today().isoformat()

    timeout_mw = httpx.Timeout(args.timeout, connect=args.timeout)
    timeout_wd = httpx.Timeout(args.timeout * 6, connect=args.timeout * 3)
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)

    with httpx.Client(headers=HTTP_HEADERS, timeout=timeout_mw, limits=limits) as mw_client, httpx.Client(
        headers=SPARQL_HEADERS, timeout=timeout_wd, limits=limits
    ) as wd_client:
        wt = fetch_wikipedia_wikitext(mw_client, cache_dir)
        entries = parse_entries_from_wikitext_section(wt)
        print(f"[1] entries={len(entries)}")

        metas = titles_to_pagemeta(
            mw_client,
            entries,
            cache_dir=cache_dir,
            batch_size=args.mw_batch,
            polite_sleep_s=0.10,
        )
        print(f"[2] metas={len(metas)}")
        qids = [m.qid for m in metas]

        enrich = enrich_qids_bulk(
            wd_client,
            qids=qids,
            cache_dir=cache_dir,
            chunk_size=args.sparql_chunk,
            polite_sleep_s=args.sparql_sleep,
            retries=args.sparql_retries,
        )

        missing_rb = [q for q in qids if enrich.get(q, {}).get("regierungsbezirk") is None]
        if missing_rb:
            rb_map = enrich_rb_bulk(
                wd_client,
                qids=missing_rb,
                cache_dir=cache_dir,
                chunk_size=args.rb_chunk,
                polite_sleep_s=args.sparql_sleep,
                retries=args.sparql_retries,
            )
            for qid, rb in rb_map.items():
                rec = enrich.get(qid)
                if rec and rec.get("regierungsbezirk") is None and rb in RB_SET:
                    rec["regierungsbezirk"] = rb

        print(f"[3] enrich_qids={len(enrich)}")

    rows: list[MunicipalityRow] = []
    missing_ags = 0

    for m in metas:
        e = enrich.get(m.qid, {})
        ags = normalize_ags(e.get("ags"))
        if not ags:
            missing_ags += 1
            continue

        name = as_str(e.get("label_de")).strip() or normalize_name(m.title)
        if re.fullmatch(r"Q\d+", name):
            name = normalize_name(m.title)

        homepage = normalize_http_url(as_str(e.get("website")))
        allowed = allowed_domains_from_url(homepage) if homepage else ""

        rb = as_str(e.get("regierungsbezirk")).strip()
        lk = as_str(e.get("landkreis")).strip()

        is_kreisfrei = 0
        lk_cf = lk.casefold().strip()
        if not lk:
            is_kreisfrei = 1
        elif any(h in lk_cf for h in KREISFREI_HINTS):
            is_kreisfrei = 1
        if is_kreisfrei == 1:
            lk = ""

        pop = e.get("population")
        pop_s = str(pop) if isinstance(pop, int) else ""
        pop_time = as_str(e.get("population_date")).strip()

        lat = e.get("lat")
        lon = e.get("lon")
        lat_s = f"{lat:.6f}" if isinstance(lat, float) else ""
        lon_s = f"{lon:.6f}" if isinstance(lon, float) else ""

        rows.append(
            MunicipalityRow(
                ags=ags,
                name=name,
                is_kreisfrei=is_kreisfrei,
                bundesland="Bayern",
                regierungsbezirk=rb,
                landkreis=lk,
                population=pop_s,
                population_date=pop_time,
                homepage_url=homepage,
                allowed_domains=allowed,
                lat=lat_s,
                lon=lon_s,
                wikipedia_url=m.url,
                wikidata_qid=m.qid,
                wikipedia_revision=m.revid,
                last_checked=last_checked,
                source="wikipedia(wikitext section) + wikidata(P439,P856,P625,P1082,P585,P131) + wikidata(P131* RB fill)",
            )
        )

    rows.sort(key=lambda r: r.ags)

    if not rows:
        raise RuntimeError("0 rows produced (unexpected).")

    write_municipalities_csv(Path(args.out_csv), rows)
    write_municipalities_sqlite(Path(args.out_sqlite), rows)

    print(f"OK: rows={len(rows)} missing_ags={missing_ags}")
    print(f"CSV:    {args.out_csv}")
    print(f"SQLite: {args.out_sqlite}")


if __name__ == "__main__":
    main()
