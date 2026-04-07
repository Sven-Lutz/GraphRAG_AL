# crawler/core/engine.py
from __future__ import annotations

import time
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Set, Tuple
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from crawler.core.canonical import Canonicalizer
from crawler.core.models import CrawlTask, FetchResult
from crawler.core.parsers.html_parser import parse_html
from crawler.core.parsers.pdf_parser import parse_pdf
from crawler.core.scheduler import PriorityScheduler
from crawler.core.storage import Storage, default_worker_id

from urllib.parse import urljoin as _urljoin

_HIGH_IMPACT_WORDS = [
    r'klima\w*', r'klimaschutz\w*', r'klimaanpassung\w*', r'klimaneutral\w*',
    r'co2\b', r'thg\b', r'treibhausgas\w*', r'emission\w*', r'treibhausgasbilanz\w*',
    r'energiebericht\w*', r'klimabericht\w*', r'klimaschutzkonzept\w*', r'klimafahrplan\w*',
    r'energie-?\s?und\s?klimakonzept\w*', r'energie\w*', r'erneuerbar\w*', r'fernw(?:ä|ae)rme\w*',
    r'nahw(?:ä|ae)rme\w*', r'w(?:ä|ae)rmenetz\w*', r'bhkw\b', r'blockheizkraftwerk\w*', r'w(?:ä|ae)rmepump\w*',
    r'geotherm\w*', r'biogas\w*', r'wasserkraft\w*', r'wasserstoff\b', r'h2\b',
    r'solar\w*', r'photovoltaik\w*', r'pv\b', r'windkraft\w*', r'windenergie\w*',
    r'speicher\w*', r'stromnetz\w*', r'smart\s?grid\w*', r'led\b', r'strombeschaffung\w*',
    r'sanier\w*', r'd(?:ä|ae)mm\w*', r'energieeffizienz\w*', r'passivhaus\w*',
    r'effizienzhaus\w*', r'kfw-\d+', r'geg\b', r'geb(?:ä|ae)udeenergiegesetz\w*',
    r'quartierskonzept\w*', r'fl(?:ä|ae)chennutzungsplan\w*', r'fnp\b', r'bebauungsplan\w*',
    r'nachverdichtung\w*', r'mobilit(?:ä|ae)t\w*', r'verkehr\w*', r'verkehrswende\w*', r'emobilit(?:ä|ae)t\w*',
    r'elektromobilit(?:ä|ae)t\w*', r'radverkehr\w*', r'radweg\w*', r'fahrrad\w*',
    r'verkehrsentwicklungsplan\w*', r'vep\b', r'lades(?:ä|ae)ul\w*', r'wallbox\w*',
    r'ladinfrastruktur\w*', r'carsharing\w*', r'(?:ö|oe)pnv\b', r'f(?:ö|oe)rder\w*',
    r'zuschuss\w*', r'mittel\w*', r'finanz\w*', r'investition\w*',
    r'projekttr(?:ä|ae)ger\w*', r'f(?:ö|oe)rderprogramm\w*', r'bewilligungsbescheid\w*',
    r'eigenanteil\w*', r'nki\b', r'kfw\b', r'bafa\b', r'efre\b', r'eler\b',
    r'eu-?f(?:ö|oe)rder\w*', r'beschluss\w*', r'beschlussvorlage\w*', r'antrag\w*', r'drucksache\w*',
    r'vorlage\w*', r'tagesordnungspunkt\w*', r'sitzung\w*', r'niederschrift\w*',
    r'protokoll\w*', r'klimabeirat\w*', r'umweltbeirat\w*', r'stadtwerk\w*'
]
HIGH_IMPACT_REGEX = re.compile(r'\b(?:' + '|'.join(_HIGH_IMPACT_WORDS) + r')', re.IGNORECASE)

_NEGATIVE_WORDS = [
    r'impressum\b', r'datenschutz\b', r'barrierefreiheit\b',
    r'stellenangebot\w*', r'(?:ö|oe)ffnungszeiten\b', r'kontakt\b'
]
NEGATIVE_REGEX = re.compile(r'\b(?:' + '|'.join(_NEGATIVE_WORDS) + r')', re.IGNORECASE)

def _segment_features(text: str) -> Tuple[int, int, int]:
    """
    Returns (impact_score, hit_count, is_negative) based on Engine's domain regex.
    This score is meant to be simple, explainable, and stable for paper methodology.
    """
    t = text or ""
    hits = len(HIGH_IMPACT_REGEX.findall(t))
    neg = 1 if NEGATIVE_REGEX.search(t) else 0

    # Strongly weight topical hits, penalize boilerplate-like segments.
    score = 25 * hits - (80 if neg else 0)

    # Slight preference for semantically richer segments.
    score += min(len(t) // 250, 8)

    return score, hits, neg


try:
    from crawler.core.traps import TrapDetector
    _trap_detector = TrapDetector(
        block_extensions=["jpg", "jpeg", "png", "gif", "svg", "mp4", "zip", "rar", "css", "js", "xml", "exe", "doc", "docx"],
        block_path_patterns=[
            '/kalender', 'veranstaltungen', 'termine', 'sitzungskalender',
            'monat=', 'jahr=', 'month=', 'year=', 'datum=',
            'weinabend', 'kirchencafe', 'sprechstunde', 'jahreshauptversammlung',
            'regio-cup', 'firmung', 'vereinsmeisterschaft',
            'print=', 'drucken=', 'ansicht=druck', 'type=print',
            'sort=', 'order=', 'orderby=', 'sortierung=',
            '/galerie', '/bilder', 'gallery', 'bildarchiv',
            '/login', '/register', '/buergerservice/anmeldung', 'warenkorb',
            'wasseruntersuchung', 'trinkwasser', 'prüfbericht', 'befund', 'analytik',
            'abfallkalender', 'abfuhrkalender', 'müllgebühren', 'abfalltrennung',
            'ferienprogramm', 'kindergarten', 'kinderhaus', 'schule', 'oberschule', 'bos-flyer',
            'wahlergebnisse', 'wahlvorschläge', 'stimmzettel', 'bekanntmachung',
            'umleitungsstrecke', 'baustelle', 'veranstaltungskalender'
        ],
        pagination_tokens=["page=", "offset=", "start=", "/page/"]
    )
except ImportError:
    _trap_detector = None

def _is_trap(url: str, depth: int) -> bool:
    if _trap_detector is None:
        return False
    return _trap_detector.should_block(url, depth)


@dataclass(frozen=True)
class EngineLimits:
    max_depth: int = 6
    max_pages_per_muni: int = 2000
    max_file_size_mb: int = 200


class Engine:
    def __init__(
        self,
        keywords: dict,
        limits: EngineLimits = EngineLimits(),
        user_agent: str = "KIT-ClimatePolicyCrawler/1.0 (research)",
        request_timeout_seconds: int = 20,
        max_redirects: int = 10,
        respect_robots_txt: bool = False,
        min_delay_seconds_per_domain: float = 0.2,
        canonicalizer: Optional[Canonicalizer] = None,
        allowed_domains_by_muni: Optional[Mapping[str, Set[str]]] = None,
        worker_id: Optional[str] = None,
        seed_additional_paths: Optional[list[str]] = None,
    ) -> None:
        self.storage = Storage()
        self.scheduler = PriorityScheduler()
        self.keywords = keywords
        self.limits = limits

        self.worker_id = worker_id or default_worker_id()

        self.timeout_connect = 5.0
        self.timeout_read = float(request_timeout_seconds)
        self.max_redirects = int(max_redirects)
        self.respect_robots_txt = bool(respect_robots_txt)
        self.min_delay_seconds_per_domain = float(min_delay_seconds_per_domain)

        self.canon = canonicalizer or Canonicalizer(
            strip_fragment=True,
            drop_query_prefixes=["utm_", "pk_"],
            drop_query_keys=["fbclid", "gclid", "session", "jsessionid", "sid", "phpsessid", "msclkid"],
            normalize_trailing_slash=True,
        )

        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=1)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        self._session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "text/html,application/pdf,application/xhtml+xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "de-DE,de;q=0.9,en;q=0.7",
            }
        )
        self._session.max_redirects = self.max_redirects

        self._last_request_ts_by_domain: Dict[str, float] = {}
        self._pages_by_muni: Dict[str, int] = {}

        self.allowed_domains_by_muni: Dict[str, Set[str]] = {
            str(k): {self._norm_domain(d) for d in v if d}
            for k, v in (allowed_domains_by_muni or {}).items()
        }
        self.seed_additional_paths: list[str] = list(seed_additional_paths or [])

    @staticmethod
    def _norm_domain(domain: str) -> str:
        d = (domain or "").strip().lower()
        if not d:
            return ""

        if "://" in d or d.startswith("//"):
            try:
                parsed = urlsplit(d if "://" in d else "http:" + d)
                d = parsed.netloc
            except Exception:
                pass
        d = d.split(":", 1)[0]
        if d.startswith("www."):
            d = d[4:]
        return d.rstrip(".")

    @staticmethod
    def _split(url: str):
        try:
            return urlsplit(url)
        except Exception:
            return None

    def _scheme_domain(self, url: str) -> Tuple[str, str]:
        s = self._split(url)
        if s is None:
            return "", ""
        return (s.scheme or "").lower(), self._norm_domain(s.netloc)

    def _is_allowed(self, muni_id: str, url: str) -> bool:
        scheme, domain = self._scheme_domain(url)
        if scheme not in ("http", "https") or not domain:
            return False
        allowed = self.allowed_domains_by_muni.get(muni_id)
        if not allowed:
            return False
        return domain in allowed

    def score(self, url: str, anchor: Optional[str]) -> int:
        u = (url or "").lower()
        a = (anchor or "").lower()
        score = 10

        if ".pdf" in u:
            score += 300

        ris_patterns = ["session", "bi/vo", "bi/si", "bi/kp", "allris", "ratsinfo", "ris."]
        if any(p in u for p in ris_patterns):
            score += 200

        if HIGH_IMPACT_REGEX.search(u):
            score += 150
        if a and HIGH_IMPACT_REGEX.search(a):
            score += 100

        if NEGATIVE_REGEX.search(u):
            score -= 50
        if a and NEGATIVE_REGEX.search(a):
            score -= 20

        return score

    def _polite_sleep(self, domain: str) -> None:
        if not domain or self.min_delay_seconds_per_domain <= 0:
            return
        now = time.time()
        last = self._last_request_ts_by_domain.get(domain)
        if last is not None:
            dt = now - last
            if dt < self.min_delay_seconds_per_domain:
                time.sleep(self.min_delay_seconds_per_domain - dt)
        self._last_request_ts_by_domain[domain] = time.time()

    def fetch(self, url: str) -> FetchResult:
        _, domain = self._scheme_domain(url)
        self._polite_sleep(domain)

        lim_bytes = int(self.limits.max_file_size_mb) * 1024 * 1024
        body = b""

        try:
            with self._session.get(
                url,
                timeout=(self.timeout_connect, self.timeout_read),
                allow_redirects=True,
                headers={"Connection": "close"},
                verify=False,
                stream=True,  # stream; enforce hard size limit below
            ) as resp:
                status = int(resp.status_code)
                headers = {str(k): str(v) for k, v in (resp.headers or {}).items()}
                content_type = str(resp.headers.get("Content-Type") or "") or None
                url_final = str(resp.url)

                cl = headers.get("Content-Length")
                if cl and cl.isdigit() and int(cl) > lim_bytes:
                    body = b""
                else:
                    data = bytearray()
                    for chunk in resp.iter_content(chunk_size=1024 * 64):
                        if not chunk:
                            break
                        data.extend(chunk)
                        if len(data) > lim_bytes:
                            data = bytearray()
                            break
                    body = bytes(data)

            return FetchResult(
                url_final=url_final,
                status_code=status,
                content_type=content_type,
                body=body,
                headers=headers,
            )

        except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout) as e:
            raise RuntimeError(f"timeout:{type(e).__name__}") from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"request_error:{type(e).__name__}:{e}") from e

    def _over_size_limit(self, fr: FetchResult) -> bool:
        lim_bytes = int(self.limits.max_file_size_mb) * 1024 * 1024
        cl = fr.headers.get("Content-Length")
        if cl:
            try:
                return int(cl) > lim_bytes
            except ValueError:
                return False
        return False

    @staticmethod
    def _looks_like_html(url: str, content_type: Optional[str]) -> bool:
        ctype = (content_type or "").lower()
        if "text/html" in ctype or "application/xhtml" in ctype:
            return True
        u = (url or "").lower()
        return u.endswith((".html", ".htm"))

    @staticmethod
    def _looks_like_pdf(url: str, content_type: Optional[str]) -> bool:
        ctype = (content_type or "").lower()
        if "application/pdf" in ctype:
            return True
        u = (url or "").lower()
        return u.endswith(".pdf")

    def run_claimed_batch(self, batch_size: int = 1) -> None:
        seeds = self.storage.claim_next_seed_jobs(worker_id=self.worker_id, limit=batch_size)
        if not seeds:
            print("[engine] no pending seed_jobs", flush=True)
            return
        self.run(seeds)

    def run(self, seeds: Iterable[Tuple[str, str]]) -> None:
        canon = self.canon.normalize
        max_depth = int(self.limits.max_depth)
        max_pages = int(self.limits.max_pages_per_muni)
        lim_bytes = int(self.limits.max_file_size_mb) * 1024 * 1024
        autostop_threshold = 300

        # Materialise seeds into a list so the iterable can be traversed
        # twice (seed enqueue + finish_seed_job loop) even when passed as a generator.
        seeds = list(seeds)

        # pages_since_last_pdf is tracked per-municipality to prevent one
        # PDF-poor municipality from triggering autostop for the next one.
        pages_since_last_pdf: Dict[str, int] = {}

        for muni_id, url in seeds:
            muni_id = str(muni_id)
            self._pages_by_muni.setdefault(muni_id, 0)

            if muni_id not in self.allowed_domains_by_muni:
                _, d = self._scheme_domain(url)
                if d:
                    self.allowed_domains_by_muni[muni_id] = {d}

            url_c_seed = canon(url) or url
            if not self._is_allowed(muni_id, url):
                self.storage.mark_visited(url_c_seed, -1, "seed out of scope")
                continue

            self.scheduler.enqueue(CrawlTask(muni_id, url, depth=0), 100)

            # Enqueue additional seed paths (e.g. /klima, /energie, /haushalt)
            for path in self.seed_additional_paths:
                extra_url = _urljoin(url.rstrip("/") + "/", path.lstrip("/"))
                extra_c = canon(extra_url)
                if extra_c and self._is_allowed(muni_id, extra_c):
                    self.scheduler.enqueue(
                        CrawlTask(muni_id, extra_c, depth=1, parent_url=url_c_seed),
                        80,
                    )

        while self.scheduler.has_next():
            task = self.scheduler.next()
            muni_pdf_count = pages_since_last_pdf.get(task.municipality_id, 0)

            if muni_pdf_count >= autostop_threshold:
                print(f"🛑 Autostop für {task.municipality_id}: Seit {autostop_threshold} Seiten kein PDF. Breche Kommune ab.", flush=True)
                # Drain remaining tasks for this municipality from the scheduler
                # by simply skipping — the per-muni page count won't advance further.
                continue

            if task.depth > max_depth:
                continue
            if not self._is_allowed(task.municipality_id, task.url):
                continue
            if _is_trap(task.url, task.depth):
                continue

            pages = self._pages_by_muni.get(task.municipality_id, 0)
            if pages >= max_pages:
                continue

            url_c = canon(task.url)
            if not url_c:
                continue

            if self.storage.is_visited(url_c):
                doc_id = self.storage.get_document_id_by_canonical_url(url_c)
                if doc_id is not None:
                    self.storage.link_document_to_municipality(task.municipality_id, doc_id)

                if self.storage.is_visited_with_error(url_c):
                    print(f"[retry] {task.url} (previously errored)", flush=True)
                else:
                    print(f"[skip] {task.url} (already visited)", flush=True)
                    continue

            try:
                print(f"[fetch] depth={task.depth} muni={task.municipality_id} url={task.url}", flush=True)

                fr = self.fetch(task.url)
                
                muni = task.municipality_id
                if self._looks_like_pdf(task.url, fr.content_type):
                    pages_since_last_pdf[muni] = 0
                else:
                    pages_since_last_pdf[muni] = pages_since_last_pdf.get(muni, 0) + 1

                status = int(fr.status_code)

                if not self._is_allowed(task.municipality_id, fr.url_final):
                    self.storage.mark_visited(url_c, status, "redirect out of scope")
                    continue

                if self._over_size_limit(fr):
                    self.storage.mark_visited(url_c, status, "oversize")
                    continue

                if not fr.body or len(fr.body) > lim_bytes:
                    self.storage.mark_visited(url_c, status, "body error/limit")
                    continue

                with self.storage.transaction():
                    doc_id = self.storage.store_raw(task.municipality_id, url_c, fr)

                    if self._looks_like_html(task.url, fr.content_type):
                        parse_result = parse_html(fr, fr.url_final)

                        if len(parse_result.segments) > 0:
                            self.storage.store_segments_scored(doc_id, parse_result.segments, _segment_features)

                        # Store HTML meta tags (description, date, author, ...)
                        if parse_result.meta:
                            self.storage.update_document_meta(doc_id, parse_result.meta)

                        # Persist outlinks for GraphRAG document link graph
                        if parse_result.out_links:
                            canonical_links = []
                            for link in parse_result.out_links:
                                link_url = link[0] if isinstance(link, tuple) else getattr(link, "url", link)
                                anchor = link[1] if isinstance(link, tuple) else getattr(link, "anchor", "")
                                lc = canon(link_url)
                                if lc:
                                    canonical_links.append((lc, anchor))
                            self.storage.store_outlinks(doc_id, canonical_links)

                        next_depth = task.depth + 1
                        if next_depth <= max_depth:
                            for link in parse_result.out_links:
                                link_url = getattr(link, "url", link[0] if isinstance(link, tuple) else link)
                                anchor = getattr(link, "anchor", link[1] if isinstance(link, tuple) else "")

                                link_c = canon(link_url)
                                if not link_c or not self._is_allowed(task.municipality_id, link_c):
                                    continue

                                prio = self.score(link_c, anchor)
                                self.scheduler.enqueue(
                                    CrawlTask(
                                        task.municipality_id,
                                        link_c,
                                        depth=next_depth,
                                        parent_url=url_c,
                                        anchor_text=anchor,
                                    ),
                                    prio,
                                )

                    elif self._looks_like_pdf(task.url, fr.content_type):
                        print(f"[parse] Extrahiere PDF: {fr.url_final}", flush=True)
                        parse_result = parse_pdf(fr, fr.url_final)

                        if len(parse_result.segments) > 0:
                            self.storage.store_segments_scored(doc_id, parse_result.segments, _segment_features)

                    self.storage.mark_visited(url_c, status)

                self._pages_by_muni[task.municipality_id] = pages + 1

            except Exception as e:
                self.storage.mark_visited(url_c, -1, str(e))

        for muni_id, _ in seeds:
            try:
                self.storage.finish_seed_job(str(muni_id), ok=True)
            except Exception:
                pass