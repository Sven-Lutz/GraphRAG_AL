# crawler/core/parsers/html_parser.py
from __future__ import annotations

import re
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from crawler.core.models import ParseResult, Segment, FetchResult

MIN_SEGMENT_LENGTH = 30

_ASSET_EXT_PATTERN = r"\.(jpg|jpeg|png|gif|webp|svg|mp4|mp3|avi|mov|mkv|zip|rar|7z|tar|gz|css|js|woff|woff2|ttf|ico)(?:\?|$)"
_RE_ASSET = re.compile(_ASSET_EXT_PATTERN, re.IGNORECASE)

DROP_TAGS_AND_SECTIONS = ["script", "style", "noscript", "svg", "nav", "footer", "aside"]

_RE_WS = re.compile(r"\s+")
_RE_MULTI_SLASH = re.compile(r"/{2,}")


def _clean_text(text: str) -> str:
    return _RE_WS.sub(" ", text or "").strip()


def _build_heading_context(path: list[tuple[int, str]]) -> str | None:
    """Build a breadcrumb string from the current heading path, e.g. 'Klimaschutz > Solarenergie'."""
    if not path:
        return None
    return " > ".join(txt for _, txt in path)


def parse_html(fetch_result: FetchResult, base_url: str) -> ParseResult:
    body = fetch_result.body or b""
    if not body:
        return ParseResult(text="", segments=[], out_links=[], meta={})

    try:
        soup = BeautifulSoup(body, "lxml")
    except Exception:
        try:
            soup = BeautifulSoup(body, "html.parser")
        except Exception:
            return ParseResult(text="", segments=[], out_links=[], meta={})

    for tag in soup.find_all(DROP_TAGS_AND_SECTIONS):
        tag.decompose()

    segments: list[Segment] = []
    order = 0
    min_len = int(MIN_SEGMENT_LENGTH)

    # Track heading hierarchy as a list of (level, text) pairs
    heading_path: list[tuple[int, str]] = []

    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
        txt = _clean_text(tag.get_text(" ", strip=True))
        if not txt or len(txt) < min_len:
            continue

        name = tag.name or ""
        if name.startswith("h") and len(name) == 2 and name[1].isdigit():
            level = int(name[1])
            # Drop any deeper headings from the path, then append this one
            heading_path = [(l, t) for l, t in heading_path if l < level]
            heading_path.append((level, txt))
            seg_type = "heading"
            heading_ctx = None  # headings don't need context — they ARE context
        elif name == "li":
            seg_type = "list_item"
            heading_ctx = _build_heading_context(heading_path)
        else:
            seg_type = "paragraph"
            heading_ctx = _build_heading_context(heading_path)

        segments.append(Segment(
            order_index=order,
            segment_type=seg_type,
            text=txt,
            heading_context=heading_ctx,
        ))
        order += 1

    links: list[tuple[str, str]] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = str(a.get("href") or "").strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue

        abs_url = urljoin(base_url, href)

        abs_url = abs_url.split("#")[0]

        if not abs_url.startswith(("http://", "https://")):
            continue

        if _RE_ASSET.search(abs_url):
            continue

        if abs_url in seen:
            continue
        seen.add(abs_url)

        if "//" in abs_url[8:]:  # 8: skip "https://"
            try:
                p = urlsplit(abs_url)
                if p.path and "//" in p.path:
                    abs_url = abs_url.replace(p.path, _RE_MULTI_SLASH.sub("/", p.path), 1)
            except Exception:
                pass

        anchor = _clean_text(a.get_text(" ", strip=True))
        links.append((abs_url, anchor))

    title_tag = soup.find("title")
    title = _clean_text(title_tag.get_text()) if title_tag else None

    full_text = _clean_text(soup.get_text(" ", strip=True))

    meta = {}
    if title:
        meta["title"] = title
    if fetch_result.content_type:
        meta["content_type"] = fetch_result.content_type

    # Extract structured metadata from <meta> tags — free enrichment for GraphRAG
    _META_MAP = {
        "description": "description",
        "og:description": "description",
        "og:title": "og_title",
        "article:published_time": "date",
        "date": "date",
        "dcterms.date": "date",
        "dcterms.modified": "date_modified",
        "author": "author",
        "dcterms.creator": "author",
        "keywords": "keywords",
        "og:type": "og_type",
    }
    for meta_tag in soup.find_all("meta"):
        attr_name = (meta_tag.get("name") or meta_tag.get("property") or "").lower().strip()
        content = (meta_tag.get("content") or "").strip()
        if not attr_name or not content:
            continue
        mapped = _META_MAP.get(attr_name)
        if mapped:
            meta.setdefault(mapped, content)

    return ParseResult(
        text=full_text,
        segments=segments,
        out_links=links,
        meta=meta,
    )