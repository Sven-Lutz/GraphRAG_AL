from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional
from urllib.parse import urljoin, urlsplit, urldefrag

from bs4 import BeautifulSoup

_WHITESPACE_RE = re.compile(r"\s+")
_SCHEMES_BLOCK_RE = re.compile(r"^(?:mailto:|tel:|javascript:|data:)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Link:
    url: str
    anchor: str


class LinkExtractor:

    def __init__(
        self,
        block_extensions: Optional[Iterable[str]] = None,
        block_path_patterns: Optional[Iterable[str]] = None,
    ) -> None:
        self.block_extensions = frozenset(
            (ext or "").lower().lstrip(".")
            for ext in (block_extensions or [])
            if ext and str(ext).strip()
        )
        self.block_path_patterns = tuple(
            (p or "").lower()
            for p in (block_path_patterns or [])
            if p and str(p).strip()
        )

    def _clean_anchor(self, text: str) -> str:
        return _WHITESPACE_RE.sub(" ", text or "").strip()

    def _normalize_url(self, base_url: str, href: str) -> str:
        abs_url = urljoin(base_url, href)
        abs_url, _ = urldefrag(abs_url)
        return abs_url.strip()

    def _is_http(self, url: str) -> bool:
        try:
            s = urlsplit(url)
        except Exception:
            return False
        return s.scheme in ("http", "https") and bool(s.netloc)

    def _is_blocked_extension(self, url: str) -> bool:
        if not self.block_extensions:
            return False
        path = (urlsplit(url).path or "").lower()
        if "." not in path:
            return False
        ext = path.rsplit(".", 1)[-1]
        return ext in self.block_extensions

    def _is_blocked_path(self, url: str) -> bool:
        if not self.block_path_patterns:
            return False
        path = (urlsplit(url).path or "").lower()
        return any(pat in path for pat in self.block_path_patterns)

    def _allowed_domain(self, host: str, allowed_domains: frozenset[str]) -> bool:
        host = host.split(":", 1)[0]
        return any(host == d or host.endswith("." + d) for d in allowed_domains)

    def extract_links(
        self,
        html: bytes,
        base_url: str,
        allowed_domains: Optional[set[str]] = None,
    ) -> list[Link]:
        allow = (
            frozenset(d.strip().lower() for d in allowed_domains if d and d.strip())
            if allowed_domains
            else frozenset()
        )

        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            return []

        out: list[Link] = []
        seen: set[str] = set()

        for tag in soup.find_all("a", href=True):
            href = str(tag.get("href", "")).strip()
            if not href or href.startswith("#") or _SCHEMES_BLOCK_RE.match(href):
                continue

            url = self._normalize_url(base_url, href)
            if not url or url in seen:
                continue

            if not self._is_http(url):
                continue

            if self._is_blocked_extension(url) or self._is_blocked_path(url):
                continue

            if allow:
                host = urlsplit(url).netloc.lower()
                if not self._allowed_domain(host, allow):
                    continue

            seen.add(url)
            anchor = self._clean_anchor(tag.get_text(" ", strip=True))

            out.append(Link(url=url, anchor=anchor))

        return out
