# crawler/core/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence
from collections.abc import Set as AbcSet

@dataclass(frozen=True, slots=True)
class CrawlTask:
    municipality_id: str
    url: str
    depth: int = 0
    parent_url: Optional[str] = None
    anchor_text: Optional[str] = None
    allowed_domains: frozenset[str] = field(default_factory=frozenset)

    def with_url(self, url: str, *, depth: Optional[int] = None, parent_url: Optional[str] = None,
                 anchor_text: Optional[str] = None) -> "CrawlTask":
        """Convenience: derive a new task while keeping allowlist/muni."""
        return CrawlTask(
            municipality_id=self.municipality_id,
            url=url,
            depth=self.depth if depth is None else depth,
            parent_url=self.parent_url if parent_url is None else parent_url,
            anchor_text=self.anchor_text if anchor_text is None else anchor_text,
            allowed_domains=self.allowed_domains,
        )

@dataclass(frozen=True, slots=True)
class FetchResult:
    url_final: str
    status_code: int
    content_type: Optional[str]
    body: bytes
    headers: Mapping[str, str]

    def header(self, name: str, default: str = "") -> str:
        """
        O(1) Header-Lookup. 
        Voraussetzung: Dem FetchResult wurden bereits lowercased Keys übergeben!
        """
        return self.headers.get(name.lower(), default)

@dataclass(frozen=True, slots=True)
class Segment:
    order_index: int
    segment_type: str
    text: str
    page_ref: Optional[str] = None
    heading_context: Optional[str] = None  # e.g. "Klimaschutz > Erneuerbare Energien"

@dataclass(frozen=True, slots=True)
class ParseResult:
    text: str
    segments: Sequence[Segment]
    out_links: Sequence[tuple[str, Optional[str]]]

    meta: Mapping[str, Any] = field(default_factory=dict)

    def iter_links(self):
        return iter(self.out_links)

def normalize_allowed_domains(domains: AbcSet[str] | None) -> frozenset[str]:
    """Lowercase + drop empties; produces a stable frozenset for CrawlTask.allowed_domains."""
    if not domains:
        return frozenset()
    return frozenset(d.strip().lower() for d in domains if d and d.strip())