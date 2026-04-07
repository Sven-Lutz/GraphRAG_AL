from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable


@dataclass(frozen=True, slots=True)
class TrapConfig:
    block_extensions: tuple[str, ...]
    block_path_patterns: tuple[str, ...]
    pagination_tokens: tuple[str, ...]
    max_pagination_depth: int = 20
    max_url_length: int = 2048
    max_query_params: int = 60
    max_repeated_param: int = 5


class TrapDetector:
    """
    Ultra-Fast Trap Detection.
    Optimizations:
    - Combined Regex (Aho-Corasick style via C-engine) instead of Python loops
    - Zero-allocation pre-checks for query lengths
    - Separation of path and query to prevent false positives
    """

    _RE_EXT = re.compile(r"\.([a-z0-9]{1,6})$", re.IGNORECASE)
    _RE_PAGE_NUM = re.compile(r"(?:/page/|page=|offset=|start=)(\d+)", re.IGNORECASE)
    _RE_QUERY_SPLIT = re.compile(r"[&;]")

    def __init__(
        self,
        block_extensions: list[str],
        block_path_patterns: list[str],
        pagination_tokens: list[str],
        max_pagination_depth: int = 20,
        *,
        max_url_length: int = 2048,
        max_query_params: int = 60,
        max_repeated_param: int = 5,
        enable_cache: bool = True,
        cache_size: int = 200_000,
    ) -> None:
        self.cfg = TrapConfig(
            block_extensions=tuple(sorted({e.lower().lstrip(".") for e in block_extensions if e})),
            block_path_patterns=tuple(p.lower() for p in block_path_patterns if p),
            pagination_tokens=tuple(sorted({t.lower() for t in pagination_tokens if t})),
            max_pagination_depth=int(max_pagination_depth),
            max_url_length=int(max_url_length),
            max_query_params=int(max_query_params),
            max_repeated_param=int(max_repeated_param),
        )

        self._block_ext = set(self.cfg.block_extensions)
        
        if self.cfg.block_path_patterns:
            escaped_paths = [re.escape(p) for p in self.cfg.block_path_patterns]
            self._path_block_re = re.compile("|".join(escaped_paths), re.IGNORECASE)
        else:
            self._path_block_re = None

        if self.cfg.pagination_tokens:
            escaped_tokens = [re.escape(t) for t in self.cfg.pagination_tokens]
            self._pagination_re = re.compile("|".join(escaped_tokens), re.IGNORECASE)
        else:
            self._pagination_re = None

        if enable_cache:
            self._should_block_impl = lru_cache(maxsize=int(cache_size))(self._should_block_impl)

    def should_block(self, url: str, depth: int) -> bool:
        return bool(self._should_block_impl(url, int(depth)))

    def _should_block_impl(self, url: str, depth: int) -> bool:
        if not url or len(url) > self.cfg.max_url_length:
            return True

        qpos = url.find("?")
        if qpos != -1:
            path_part = url[:qpos].lower()
            query_part = url[qpos + 1 :].lower()
        else:
            path_part = url.lower()
            query_part = ""

        m = self._RE_EXT.search(path_part)
        if m and m.group(1) in self._block_ext:
            return True

        if self._path_block_re and self._path_block_re.search(path_part):
            return True

        if query_part:
            param_count = query_part.count("&") + query_part.count(";") + 1
            if param_count > self.cfg.max_query_params:
                return True

            parts = self._RE_QUERY_SPLIT.split(query_part)
            counts: dict[str, int] = {}
            for part in parts:
                if not part:
                    continue
                key = part.split("=", 1)[0]
                if not key:
                    continue
                counts[key] = counts.get(key, 0) + 1
                if counts[key] > self.cfg.max_repeated_param:
                    return True

        if self._pagination_re and self._pagination_re.search(url):
            m2 = self._RE_PAGE_NUM.search(url)
            if m2:
                try:
                    if int(m2.group(1)) > self.cfg.max_pagination_depth:
                        return True
                except ValueError:
                    pass
            
            if depth > self.cfg.max_pagination_depth:
                return True

        return False