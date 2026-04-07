from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlsplit

import httpx

from .models import FetchResult


@dataclass(frozen=True, slots=True)
class FetcherConfig:
    user_agent: str
    global_concurrency: int = 100
    per_domain_concurrency: int = 4
    timeout_seconds: float = 20.0
    max_redirects: int = 10
    min_delay_seconds_per_domain: float = 0.0
    accept_language: str = "de-DE,de;q=0.9,en;q=0.7"
    max_keepalive_connections: int = 100
    max_connections: int = 500
    force_connection_close: bool = False


class Fetcher:

    def __init__(self, cfg: FetcherConfig) -> None:
        self.cfg = cfg

        self.global_sem = asyncio.Semaphore(int(cfg.global_concurrency))
        self.per_host_limit = int(cfg.per_domain_concurrency)
        self.timeout = httpx.Timeout(
            connect=5.0,
            read=float(cfg.timeout_seconds),
            write=10.0,
            pool=float(cfg.timeout_seconds),
        )

        self._host_sems: Dict[str, asyncio.Semaphore] = {}
        self._host_last_ts: Dict[str, float] = {}
        self._client: Optional[httpx.AsyncClient] = None

        self._limits = httpx.Limits(
            max_keepalive_connections=int(cfg.max_keepalive_connections),
            max_connections=int(cfg.max_connections),
        )

    async def __aenter__(self) -> "Fetcher":
        headers = {
            "User-Agent": self.cfg.user_agent,
            "Accept": "text/html,application/pdf,application/xhtml+xml;q=0.9,*/*;q=0.8",
            "Accept-Language": self.cfg.accept_language,
        }
        if self.cfg.force_connection_close:
            headers["Connection"] = "close"

        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=self.timeout,
            follow_redirects=True,
            limits=self._limits,
            max_redirects=int(self.cfg.max_redirects),
            verify=False,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _host(url: str) -> str:
        try:
            host = urlsplit(url).netloc.lower()
        except Exception:
            return ""
        return host.split(":", 1)[0]

    def _host_sem(self, host: str) -> asyncio.Semaphore:
        sem = self._host_sems.get(host)
        if sem is None:
            sem = asyncio.Semaphore(self.per_host_limit)
            self._host_sems[host] = sem
        return sem

    async def _polite_wait(self, host: str) -> None:
        d = float(self.cfg.min_delay_seconds_per_domain)
        if not host or d <= 0:
            return

        loop = asyncio.get_running_loop()
        now = loop.time()
        last = self._host_last_ts.get(host)
        if last is not None:
            dt = now - last
            if dt < d:
                await asyncio.sleep(d - dt)
        self._host_last_ts[host] = loop.time()

    @staticmethod
    def _content_type(headers: httpx.Headers) -> str:
        ct = headers.get("content-type", "") or ""
        return ct.split(";", 1)[0].strip().lower()

    async def fetch(self, url: str) -> FetchResult:
        if self._client is None:
            raise RuntimeError("Fetcher must be used as an async context manager.")

        host = self._host(url)

        async with self.global_sem, self._host_sem(host):
            await self._polite_wait(host)

            try:
                r = await self._client.get(url)
            except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout) as e:
                raise RuntimeError(f"timeout:{type(e).__name__}") from e
            except httpx.RequestError as e:
                raise RuntimeError(f"request_error:{type(e).__name__}:{e}") from e

            ct = self._content_type(r.headers)

            return FetchResult(
                url_final=str(r.url),
                status_code=int(r.status_code),
                content_type=ct or None,
                body=r.content or b"",
                headers={str(k): str(v) for k, v in dict(r.headers).items()},
            )
