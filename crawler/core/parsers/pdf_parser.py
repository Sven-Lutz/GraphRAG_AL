# crawler/core/parsers/pdf_parser.py
from __future__ import annotations

import subprocess
import logging
import re
from typing import Any

from crawler.core.models import ParseResult, Segment

logger = logging.getLogger(__name__)

_RE_EXCESSIVE_SPACES = re.compile(r"[ \t]{3,}")
_RE_PARA_BREAK = re.compile(r"\n{2,}")

# Pages longer than this are split into sub-chunks at paragraph boundaries
_MAX_PAGE_CHUNK_CHARS = 3000
_MIN_CHUNK_CHARS = 200


def _clean_pdf_text(text: str) -> str:
    cleaned = _RE_EXCESSIVE_SPACES.sub(" ", text)
    return cleaned.strip()


def _chunk_page(page_text: str, page_num: int) -> list[Segment]:
    """
    Split a single PDF page into smaller segments when it exceeds _MAX_PAGE_CHUNK_CHARS.
    Splits at paragraph boundaries (double newlines) and merges tiny fragments.
    Each sub-chunk keeps the same page_ref so provenance is preserved.
    """
    if len(page_text) <= _MAX_PAGE_CHUNK_CHARS:
        return [Segment(
            order_index=page_num * 100,
            segment_type="pdf_page",
            text=page_text,
            page_ref=str(page_num + 1),
        )]

    raw_chunks = _RE_PARA_BREAK.split(page_text)
    merged: list[str] = []
    current = ""
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if current and len(current) + len(chunk) + 2 <= _MAX_PAGE_CHUNK_CHARS:
            current = current + "\n\n" + chunk
        else:
            if current:
                merged.append(current)
            current = chunk
    if current:
        merged.append(current)

    segments = []
    for sub_idx, chunk in enumerate(merged):
        if len(chunk) >= _MIN_CHUNK_CHARS:
            segments.append(Segment(
                order_index=page_num * 100 + sub_idx,
                segment_type="pdf_page",
                text=chunk,
                page_ref=str(page_num + 1),
            ))

    # Fallback: if everything got filtered return the full page text
    return segments or [Segment(
        order_index=page_num * 100,
        segment_type="pdf_page",
        text=page_text,
        page_ref=str(page_num + 1),
    )]


def parse_pdf(fetch_result: Any, url: str) -> ParseResult:
    if not fetch_result.body or len(fetch_result.body) < 100:
        logger.warning(f"PDF {url} ist leer oder zu klein zum Parsen.")
        return ParseResult(text="", segments=[], out_links=[])

    if b"%PDF-" not in fetch_result.body[:1024]:
        logger.warning(f"Datei von {url} ist kein valides PDF (Magic Bytes fehlen).")
        return ParseResult(text="", segments=[], out_links=[])

    try:
        proc = subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", "-", "-"],
            input=fetch_result.body,
            capture_output=True,
            timeout=30,
        )

        stderr_output = proc.stderr.decode("utf-8", errors="ignore")

        if proc.returncode != 0 or "pdftotext version" in stderr_output.lower():
            logger.warning(
                f"pdftotext konnte PDF {url} nicht verarbeiten. Stderr: {stderr_output[:50]}..."
            )
            return ParseResult(text="", segments=[], out_links=[])

        full_text = proc.stdout.decode("utf-8", errors="replace")

        pages = full_text.split("\x0c")  # form-feed page break
        segments = []

        for i, page_text in enumerate(pages):
            cleaned_text = _clean_pdf_text(page_text)
            if cleaned_text:
                segments.extend(_chunk_page(cleaned_text, i))

        return ParseResult(
            text=_clean_pdf_text(full_text),
            segments=segments,
            out_links=[],
        )

    except FileNotFoundError:
        logger.error(
            "'pdftotext' fehlt. Bitte Poppler installieren (z.B. 'brew install poppler')."
        )
        return ParseResult(text="", segments=[], out_links=[])
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout beim Parsen von PDF: {url}")
        return ParseResult(text="", segments=[], out_links=[])
    except Exception as e:
        logger.error(f"Unerwarteter Fehler bei PDF {url}: {e}")
        return ParseResult(text="", segments=[], out_links=[])