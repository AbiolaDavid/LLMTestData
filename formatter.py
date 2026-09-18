"""
Response formatting helpers for SOC 101 RAG answers.
Detects and pretty-prints pipe-separated tables
Formats source metadata in a fixed structure:
Source
Module
Page
Course Title
Cleans common PDF/extraction artifacts
"""
from __future__ import annotations
import re
from typing import Any

# ---------------------------------------------------------------------------
# Low-level cleaners
# ---------------------------------------------------------------------------
_MULTISPACE = re.compile(r"[ \t]{2,}")
_MULTINEWLINE = re.compile(r"\n{3,}")
_PAGE_NOISE = re.compile(r"^=+\sPage\s+\d+\s=+$", re.MULTILINE | re.IGNORECASE)

def _clean_text(text: str) -> str:
    """Remove common extraction noise and normalize whitespace."""
    if not text:
        return ""
    text = _PAGE_NOISE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTISPACE.sub(" ", text)
    text = _MULTINEWLINE.sub("\n\n", text)
    return text.strip()

# ---------------------------------------------------------------------------
# Table detection & pretty-printing
# ---------------------------------------------------------------------------
def _looks_like_table_row(line: str) -> bool:
    """A line that contains at least two '|' separators is treated as a table row."""
    return line.count("|") >= 2

def _split_row(line: str) -> list[str]:
    cells = [c.strip() for c in line.split("|")]
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return cells

def _is_separator_row(cells: list[str]) -> bool:
    if not cells:
        return False
    return all(re.fullmatch(r":?-{3,}:?", c or "") for c in cells)

def _column_widths(rows: list[list[str]]) -> list[int]:
    if not rows:
        return []
    n = max(len(r) for r in rows)
    widths = [0] * n
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    return widths

def _format_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    data_rows = [r for r in rows if not _is_separator_row(r)]
    if not data_rows:
        return ""
    ncols = max(len(r) for r in data_rows)
    data_rows = [r + [""] * (ncols - len(r)) for r in data_rows]
    widths = _column_widths(data_rows)
    
    def fmt_row(cells: list[str]) -> str:
        parts = [cells[i].ljust(widths[i]) for i in range(ncols)]
        return "| " + " | ".join(parts) + " |"
        
    sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    lines = [fmt_row(data_rows[0]), sep]
    for row in data_rows[1:]:
        lines.append(fmt_row(row))
    return "\n".join(lines)

def prettify_tables(text: str) -> str:
    """Replace consecutive pipe-separated lines with aligned tables."""
    if not text or "|" not in text:
        return text
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        if _looks_like_table_row(lines[i]):
            block: list[list[str]] = []
            while i < n and _looks_like_table_row(lines[i]):
                block.append(_split_row(lines[i]))
                i += 1
            if len(block) >= 2:
                out.append(_format_table(block))
            else:
                out.append("|".join(block[0]) if block else "")
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)

# ---------------------------------------------------------------------------
# Source / metadata formatting (required structure)
# ---------------------------------------------------------------------------
def _get(item: dict[str, Any] | Any, key: str, default: str = "") -> str:
    if isinstance(item, dict):
        val = item.get(key)
    else:
        val = getattr(item, key, None)
    if val is None:
        return default
    return str(val).strip()

def format_source(item: dict[str, Any] | Any) -> str:
    """
    Format a single source in the exact required layout:
    Source
    Module
    Page
    Course Title
    """
    heading = _get(item, "heading")          # used as "Source" (section / topic)
    module = _get(item, "module_title")      # Module
    page = _get(item, "page")                # Page
    course = _get(item, "source")            # Course Title
    
    # Fallbacks so the block never looks empty
    if not heading:
        heading = _get(item, "chunk_id") or "—"
    if not module:
        module = "—"
    if not page:
        page = "—"
    if not course:
        course = "—"
        
    # Normalise page display
    page_str = str(page)
    if page_str.isdigit():
        page_display = f"p. {page_str}"
    else:
        page_display = f"pp. {page_str}" if page_str != "—" else "—"
        
    return (
        f"Source: {heading}\n"
        f"Module: {module}\n"
        f"Page: {page_display}\n"
        f"Course Title: {course}"
    )

def format_sources(sources: list[Any], *, max_sources: int = 5) -> str:
    """Render a list of sources as numbered blocks."""
    if not sources:
        return ""
    lines = ["\n---\n**Sources**\n"]
    for i, s in enumerate(sources[:max_sources], start=1):
        lines.append(f"{i}.")
        lines.append(format_source(s))
        lines.append("")  # blank line between sources
    if len(sources) > max_sources:
        lines.append(f"… and {len(sources) - max_sources} more")
    return "\n".join(lines).rstrip()

# ---------------------------------------------------------------------------
# High-level response formatter
# ---------------------------------------------------------------------------
def format_answer(
    answer: str,
    sources: list[Any] | None = None,
    *,
    include_sources: bool = True,
    max_sources: int = 5,
) -> str:
    """
    Produce a clean, human-readable response:
    1. Clean text
    2. Pretty-print any pipe tables
    3. Append well-formatted source citations
    """
    body = _clean_text(answer or "")
    body = prettify_tables(body)
    if include_sources and sources:
        body = body.rstrip() + "\n" + format_sources(sources, max_sources=max_sources)
    return body.strip()

def format_search_hit(hit: dict[str, Any], *, index: int | None = None) -> str:
    """Pretty-print a single search result."""
    text = prettify_tables(_clean_text(hit.get("text") or ""))
    header = format_source(hit)
    prefix = f"[{index}]\n" if index is not None else ""
    return f"{prefix}{header}\n\n{text}"

def format_search_results(hits: list[dict[str, Any]]) -> str:
    """Pretty-print a list of hybrid-search hits."""
    if not hits:
        return "No matching passages found."
    blocks = [format_search_hit(h, index=i) for i, h in enumerate(hits, start=1)]
    return "\n\n" + ("\n\n" + "─" * 60 + "\n\n").join(blocks)
