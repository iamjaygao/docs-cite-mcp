"""
MCP server exposing a markdown documentation tree to Claude Code / VS Code agents.

Every search result carries path + line span, and read_span lets the agent go
verify that span against the file. The point is that an agent using this can
show where an answer came from instead of producing it from context.

Works on any directory of .md files -- a docs/ folder, an engineering report
tree, or an Obsidian vault.

Requires the MCP Python SDK v2 (FastMCP was renamed to MCPServer in 2.0).

Run:
    export DOCS_ROOT=/path/to/docs
    pip install "mcp>=2,<3"
    python server.py

Register with Claude Code:
    claude mcp add docs-cite -- python /path/to/server.py
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from docs_index import DocsIndex

DOCS_ROOT = os.environ.get("DOCS_ROOT", "./sample-docs")

# Guardrails: bounds on anything an agent can ask for.
MAX_RESULTS = 20
MAX_SNIPPET_CHARS = 1200
MAX_SPAN_LINES = 400

mcp = MCPServer("docs-cite", version="0.1.0")

# SDK v2 runs synchronous handlers on worker threads, so tools can now execute
# concurrently. The index is shared mutable state: without this lock two racing
# calls can both build it, and a reindex can swap chunks out from under a search.
_index_lock = threading.Lock()
_index: DocsIndex | None = None


def get_index() -> DocsIndex:
    global _index
    with _index_lock:
        if _index is None:
            index = DocsIndex(DOCS_ROOT)
            index.build()
            _index = index
        return _index


def _truncate(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_SNIPPET_CHARS:
        return text, False
    return text[:MAX_SNIPPET_CHARS], True


@mcp.tool()
def search_notes(query: str, k: int = 5) -> dict:
    """
    Search the docs and return passages with their exact source location.

    Every hit includes `path`, `line_start` and `line_end`. Cite those rather
    than paraphrasing from memory; call read_span to verify before relying on a
    passage.

    Args:
        query: natural-language or keyword query.
        k: number of passages to return (1-20).
    """
    query = (query or "").strip()
    if not query:
        return {"ok": False, "error": "empty_query", "detail": "query must be non-empty"}
    k = max(1, min(int(k), MAX_RESULTS))
    try:
        hits = get_index().search(query, k=k)
    except Exception as exc:  # structured failure, not a stack trace into the agent
        return {"ok": False, "error": type(exc).__name__, "detail": str(exc)}

    results = []
    for hit in hits:
        text, truncated = _truncate(hit.pop("text"))
        results.append({**hit, "text": text, "truncated": truncated})
    return {
        "ok": True,
        "query": query,
        "returned": len(results),
        "results": results,
        "note": "cite path + line_start-line_end; call read_span to verify",
    }


@mcp.tool()
def read_span(path: str, line_start: int, line_end: int) -> dict:
    """
    Read an exact line range from a note, so a cited passage can be checked.

    Args:
        path: root-relative path, e.g. "reports/phase-1.md".
        line_start: 1-based first line, inclusive.
        line_end: 1-based last line, inclusive.
    """
    if line_end - line_start + 1 > MAX_SPAN_LINES:
        return {
            "ok": False,
            "error": "span_too_large",
            "detail": f"requested {line_end - line_start + 1} lines, max {MAX_SPAN_LINES}",
        }
    try:
        return {"ok": True, **get_index().read_span(path, line_start, line_end)}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "detail": str(exc)}


@mcp.tool()
def list_backlinks(note: str) -> dict:
    """
    List documents whose wikilinks point at `note` (filename without .md).

    Useful for pulling the surrounding context of a concept before answering.
    """
    stem = Path(note).stem
    try:
        return {"ok": True, "note": stem, "backlinks": get_index().backlinks(stem)}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "detail": str(exc)}


@mcp.tool()
def reindex(force: bool = False) -> dict:
    """Rescan the docs tree. Only changed files are reparsed unless force=True."""
    try:
        index = get_index()
        with _index_lock:
            return {"ok": True, **index.build(force=force)}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "detail": str(exc)}


if __name__ == "__main__":
    mcp.run()