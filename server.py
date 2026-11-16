"""
MCP server exposing a documentation tree to Claude Code / VS Code agents.

Every search result carries path + line span (plus the page, for a PDF), and
read_span lets the agent go verify that span against the file. The point is
that an agent using this can show where an answer came from instead of
producing it from context.

Works on any directory of Markdown, PDF and Word (.docx) files -- a docs/
folder, an engineering report tree, or an Obsidian vault.

The tool contract has two layers:

  1. Schema. Parameter bounds and descriptions live in the input schema, so the
     client sees them before calling; return types are TypedDicts, so every
     tool also publishes an output schema. Arguments that violate the input
     schema are rejected by the SDK as an `isError` result naming the field.
  2. Runtime. Anything the schema cannot express (span size, paths outside the
     root, missing files) comes back as {"ok": false, "error": <code>, ...}
     with a stable code from ERROR_CODES -- never a stack trace.

Requires the MCP Python SDK v2 (FastMCP was renamed to MCPServer in 2.0).

Run:
    export DOCS_ROOT=/path/to/docs
    pip install -r requirements.txt     # mcp, plus pypdf, python-docx, fastembed
    python server.py --prepare          # optional: download the model, build the index
    python server.py

Register with Claude Code:
    claude mcp add docs-cite --env DOCS_ROOT=/path/to/docs -- python /path/to/server.py
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Annotated, Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field
# pydantic needs typing_extensions.TypedDict to build schemas on Python < 3.12
from typing_extensions import NotRequired, TypedDict

from docs_index import (
    DEFAULT_MAX_PER_DOC,
    DocsIndex,
    InvalidSpan,
    MissingDependency,
    NoTextLayer,
    PageOutOfRange,
    PageRequired,
    PathOutsideRoot,
    UnreadableDocument,
    UnsupportedFormat,
)

DOCS_ROOT = os.environ.get("DOCS_ROOT", "./sample-docs")

# Guardrails: bounds on anything an agent can ask for.
MAX_RESULTS = 20
MAX_SNIPPET_CHARS = 1200
MAX_SPAN_LINES = 400

ERROR_CODES = {
    "empty_query": "query is blank after trimming whitespace",
    "span_too_large": f"read_span asked for more than {MAX_SPAN_LINES} lines",
    "invalid_span": "line_end is before line_start, or the span is empty after clamping",
    "path_outside_root": "path resolves outside DOCS_ROOT",
    "not_found": "no such file under DOCS_ROOT",
    "page_required": "read_span on a PDF needs a page",
    "page_out_of_range": "the PDF has no such page, or a page was given for a non-PDF file",
    "unsupported_format": "not a Markdown, PDF or Word (.docx) file",
    "missing_dependency": "the optional library is not installed: pypdf, python-docx, or "
                          "fastembed for mode=dense",
    "no_text_layer": "PDF has no extractable text (a scan); OCR is not supported",
    "unreadable": "file cannot be read: bad UTF-8, corrupt, encrypted, or permissions",
    "internal_error": "unexpected failure; detail carries the exception text",
}

mcp = MCPServer("docs-cite", version="0.4.0")

# SDK v2 runs synchronous handlers on worker threads, so tools execute
# concurrently. The index makes that safe itself (builds are serialised, and a
# search holds one immutable snapshot); this lock only guards the lazy creation,
# so two first calls cannot both build it.
_index_lock = threading.Lock()
_index: DocsIndex | None = None


def get_index() -> DocsIndex:
    global _index
    with _index_lock:
        if _index is None:
            # hybrid retrieval when fastembed is installed; DOCS_CITE_EMBED_MODEL=none
            # or a missing fastembed falls back to BM25, and every result says which
            from dense import default_embedder
            index = DocsIndex(DOCS_ROOT, embedder=default_embedder())
            index.build()
            _index = index
        return _index


def _truncate(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_SNIPPET_CHARS:
        return text, False
    return text[:MAX_SNIPPET_CHARS], True


def _error(code: str, detail: str) -> dict:
    assert code in ERROR_CODES, code
    return {"ok": False, "error": code, "detail": detail}


def _from_exception(exc: Exception) -> dict:
    if isinstance(exc, PathOutsideRoot):
        return _error("path_outside_root", str(exc))
    if isinstance(exc, InvalidSpan):
        return _error("invalid_span", str(exc))
    if isinstance(exc, FileNotFoundError):
        return _error("not_found", str(exc))
    if isinstance(exc, PageRequired):
        return _error("page_required", str(exc))
    if isinstance(exc, PageOutOfRange):
        return _error("page_out_of_range", str(exc))
    if isinstance(exc, UnsupportedFormat):
        return _error("unsupported_format", str(exc))
    if isinstance(exc, MissingDependency):
        return _error("missing_dependency", str(exc))
    if isinstance(exc, NoTextLayer):
        return _error("no_text_layer", str(exc))
    if isinstance(exc, (UnreadableDocument, UnicodeDecodeError, OSError)):
        return _error("unreadable", str(exc))
    return _error("internal_error", f"{type(exc).__name__}: {exc}")


# ---------- output types (published as each tool's outputSchema) ----------

class _Status(TypedDict):
    ok: bool
    error: NotRequired[str]     # one of ERROR_CODES, present only when ok is false
    detail: NotRequired[str]


class SearchHit(TypedDict):
    chunk_id: str
    path: str
    heading: str
    page: int | None            # PDF page (1-based); None for Markdown and Word
    line_start: int             # within `page` for a PDF; a block number for Word
    line_end: int
    cite: str                   # "a.md:3-8", "b.pdf p.2:1-4", "c.docx ¶3-5"
    score: float                # BM25 (lexical), cosine (dense), or RRF (hybrid)
    lexical_rank: int | None    # position in the BM25 ranking, if it ranked at all
    dense_rank: int | None      # position in the vector ranking; None when lexical-only
    text: str
    truncated: bool


class SearchResult(_Status):
    query: NotRequired[str]
    mode: NotRequired[str]      # the retrieval mode actually used
    returned: NotRequired[int]
    results: NotRequired[list[SearchHit]]
    note: NotRequired[str]


class SpanResult(_Status):
    path: NotRequired[str]
    page: NotRequired[int | None]
    line_start: NotRequired[int]
    line_end: NotRequired[int]
    total_lines: NotRequired[int]   # lines in the file, or in the page for a PDF
    total_pages: NotRequired[int]   # 1 for Markdown and Word
    text: NotRequired[str]


class BacklinksResult(_Status):
    note: NotRequired[str]
    backlinks: NotRequired[list[str]]


class ReindexResult(_Status):
    notes_indexed: NotRequired[int]
    chunks: NotRequired[int]
    files_reparsed: NotRequired[int]
    files_reused: NotRequired[int]
    chunks_embedded: NotRequired[int]           # vectors computed in this build (0 = all cached)
    embedding_model: NotRequired[str | None]    # None when the index is lexical-only
    files_skipped: NotRequired[list[str]]       # unreadable, scanned, or missing a parser
    cache_written: NotRequired[bool]
    cache_error: NotRequired[str | None]        # set when the cache could not be saved


_READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)


# ---------- tools ----------

@mcp.tool(title="Search docs", annotations=_READ_ONLY)
def search_notes(
    query: Annotated[str, Field(min_length=1, description="Natural-language or keyword query.")],
    k: Annotated[int, Field(ge=1, le=MAX_RESULTS, description="Number of passages to return.")] = 5,
    max_per_doc: Annotated[int, Field(
        ge=1, le=MAX_RESULTS,
        description="Most passages any one document may contribute. The default keeps a long "
                    "document from filling every slot; raise it to see more of one document.",
    )] = DEFAULT_MAX_PER_DOC,
    mode: Annotated[Literal["hybrid", "lexical", "dense"] | None, Field(
        description="hybrid (default when an embedding model is installed) fuses keyword and "
                    "semantic rankings; lexical is keyword-only (exact names, numbers, codes); "
                    "dense is semantic-only. Omit for the server's default.",
    )] = None,
) -> SearchResult:
    """
    Search Markdown, PDF and Word docs; return passages with their exact source location.

    Every hit includes `path`, `line_start`, `line_end`, `page` (PDFs only) and
    a ready-made `cite` string. Cite those rather than paraphrasing from memory;
    call read_span with the same path, lines and page to verify a passage before
    relying on it. Snippets longer than 1200 characters come back with
    truncated=true. `mode` in the result says which retrieval was used.
    """
    query = query.strip()
    if not query:
        return _error("empty_query", "query must contain non-whitespace text")
    k = max(1, min(int(k), MAX_RESULTS))   # schema enforces this; kept for direct calls
    try:
        index = get_index()
        used = mode or index.default_mode
        hits = index.search(query, k=k, max_per_doc=max_per_doc, mode=used)
    except Exception as exc:  # structured failure, not a stack trace into the agent
        return _from_exception(exc)

    results = []
    for hit in hits:
        text, truncated = _truncate(hit.pop("text"))
        results.append({**hit, "text": text, "truncated": truncated})
    return {
        "ok": True,
        "query": query,
        "mode": used,
        "returned": len(results),
        "results": results,
        "note": "cite each hit's `cite`; call read_span (with page for PDFs) to verify",
    }


@mcp.tool(title="Read line span", annotations=_READ_ONLY)
def read_span(
    path: Annotated[str, Field(min_length=1, description='Root-relative path, e.g. "reports/phase-1.md".')],
    line_start: Annotated[int, Field(ge=1, description="1-based first line, inclusive.")],
    line_end: Annotated[int, Field(ge=1, description="1-based last line, inclusive.")],
    page: Annotated[int | None, Field(
        ge=1, description="PDF page, 1-based. Required for PDFs; omit for Markdown and Word.",
    )] = None,
) -> SpanResult:
    """
    Read an exact line range from a document, so a cited passage can be checked.

    Lines are those search_notes reported: file lines for Markdown, lines within
    `page` for a PDF, blocks (paragraphs and table rows) for Word. At most 400
    lines per call. A span running past the end is clamped to the last line;
    total_lines reports the length.
    """
    if line_end < line_start:
        return _error("invalid_span", f"line_end {line_end} is before line_start {line_start}")
    if line_end - line_start + 1 > MAX_SPAN_LINES:
        return _error(
            "span_too_large",
            f"requested {line_end - line_start + 1} lines, max {MAX_SPAN_LINES}",
        )
    try:
        return {"ok": True, **get_index().read_span(path, line_start, line_end, page)}
    except Exception as exc:
        return _from_exception(exc)


@mcp.tool(title="List backlinks", annotations=_READ_ONLY)
def list_backlinks(
    note: Annotated[str, Field(
        min_length=1,
        description='Note name ("evaluation-protocol") or root-relative path '
                    '("reports/evaluation-protocol.md"); a path disambiguates notes '
                    "that share a name in different folders.",
    )],
) -> BacklinksResult:
    """
    List documents whose [[wikilinks]] point at `note`, as root-relative paths.

    Useful for pulling the surrounding context of a concept before answering.
    """
    try:
        return {"ok": True, "note": note, "backlinks": get_index().backlinks(note)}
    except Exception as exc:
        return _from_exception(exc)


@mcp.tool(
    title="Rebuild index",
    # writes the index cache next to the corpus, but never touches a document
    annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
)
def reindex(
    force: Annotated[bool, Field(description="Reparse every file instead of only changed ones.")] = False,
) -> ReindexResult:
    """Rescan the docs tree. Only changed files are reparsed unless force is true."""
    try:
        return {"ok": True, **get_index().build(force=force)}
    except Exception as exc:
        return _from_exception(exc)


def prepare() -> int:
    """
    Download the embedding model and build the index ahead of time, so the
    agent's first search is not the one that waits for it.
    """
    import json

    from dense import default_embedder
    stats = DocsIndex(DOCS_ROOT, embedder=default_embedder()).build()   # one build, its own stats
    print(json.dumps(stats, indent=2), file=sys.stderr)
    return 1 if stats["notes_indexed"] == 0 else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--prepare"]:
        raise SystemExit(prepare())
    mcp.run()
