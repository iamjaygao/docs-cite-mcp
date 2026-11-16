"""
The loop an agent runs against this server, over the same stdio transport
Claude Code uses: launch server.py as a subprocess, search, cite the top hits by
path + line span, then read each span back and confirm it matches.

Run from the repo root:
    python examples/cite_and_verify.py "what is the BM25 baseline NDCG"
    python examples/cite_and_verify.py --docs /path/to/your/docs "your question"

Exits non-zero if any citation fails to verify.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from mcp import StdioServerParameters
from mcp.client import Client

REPO = Path(__file__).resolve().parent.parent


async def main(query: str, docs: Path, k: int) -> int:
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(REPO / "server.py")],
        # the server is a subprocess: DOCS_ROOT must be passed, not inherited
        env={**os.environ, "DOCS_ROOT": str(docs)},
    )
    async with Client(params) as client:
        tools = (await client.list_tools()).tools
        print(f"connected: {', '.join(t.name for t in tools)}\n")

        found = (await client.call_tool("search_notes", {"query": query, "k": k})).structured_content
        if not found.get("ok"):
            print(f"search failed: {found.get('error')} -- {found.get('detail')}")
            return 1
        if not found["results"]:
            print(f"no passages matched {query!r}")
            return 0

        verified = 0
        for i, hit in enumerate(found["results"], start=1):
            cite = f"{hit['path']}:{hit['line_start']}-{hit['line_end']}"
            print(f"[{i}] {cite}  ({hit['heading'] or 'top of file'}, score {hit['score']})")
            span = (await client.call_tool("read_span", {
                "path": hit["path"],
                "line_start": hit["line_start"],
                "line_end": hit["line_end"],
            })).structured_content
            # a truncated snippet is a prefix of the full span, not the whole of it
            ok = span.get("ok") and (
                span["text"].startswith(hit["text"]) if hit["truncated"] else span["text"] == hit["text"]
            )
            verified += bool(ok)
            first_line = next((ln for ln in span.get("text", "").splitlines() if ln.strip()), "")
            print(f"    {'verified' if ok else 'MISMATCH'}: {first_line[:90]}")

        print(f"\n{verified}/{len(found['results'])} citations verified against the source files")
        return 0 if verified == len(found["results"]) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query")
    ap.add_argument("--docs", type=Path, default=REPO / "sample-docs")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main(args.query, args.docs.resolve(), args.k)))
