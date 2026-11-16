"""
End-to-end checks of the MCP tool contract, through a real MCP client session.

test_index.py covers the index; this covers what an agent actually receives:
the published input/output schemas, tool annotations, and every error code.
The server runs in-process against a throwaway copy of sample-docs, so the
repo's own corpus is never written to.

Run: python test_server.py     (needs: pip install "mcp>=2,<3")
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = Path(tempfile.mkdtemp(prefix="docs_cite_srv_"))
shutil.copytree(HERE / "sample-docs", CORPUS, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(".docs_cite_index.*"))
os.environ["DOCS_ROOT"] = str(CORPUS)   # must be set before server is imported

from mcp.client import Client          # noqa: E402
from pydantic import TypeAdapter       # noqa: E402

import server                          # noqa: E402

try:
    import fastembed  # noqa: F401
    HAVE_MODEL = os.environ.get("DOCS_CITE_EMBED_MODEL", "").lower() != "none"
except ImportError:
    HAVE_MODEL = False

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        failures.append(name)


async def call(client: Client, tool: str, args: dict):
    result = await client.call_tool(tool, args)
    return result, result.structured_content or {}


async def main() -> int:
    async with Client(server.mcp) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}

        print("published contract")
        check("exposes exactly the four tools",
              set(tools) == {"search_notes", "read_span", "list_backlinks", "reindex"},
              f"-> {sorted(tools)}")
        k_schema = tools["search_notes"].input_schema["properties"]["k"]
        check("k bounds are in the input schema",
              k_schema.get("minimum") == 1 and k_schema.get("maximum") == server.MAX_RESULTS,
              f"-> {k_schema}")
        cap = tools["search_notes"].input_schema["properties"].get("max_per_doc", {})
        check("max_per_doc is published with its default",
              cap.get("default") == 2 and cap.get("minimum") == 1, f"-> {cap}")
        check("read_span line numbers are 1-based in the schema",
              all(tools["read_span"].input_schema["properties"][p].get("minimum") == 1
                  for p in ("line_start", "line_end")))
        undocumented = [
            f"{name}.{param}"
            for name, tool in tools.items()
            for param, spec in tool.input_schema["properties"].items()
            if not spec.get("description")
        ]
        check("every parameter has a description", not undocumented, f"-> {undocumented}")
        check("every tool publishes an output schema",
              all(t.output_schema for t in tools.values()))
        check("search, read_span, backlinks are marked read-only",
              all(tools[n].annotations and tools[n].annotations.read_only_hint
                  for n in ("search_notes", "read_span", "list_backlinks")))
        ann = tools["reindex"].annotations
        check("reindex is marked writing but non-destructive",
              ann is not None and ann.read_only_hint is False and ann.destructive_hint is False)

        print("\nsearch -> cite -> verify")
        result, out = await call(client, "search_notes", {"query": "BM25 baseline NDCG", "k": 3})
        check("search succeeds", not result.is_error and out.get("ok") is True, f"-> {out}")
        TypeAdapter(server.SearchResult).validate_python(out)
        check("result validates against SearchResult", True)
        hits = out.get("results", [])
        check("top hit is the baselines report",
              bool(hits) and hits[0]["path"] == "reports/ranking-baselines.md",
              f"-> {[h['path'] for h in hits]}")
        if hits:
            top = hits[0]
            _, span = await call(client, "read_span", {
                "path": top["path"], "line_start": top["line_start"], "line_end": top["line_end"],
            })
            check("read_span returns the exact cited text",
                  span.get("ok") and span.get("text") == top["text"], f"-> {span}")

        page_schema = tools["read_span"].input_schema["properties"].get("page", {})
        check("read_span publishes an optional page", "page" not in
              tools["read_span"].input_schema.get("required", []) and page_schema, f"-> {page_schema}")
        _, found = await call(client, "search_notes", {"query": "median latency fell", "k": 1})
        pdf_hit = (found.get("results") or [{}])[0]
        check("a PDF hit carries page and cite",
              pdf_hit.get("page") == 1 and pdf_hit.get("cite", "").startswith("reports/latency-review.pdf p.1:"),
              f"-> {pdf_hit.get('cite')}")
        if pdf_hit:
            _, span = await call(client, "read_span", {
                "path": pdf_hit["path"], "line_start": pdf_hit["line_start"],
                "line_end": pdf_hit["line_end"], "page": pdf_hit["page"],
            })
            check("PDF citation verifies through read_span", span.get("text") == pdf_hit["text"],
                  f"-> {span}")
        _, found = await call(client, "search_notes", {"query": "roll back a release", "k": 1})
        docx_hit = (found.get("results") or [{}])[0]
        check("a Word hit is cited by block", "deployment-runbook.docx ¶" in docx_hit.get("cite", ""),
              f"-> {docx_hit.get('cite')}")

        print("\nretrieval modes")
        mode_schema = tools["search_notes"].input_schema["properties"].get("mode", {})
        enum = {v for opt in mode_schema.get("anyOf", [mode_schema]) for v in opt.get("enum", [])}
        check("mode is published as an enum", enum == {"hybrid", "lexical", "dense"}, f"-> {mode_schema}")
        expected = "hybrid" if HAVE_MODEL else "lexical"
        _, out = await call(client, "search_notes", {"query": "BM25 baseline NDCG", "k": 3})
        check(f"default search reports mode={expected}", out.get("mode") == expected, f"-> {out.get('mode')}")
        _, out = await call(client, "search_notes", {"query": "BM25 baseline NDCG", "k": 3, "mode": "lexical"})
        check("mode=lexical is honoured and carries no dense rank",
              out.get("mode") == "lexical"
              and all(h["dense_rank"] is None for h in out.get("results", [])), f"-> {out.get('mode')}")
        if HAVE_MODEL:
            _, out = await call(client, "search_notes", {"query": "how far did median latency fall",
                                                         "k": 3, "mode": "dense"})
            check("mode=dense finds the PDF by meaning",
                  out.get("results") and out["results"][0]["path"] == "reports/latency-review.pdf",
                  f"-> {[h['cite'] for h in out.get('results', [])]}")
        result, _ = await call(client, "search_notes", {"query": "x", "mode": "semantic"})
        check("an unknown mode is rejected by the schema", result.is_error)

        _, links = await call(client, "list_backlinks", {"note": "evaluation-protocol.md"})
        check("backlinks resolve through the client",
              links.get("backlinks") == ["reports/ranking-baselines.md"], f"-> {links}")

        _, again = await call(client, "reindex", {})
        check("reindex reuses unchanged files", again.get("files_reparsed") == 0, f"-> {again}")

        print("\nschema-level rejection (isError, names the field)")
        for args, field in (({"query": "bm25", "k": 50}, "k"),
                            ({"query": "bm25", "k": 0}, "k"),
                            ({"query": ""}, "query")):
            result, _ = await call(client, "search_notes", args)
            text = " ".join(getattr(c, "text", "") for c in result.content)
            check(f"rejects {args}", result.is_error and field in text, f"-> {text[:120]}")

        print("\nruntime error codes (ok=false, stable code)")
        cases = [
            ("search_notes", {"query": "   "}, "empty_query"),
            ("read_span", {"path": "../../etc/passwd", "line_start": 1, "line_end": 2},
             "path_outside_root"),
            ("read_span", {"path": "nope.md", "line_start": 1, "line_end": 2}, "not_found"),
            ("read_span", {"path": "guides/indexing.md", "line_start": 9, "line_end": 3},
             "invalid_span"),
            ("read_span", {"path": "guides/indexing.md", "line_start": 1, "line_end": 1000},
             "span_too_large"),
            ("read_span", {"path": "reports/latency-review.pdf", "line_start": 1, "line_end": 2},
             "page_required"),
            ("read_span", {"path": "reports/latency-review.pdf", "line_start": 1, "line_end": 2,
                           "page": 7}, "page_out_of_range"),
            ("read_span", {"path": "guides/indexing.md", "line_start": 1, "line_end": 2, "page": 1},
             "page_out_of_range"),
        ]
        for tool, args, code in cases:
            result, out = await call(client, tool, args)
            check(f"{tool} -> {code}",
                  not result.is_error and out.get("ok") is False and out.get("error") == code,
                  f"-> {out}")
        codes_used = {code for *_, code in cases}
        check("every code exercised here is in ERROR_CODES",
              codes_used <= set(server.ERROR_CODES))

    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    try:
        code = asyncio.run(main())
    finally:
        shutil.rmtree(CORPUS, ignore_errors=True)
    sys.exit(code)
