"""
The invariant that makes a citation real: for every chunk, the recorded line
span must reproduce the chunk text exactly when sliced out of the source file.
If this test fails, the server is inventing provenance.

Run: python test_index.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from unittest import mock

import docs_index
from docs_index import DocsIndex, chunk_markdown, tokenize

FIXTURES = {
    "research/esci.md": (
        "# ESCI Ranking\n"
        "Baseline work on Amazon product search.\n"
        "\n"
        "## Task 1 benchmark\n"
        "BM25 scores 0.8198 full-list NDCG against a 0.7503 random floor.\n"
        "Two-Tower reaches 0.8245 and LambdaMART 0.8429.\n"
        "\n"
        "## Multilingual\n"
        "The encoder is English-only, so JP retrieval degrades zero-shot.\n"
        "See [[two-tower]] for the training setup.\n"
    ),
    "research/two-tower.md": (
        "# Two-Tower\n"
        "Trained on US pairs only with msmarco-distilbert-base-v3.\n"
        "Recall@100 moves from 0.4598 to 0.4649 after the training fix.\n"
    ),
    "notes/中文笔记.md": (
        "# 检索评测\n"
        "混合检索用 RRF 融合，召回率明显好于单一通道。\n"
        "中文分词缺失会让 BM25 完全失效。\n"
    ),
}

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        failures.append(name)


def build_corpus(root: Path) -> None:
    for rel, body in FIXTURES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")


def corpus(files: dict[str, str]) -> Path:
    root = Path(tempfile.mkdtemp(prefix="docs_sys_"))
    _scratch.append(root)
    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return root


_scratch: list[Path] = []


def system_checks() -> None:
    """Each check here reproduced a real defect before the fix that made it pass."""
    try:
        print("\nfreshness: a citation never points into an older version of the file")
        root = corpus({"a.md": "# A\nalpha bravo charlie delta echo foxtrot\n"})
        idx = DocsIndex(root, refresh_interval=3600)     # sweep disabled: test the hit check
        idx.build()
        (root / "a.md").write_text(
            "# A\nnew intro line inserted here\nmore filler text\n"
            "alpha bravo charlie delta echo foxtrot\n", encoding="utf-8")
        hit = idx.search("foxtrot")[0]
        span = idx.read_span(hit["path"], hit["line_start"], hit["line_end"])
        check("edited file is re-read before it is cited", span["text"] == hit["text"],
              f"-> cited {hit['line_start']}-{hit['line_end']}")

        idx = DocsIndex(root, refresh_interval=0)
        idx.build()
        (root / "b.md").write_text("# B\nnovel golf hotel india term\n", encoding="utf-8")
        check("new file is found without a manual reindex",
              [h["path"] for h in idx.search("hotel india")] == ["b.md"])
        (root / "b.md").unlink()
        check("deleted file stops being returned", idx.search("hotel india") == [])

        print("\ncache versioning")
        root = corpus({"a.md": "# A\nsome text here for the chunk\n"})
        DocsIndex(root).build()
        cache = root / ".docs_cite_index.json"
        raw = json.loads(cache.read_text())
        raw["key"][0] = -1                                 # written by "other" code
        for rec in raw["files"].values():
            rec["chunks"][0]["heading"] = "STALE-FROM-OLD-CHUNKER"
        cache.write_text(json.dumps(raw))
        idx = DocsIndex(root)
        stats = idx.build()
        check("cache from another chunker version is discarded",
              idx.chunks[0].heading == "A" and stats["files_reparsed"] == 1,
              f"-> {idx.chunks[0].heading!r}")
        legacy = {"version": 1, "stamps": {}, "links": {}, "chunks": []}
        cache.write_text(json.dumps(legacy))
        check("pre-versioning cache layout is discarded, not crashed on",
              DocsIndex(root).build()["files_reparsed"] == 1)

        print("\nconcurrency: search during reindex sees one consistent snapshot")
        body = "\n".join(f"line {i} kilo lima mike november oscar" for i in range(200))
        root = corpus({f"d{i}.md": f"# D{i}\n{body}\n" for i in range(40)})
        idx = DocsIndex(root, refresh_interval=3600)
        idx.build()
        errors: list[str] = []
        stop = threading.Event()

        def searcher() -> None:
            while not stop.is_set():
                try:
                    idx.search("kilo november", k=5)
                except Exception as exc:  # noqa: BLE001 -- any failure is the finding
                    errors.append(type(exc).__name__)

        threads = [threading.Thread(target=searcher) for _ in range(4)]
        for t in threads:
            t.start()
        for _ in range(5):
            idx.build(force=True)
        stop.set()
        for t in threads:
            t.join()
        check("no errors from searches racing 5 forced rebuilds", not errors,
              f"-> {len(errors)} errors {set(errors)}")

        print("\ncorpus boundaries")
        outside = corpus({"secret.md": "# S\nzulu yankee xray whiskey\n"})
        root = corpus({"a.md": "# A\nplain content line here\n"})
        (root / "link.md").symlink_to(outside / "secret.md")
        (root / "inner.md").symlink_to(root / "a.md")
        idx = DocsIndex(root)
        idx.build()
        check("symlink escaping the root is not indexed", idx.search("zulu") == [])
        check("symlink inside the root is still indexed",
              any(c.path == "inner.md" for c in idx.chunks))

        print("\nbacklinks are keyed by path, not by file name")
        root = corpus({
            "x/notes.md": "# x\nsee [[target]] here\n",
            "y/notes.md": "# y\nnothing linked from here\n",
            "z/other.md": "# z\nsee [[y/notes]] and [[target.md]]\n",
            "target.md": "# t\nbody text\n",
        })
        idx = DocsIndex(root)
        idx.build()
        check("same-named notes in two folders keep their own links",
              idx.backlinks("target") == ["x/notes.md", "z/other.md"],
              f"-> {idx.backlinks('target')}")
        check("a folder-qualified link resolves to that folder only",
              idx.backlinks("y/notes.md") == ["z/other.md"]
              and idx.backlinks("x/notes") == [],
              f"-> y:{idx.backlinks('y/notes.md')} x:{idx.backlinks('x/notes')}")

        print("\nencodings")
        root = corpus({})
        (root / "bom.md").write_text("﻿# Title\nbody text line here\n## Sub\nmore body text\n",
                                     encoding="utf-8")
        (root / "crlf.md").write_bytes(b"# Win\r\nwindows line endings here\r\n")
        (root / "latin1.md").write_bytes("# Caf\xe9\nnot utf-8 at all here\n".encode("latin-1"))
        idx = DocsIndex(root)
        stats = idx.build()
        heads = [c.heading for c in idx.chunks if c.path == "bom.md"]
        check("byte-order mark does not hide the first heading",
              heads == ["Title", "Title > Sub"], f"-> {heads}")
        crlf = idx.search("windows endings")[0]
        check("CRLF spans round-trip through read_span",
              idx.read_span("crlf.md", crlf["line_start"], crlf["line_end"])["text"] == crlf["text"])
        check("unreadable file is reported, not silently dropped",
              stats["files_skipped"] == ["latin1.md"], f"-> {stats['files_skipped']}")

        print("\na read-only corpus is still searchable")
        root = corpus({"a.md": "# A\nquebec romeo sierra tango\n"})
        os.chmod(root, 0o555)
        try:
            idx = DocsIndex(root)
            stats = idx.build()
            check("build succeeds without a writable cache",
                  stats["notes_indexed"] == 1 and stats["cache_written"] is False
                  and stats["cache_error"], f"-> {stats}")
            check("search works on it", bool(idx.search("romeo")))
        finally:
            os.chmod(root, 0o755)

        print("\nlong sections split at a paragraph break")
        para = lambda tag, n: [f"{tag} sentence {i} of the paragraph" for i in range(n)]  # noqa: E731
        lines = ["# Long"] + para("first", 30) + [""] + para("second", 30)
        spans = [(c.line_start, c.line_end) for c in chunk_markdown("long.md", "\n".join(lines))]
        check("cut lands on the blank line, not mid-paragraph",
              spans == [(1, 32), (33, 62)], f"-> {spans}")
        check("no chunk exceeds MAX_CHUNK_LINES",
              all(e - s + 1 <= docs_index.MAX_CHUNK_LINES for s, e in spans))
        dense = "\n".join(["# Dense"] + para("x", 90))
        dense_spans = [(c.line_start, c.line_end) for c in chunk_markdown("dense.md", dense)]
        check("without a blank line it falls back to the hard cut",
              dense_spans == [(1, 40), (41, 80), (81, 91)], f"-> {dense_spans}")

        print("\nresult crowding: chunks per document are capped")
        long_doc = "\n\n".join(f"## Part {i}\npapa quebec detail {i}" for i in range(12))
        root = corpus({"long.md": f"# Long\n{long_doc}\n",
                       "short.md": "# Short\npapa quebec, the authoritative answer\n"})
        idx = DocsIndex(root)
        idx.build()
        uncapped = [h["path"] for h in idx.search("papa quebec", k=5)]
        capped = [h["path"] for h in idx.search("papa quebec", k=5, max_per_doc=2)]
        check("uncapped, the long document can crowd the list",
              uncapped.count("long.md") >= 4, f"-> {uncapped}")
        check("capped at 2, the long document holds at most two slots",
              capped.count("long.md") == 2 and "short.md" in capped, f"-> {capped}")

        print("\nincremental rebuild does no work when nothing changed")
        root = corpus({f"d{i}.md": f"# D{i}\n{body}\n" for i in range(40)})
        DocsIndex(root).build()
        cache = root / ".docs_cite_index.json"
        before = cache.stat().st_mtime_ns
        fresh = DocsIndex(root)                            # new process: starts from the cache
        with mock.patch("docs_index.tokenize", side_effect=AssertionError("re-tokenized")):
            stats = fresh.build()
        check("no file is re-tokenized from a warm cache", stats["files_reparsed"] == 0)
        check("an unchanged index does not rewrite its cache",
              cache.stat().st_mtime_ns == before and stats["cache_written"] is False)
    finally:
        for path in _scratch:
            os.chmod(path, 0o755)
            shutil.rmtree(path, ignore_errors=True)


def format_checks() -> None:
    """PDF and Word: the same round-trip invariant, plus the ways those files go wrong."""
    try:
        import docx  # noqa: F401
        import pypdf  # noqa: F401
    except ImportError:
        print("\nPDF / Word  SKIPPED (pip install pypdf python-docx to run)")
        return
    from docs_index import PageOutOfRange, PageRequired
    from sample_files import write_docx, write_pdf

    try:
        print("\nPDF: citations are page + line, and round-trip")
        root = corpus({"notes.md": "# Notes\nplain markdown sitting next to binaries\n"})
        long_page = [f"row {i} of the appendix table" for i in range(55)]
        write_pdf(root / "report.pdf", [
            ["Quarterly Report", "Revenue grew in the alpha region."],
            ["Appendix", *long_page],
            ["Closing remarks about the bravo initiative."],
        ])
        idx = DocsIndex(root)
        stats = idx.build()
        pdf_chunks = [c for c in idx.chunks if c.path == "report.pdf"]
        check("every PDF chunk names its page", all(c.page for c in pdf_chunks))
        check("a long page is split, and no chunk crosses a page",
              [c.page for c in pdf_chunks] == [1, 2, 2, 3], f"-> {[c.cite for c in pdf_chunks]}")
        bad = [c.cite for c in pdf_chunks
               if idx.read_span(c.path, c.line_start, c.line_end, c.page)["text"] != c.text]
        check("every PDF span reproduces its chunk text", not bad, f"-> {bad}")
        hit = idx.search("bravo initiative")[0]
        check("search cites page and lines", hit["cite"] == "report.pdf p.3:1-1", f"-> {hit['cite']}")
        for call, exc_type, label in (
            (lambda: idx.read_span("report.pdf", 1, 2), PageRequired, "PDF without page"),
            (lambda: idx.read_span("report.pdf", 1, 2, page=9), PageOutOfRange, "page past the end"),
            (lambda: idx.read_span("notes.md", 1, 2, page=1), PageOutOfRange, "page on Markdown"),
        ):
            try:
                call()
                check(f"rejects {label}", False, "-> no error")
            except exc_type:
                check(f"rejects {label}", True)
        check("markdown beside binaries is unaffected", stats["notes_indexed"] == 2)

        write_pdf(root / "report.pdf", [["Quarterly Report", "Revenue now grew in the charlie region."]])
        hit = idx.search("charlie region")[0]
        span = idx.read_span(hit["path"], hit["line_start"], hit["line_end"], hit["page"])
        check("an edited PDF is re-extracted before it is cited", span["text"] == hit["text"])

        print("\nPDF: files that cannot be cited are reported, not crashed on")
        root = corpus({"ok.md": "# Fine\nthis file indexes normally\n"})
        write_pdf(root / "scan.pdf", [[], []])                 # pages with no text layer
        (root / "broken.pdf").write_bytes(b"%PDF-1.4 this is not really a pdf")
        idx = DocsIndex(root)
        stats = idx.build()
        check("scanned and corrupt PDFs land in files_skipped",
              stats["files_skipped"] == ["broken.pdf", "scan.pdf"] and stats["notes_indexed"] == 1,
              f"-> {stats['files_skipped']}")
        reasons = {rel: rec.error for rel, rec in idx._snap.files.items() if rec.error}
        check("a scan is reported as having no text layer",
              (reasons.get("scan.pdf") or "").startswith("NoTextLayer"), f"-> {reasons}")

        print("\nWord: blocks are lines, styles are headings, tables are rows")
        root = corpus({})
        write_docx(root / "runbook.docx", [
            ("h", 1, "Runbook"),
            ("p", "Intro paragraph for the service."),
            ("h", 2, "Rollback"),
            ("p", "Reinstall the delta release tag, then restart."),
            ("p", "# not a heading, just body text that starts with a hash"),
            ("h", 2, "Matrix"),
            ("table", [["Python", "Status"], ["3.10", "echo tested"]]),
        ])
        (root / "~$runbook.docx").write_bytes(b"lock file")
        idx = DocsIndex(root)
        stats = idx.build()
        docx_chunks = [c for c in idx.chunks if c.path == "runbook.docx"]
        check("Heading styles become the breadcrumb",
              [c.heading for c in docx_chunks] == ["Runbook", "Runbook > Rollback", "Runbook > Matrix"],
              f"-> {[c.heading for c in docx_chunks]}")
        bad = [c.cite for c in docx_chunks
               if idx.read_span(c.path, c.line_start, c.line_end)["text"] != c.text]
        check("every Word span reproduces its chunk text", not bad, f"-> {bad}")
        hit = idx.search("echo tested")[0]
        check("table rows are searchable and cited by block",
              hit["cite"] == "runbook.docx ¶6-8" and "| 3.10 | echo tested |" in hit["text"],
              f"-> {hit['cite']} {hit['text']!r}")
        check("body text starting with # is escaped, not promoted to a heading",
              "\\# not a heading" in docx_chunks[1].text, f"-> {docx_chunks[1].text!r}")
        check("Word lock files (~$name.docx) are not documents",
              stats["notes_indexed"] == 1 and not stats["files_skipped"], f"-> {stats}")

        print("\nwithout the optional libraries, Markdown still works")
        root = corpus({"a.md": "# A\nfoxtrot golf markdown content\n"})
        write_pdf(root / "b.pdf", [["foxtrot golf in a pdf"]])
        with mock.patch.dict(sys.modules, {"pypdf": None}):
            idx = DocsIndex(root)
            stats = idx.build()
        check("PDF is skipped with the reason, Markdown is indexed",
              stats["files_skipped"] == ["b.pdf"] and stats["notes_indexed"] == 1
              and "pip install pypdf" in (idx._snap.files["b.pdf"].error or ""),
              f"-> {stats}")
        idx = DocsIndex(root)                          # same cache, pypdf now importable
        stats = idx.build()
        check("installing pypdf later indexes the PDF without touching the file",
              stats["files_skipped"] == [] and stats["notes_indexed"] == 2, f"-> {stats}")
        with mock.patch.dict(sys.modules, {"pypdf": None}):
            idx = DocsIndex(root)                      # cache built *with* pypdf
            stats = idx.build()
            cited = [h["cite"] for h in idx.search("foxtrot golf", k=5)]
        check("a cache built with pypdf never serves PDF citations it cannot verify",
              stats["files_skipped"] == ["b.pdf"] and not any("b.pdf" in c for c in cited),
              f"-> skipped {stats['files_skipped']} cited {cited}")
    finally:
        for path in _scratch:
            shutil.rmtree(path, ignore_errors=True)


def dense_checks() -> None:
    """
    Hybrid retrieval plumbing, with a deterministic hashed stand-in for the
    model: caching, incremental re-embedding, invalidation, fusion, the cap.
    """
    try:
        import numpy as np
        from dense import HashEmbedder
    except ImportError:
        print("\ndense / hybrid  SKIPPED (pip install fastembed to run)")
        return
    from docs_index import MissingDependency

    class Counting(HashEmbedder):
        """Counts passages embedded, so a test can prove the cache was used."""
        def __init__(self, dim: int = 64):
            super().__init__(dim)
            self.passages = 0

        def embed(self, texts, kind):
            if kind == "passage":
                self.passages += len(texts)
            return super().embed(texts, kind)

    try:
        print("\nhybrid retrieval: vectors are cached, incremental, and versioned")
        files = {f"doc{i}.md": f"# Doc {i}\nsierra tango topic number {i} here\n" for i in range(6)}
        files["long.md"] = "# Long\n" + "\n\n".join(
            f"## Part {i}\nuniform victor detail {i}" for i in range(10))
        root = corpus(files)
        emb = Counting()
        idx = DocsIndex(root, embedder=emb, refresh_interval=3600)
        stats = idx.build()
        n_chunks = stats["chunks"]
        check("first build embeds every chunk once",
              stats["chunks_embedded"] == n_chunks == emb.passages, f"-> {stats}")
        check("default mode is hybrid when a model is present", idx.default_mode == "hybrid")

        hits = idx.search("sierra tango topic", k=5)
        check("hybrid hits report both ranks",
              all("lexical_rank" in h and "dense_rank" in h for h in hits) and hits[0]["dense_rank"])
        capped = [h["path"] for h in idx.search("uniform victor detail", k=6, max_per_doc=2)]
        check("max_per_doc holds after fusion", capped.count("long.md") <= 2, f"-> {capped}")

        fresh = Counting()
        again = DocsIndex(root, embedder=fresh, refresh_interval=3600)
        stats = again.build()
        check("a new process loads vectors from the cache instead of re-embedding",
              stats["chunks_embedded"] == 0 and fresh.passages == 0, f"-> {stats}")
        check("cached vectors are bit-identical to freshly computed ones",
              np.array_equal(again._snap.matrix, idx._snap.matrix))
        check("so rankings are identical across restarts",
              [h["chunk_id"] for h in again.search("sierra tango topic", k=5)]
              == [h["chunk_id"] for h in hits])

        (root / "doc0.md").write_text("# Doc 0\nsierra tango rewritten for the test\n", encoding="utf-8")
        stats = again.build()
        check("an edited file re-embeds only its own chunks",
              stats["chunks_embedded"] == 1, f"-> {stats['chunks_embedded']}")

        other = Counting(dim=32)                           # "a different model"
        stats = DocsIndex(root, embedder=other, refresh_interval=3600).build()
        check("switching models discards the cached vectors",
              stats["chunks_embedded"] == n_chunks, f"-> {stats['chunks_embedded']}")

        lexical_only = DocsIndex(root, refresh_interval=3600)
        lexical_only.build()
        check("without a model the default mode is lexical", lexical_only.default_mode == "lexical")
        try:
            lexical_only.search("sierra", mode="dense")
            check("mode=dense without a model is refused", False, "-> no error")
        except MissingDependency:
            check("mode=dense without a model is refused", True)
        plain = lexical_only.search("sierra tango", k=3)
        check("lexical results carry no dense rank", all(h["dense_rank"] is None for h in plain))
    finally:
        for path in _scratch:
            shutil.rmtree(path, ignore_errors=True)


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="docs_"))
    try:
        build_corpus(root)
        index = DocsIndex(root)
        stats = index.build(force=True)
        print(f"\nindexed {stats['notes_indexed']} notes -> {stats['chunks']} chunks\n")

        print("line-span round trip")
        for chunk in index.chunks:
            lines = (root / chunk.path).read_text(encoding="utf-8").split("\n")
            sliced = "\n".join(lines[chunk.line_start - 1 : chunk.line_end])
            if sliced != chunk.text:
                check(f"{chunk.chunk_id}", False, "span does not reproduce text")
                break
        else:
            check("every chunk span reproduces its source text", True)

        print("\nline numbers are 1-based and in range")
        ok = all(
            1 <= c.line_start <= c.line_end <= len((root / c.path).read_text().split("\n"))
            for c in index.chunks
        )
        check("spans within file bounds", ok)

        print("\nretrieval")
        hits = index.search("LambdaMART NDCG", k=3)
        check("finds the benchmark section", bool(hits) and "esci" in hits[0]["path"])
        check("hit carries a line span", bool(hits) and hits[0]["line_end"] >= hits[0]["line_start"])

        cjk = index.search("中文分词", k=3)
        check("CJK bigrams retrieve Chinese notes", bool(cjk), "-> empty result")

        print("\nread_span verifies a citation")
        if hits:
            top = hits[0]
            span = index.read_span(top["path"], top["line_start"], top["line_end"])
            check("read_span matches the search snippet", span["text"] == top["text"])

        print("\nbacklinks")
        check("esci links to two-tower",
              index.backlinks("two-tower") == ["research/esci.md"],
              f"-> {index.backlinks('two-tower')}")

        print("\npath guard")
        for bad in ("../../etc/passwd", "/etc/passwd"):
            try:
                index.resolve(bad)
                check(f"rejects {bad}", False, "-> resolved")
            except (ValueError, FileNotFoundError):
                check(f"rejects {bad}", True)

        print("\nincremental rebuild")
        again = index.build(force=False)
        check("reuses cached files on a clean rebuild", again["files_reparsed"] == 0,
              f"-> reparsed {again['files_reparsed']}")

        print("\ntokenizer")
        check("splits latin words", tokenize("Recall@100 NDCG") == ["recall", "100", "ndcg"])
        check("emits CJK bigrams", "检索" in tokenize("检索评测"))

        print("\nempty file does not crash the chunker")
        check("empty source -> no chunks", chunk_markdown("empty.md", "") == [])

        print("\ncode fences")
        fenced = (
            "# Setup\n"
            "Install the package and configure the shell.\n"
            "```bash\n"
            "# install dependencies first\n"
            "pip install mcp\n"
            "```\n"
            "~~~~python\n"
            "# a comment, not a heading\n"
            "~~~\n"                       # too short: does not close a ~~~~ fence
            "## still inside the fence\n"
            "~~~~\n"
            "## Usage\n"
            "Run the server with the docs root set.\n"
        )
        crumbs = [c.heading for c in chunk_markdown("fenced.md", fenced)]
        check("# inside a fence is not a heading",
              not any("install dependencies" in h or "a comment" in h for h in crumbs),
              f"-> {crumbs}")
        check("shorter closing marker does not end a fence",
              not any("still inside" in h for h in crumbs), f"-> {crumbs}")
        check("headings after the fence still split",
              crumbs == ["Setup", "Setup > Usage"], f"-> {crumbs}")
        unclosed = chunk_markdown("open.md", "# Top\nsome intro text here\n```\n# not a heading\n")
        check("unclosed fence runs to end of file",
              [c.heading for c in unclosed] == ["Top"], f"-> {[c.heading for c in unclosed]}")

        system_checks()
        format_checks()
        dense_checks()

        print()
        if failures:
            print(f"{len(failures)} FAILED: {', '.join(failures)}")
            return 1
        print("all checks passed")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
