"""
The invariant that makes a citation real: for every chunk, the recorded line
span must reproduce the chunk text exactly when sliced out of the source file.
If this test fails, the server is inventing provenance.

Run: python test_index.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

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
        check("esci links to two-tower", "esci" in index.backlinks("two-tower"))

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
