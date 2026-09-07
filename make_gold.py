"""
Interactive gold-set builder.

The point of the workflow it enforces: you label a query BEFORE seeing what the
retriever returned. If you find the answer document by searching with the system
under test, you can only ever label what it already finds, Recall@K climbs toward
1.0, and the misses you were trying to measure are defined out of existence.

So: write the query, find the answer yourself (the corpus map and /grep are here
for that -- both are exhaustive, not ranked), label it, and only then does the
tool show you where BM25 actually put it.

Usage:
    python make_gold.py --docs ./sample-docs --out gold.jsonl

Commands inside the prompt:
    /list            corpus map -- every document with its headings
    /grep <text>     exhaustive substring search (not BM25, no ranking)
    /show <n>        print the head of document n
    /stats           how many queries so far, and reveal-rank distribution
    /undo            drop the last labelled query
    /quit            save and exit
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from docs_index import DocsIndex

REVEAL_DEPTH = 10


def outline(text: str, limit: int = 4) -> list[str]:
    heads = [
        m.group(2)
        for m in (re.match(r"^(#{1,3})\s+(.*\S)\s*$", line) for line in text.split("\n"))
        if m
    ]
    return heads[:limit]


class Corpus:
    def __init__(self, root: Path):
        self.root = root
        self.paths: list[str] = []
        self.text: dict[str, str] = {}
        for path in sorted(root.rglob("*.md")):
            if any(part.startswith(".") for part in path.relative_to(root).parts):
                continue
            rel = path.relative_to(root).as_posix()
            try:
                self.text[rel] = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            self.paths.append(rel)

    def show_map(self) -> None:
        print()
        for i, rel in enumerate(self.paths, start=1):
            heads = outline(self.text[rel])
            lines = len(self.text[rel].split("\n"))
            print(f"  [{i:>3}] {rel}  ({lines} lines)")
            if heads:
                print(f"        {' | '.join(heads)}")
        print()

    def grep(self, needle: str) -> None:
        needle_l = needle.lower()
        hits = 0
        print()
        for i, rel in enumerate(self.paths, start=1):
            for lineno, line in enumerate(self.text[rel].split("\n"), start=1):
                if needle_l in line.lower():
                    print(f"  [{i:>3}] {rel}:{lineno}  {line.strip()[:100]}")
                    hits += 1
                    if hits >= 60:
                        print("  ... (truncated at 60 matches)")
                        return
        print(f"  {hits} matches\n" if hits else "  no matches\n")

    def show(self, index: int, lines: int = 30) -> None:
        rel = self.paths[index - 1]
        body = self.text[rel].split("\n")[:lines]
        print(f"\n  --- {rel} (first {len(body)} lines) ---")
        for lineno, line in enumerate(body, start=1):
            print(f"  {lineno:>4} | {line}")
        print()


def normalise(raw: str) -> str:
    """
    Accept what a Chinese IME actually produces, plus an accidentally pasted
    prompt. Rejecting a fullwidth comma only costs the annotator a retype.
    """
    cleaned = raw.strip()
    for prefix in ("relevant>", "query>"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    return cleaned.replace("\uff0c", ",").replace("\u3001", ",").replace("\u3000", " ")


def load_existing(out: Path) -> list[dict]:
    if not out.exists():
        return []
    return [
        json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save(out: Path, rows: list[dict]) -> None:
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--out", default="gold.jsonl")
    ap.add_argument("--no-reveal", action="store_true",
                    help="skip the post-label BM25 rank feedback")
    args = ap.parse_args()

    root = Path(args.docs).expanduser().resolve()
    corpus = Corpus(root)
    if not corpus.paths:
        print(f"no .md files under {root}")
        return 1

    out = Path(args.out)
    rows = load_existing(out)
    seen = {r["query"] for r in rows}

    index = DocsIndex(root)
    index.build()

    print(f"\ncorpus: {len(corpus.paths)} documents under {root}")
    print(f"gold:   {len(rows)} queries already in {out}")
    print("\nWrite the query first. Find the answer with /list or /grep -- not with")
    print("the retriever. Type /list to see the corpus map, /quit when done.\n")

    reveal_ranks: list[int | None] = []

    while True:
        try:
            query = normalise(input("query> "))
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query == "/quit":
            break
        if query == "/list":
            corpus.show_map()
            continue
        if query.startswith("/grep "):
            corpus.grep(query[6:].strip())
            continue
        if query.startswith("/show "):
            try:
                corpus.show(int(query[6:].strip()))
            except (ValueError, IndexError):
                print("  usage: /show <document number>")
            continue
        if query == "/stats":
            found = [r for r in reveal_ranks if r is not None]
            print(f"\n  {len(rows)} queries labelled this session and before")
            if reveal_ranks:
                print(f"  of {len(reveal_ranks)} revealed: {len(found)} found in top "
                      f"{REVEAL_DEPTH}, {len(reveal_ranks) - len(found)} missed")
            print()
            continue
        if query == "/undo":
            if rows:
                dropped = rows.pop()
                seen.discard(dropped["query"])
                save(out, rows)
                print(f"  dropped: {dropped['query']}\n")
            continue
        if query.startswith("/"):
            print("  unknown command\n")
            continue
        if query in seen:
            print("  already labelled; skipping\n")
            continue

        print("  which documents answer this? numbers from /list, comma separated")
        print("  (blank = no document answers it, which is itself worth recording)")
        picked = None
        while True:
            try:
                picked = normalise(input("  relevant> "))
            except (EOFError, KeyboardInterrupt):
                print()
                picked = None
                break
            # inspection commands work here too, so you can check before labelling
            if picked == "/list":
                corpus.show_map()
                continue
            if picked.startswith("/grep "):
                corpus.grep(picked[6:].strip())
                continue
            if picked.startswith("/show "):
                try:
                    corpus.show(int(picked[6:].strip()))
                except (ValueError, IndexError):
                    print("  usage: /show <document number>")
                continue
            if picked == "/skip":
                picked = None
                break
            break
        if picked is None:
            print("  skipped\n")
            continue

        relevant: list[str] = []
        bad = False
        for token in (t.strip() for t in picked.split(",") if t.strip()):
            try:
                relevant.append(corpus.paths[int(token) - 1])
            except (ValueError, IndexError):
                print(f"  '{token}' is not a document number; nothing saved\n")
                bad = True
                break
        if bad:
            continue

        relevant = list(dict.fromkeys(relevant))   # a repeated number is a typo
        rows.append({"query": query, "relevant": relevant})
        seen.add(query)
        save(out, rows)

        if args.no_reveal or not relevant:
            print(f"  saved ({len(rows)} total)\n")
            continue

        ranked: list[str] = []
        for hit in index.search(query, k=REVEAL_DEPTH * 3):
            if hit["path"] not in ranked:
                ranked.append(hit["path"])
        best = next(
            (i for i, path in enumerate(ranked[:REVEAL_DEPTH], start=1) if path in relevant),
            None,
        )
        reveal_ranks.append(best)
        if best == 1:
            verdict = "rank 1"
        elif best:
            verdict = f"rank {best}"
        else:
            verdict = f"NOT in top {REVEAL_DEPTH}  <-- a real miss, keep it"
        print(f"  saved ({len(rows)} total).  BM25: {verdict}\n")

    save(out, rows)
    print(f"\nwrote {len(rows)} queries to {out}")
    empty = sum(1 for r in rows if not r["relevant"])
    if empty:
        print(f"{empty} of them have no relevant document -- eval_search.py skips those,")
        print("but they are worth keeping as a record of coverage gaps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())