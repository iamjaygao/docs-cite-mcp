"""
Line-accurate chunking + BM25 retrieval over a markdown documentation tree.

Design rule: every chunk carries the exact 1-based line span it came from, and
that span must round-trip against the source file. If it doesn't, a citation is
decoration rather than a reference. tests/test_index.py enforces this.

Stdlib only, so it runs anywhere without an install step.
"""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

SKIP_DIRS = {".obsidian", ".trash", ".git", "node_modules", ".DS_Store"}
MAX_CHUNK_LINES = 40          # long sections get split further
MIN_CHUNK_CHARS = 20          # drop near-empty fragments

_WORD = re.compile(r"[a-z0-9]+")
_WIKILINK = re.compile(r"\[\[([^\]|#]+)")


def _is_cjk(ch: str) -> bool:
    return any(
        start <= ord(ch) <= end
        for start, end in (
            (0x4E00, 0x9FFF),    # CJK unified
            (0x3040, 0x30FF),    # kana
            (0xAC00, 0xD7AF),    # hangul
        )
    )


def tokenize(text: str) -> list[str]:
    """
    Latin text -> word tokens. CJK runs -> character bigrams plus unigrams.

    Whitespace tokenization silently fails on Chinese and Japanese; bigrams are
    the cheap fix that does not need a segmenter. (Same failure mode I hit on
    the ES/JP locales in the ESCI work.)
    """
    text = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    cjk_run: list[str] = []

    def flush_cjk() -> None:
        if not cjk_run:
            return
        tokens.extend(cjk_run)
        tokens.extend(
            cjk_run[i] + cjk_run[i + 1] for i in range(len(cjk_run) - 1)
        )
        cjk_run.clear()

    for match in re.finditer(r".", text, re.DOTALL):
        ch = match.group(0)
        if _is_cjk(ch):
            cjk_run.append(ch)
        else:
            flush_cjk()
    flush_cjk()
    tokens.extend(_WORD.findall(text))
    return tokens


@dataclass
class Chunk:
    chunk_id: str
    path: str          # relative to docs root
    heading: str       # "H1 > H2 > H3" breadcrumb, "" at file top
    line_start: int    # 1-based, inclusive
    line_end: int      # 1-based, inclusive
    text: str


def chunk_markdown(rel_path: str, source: str) -> list[Chunk]:
    """Split a note on headings, then on length, preserving line numbers."""
    lines = source.split("\n")
    stack: list[tuple[int, str]] = []          # (level, title)
    sections: list[tuple[str, int, int]] = []  # (breadcrumb, start, end) 1-based
    cur_start = 1
    cur_crumb = ""

    for idx, line in enumerate(lines, start=1):
        m = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if not m:
            continue
        if idx > cur_start:
            sections.append((cur_crumb, cur_start, idx - 1))
        level, title = len(m.group(1)), m.group(2)
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        cur_crumb = " > ".join(t for _, t in stack)
        cur_start = idx
    sections.append((cur_crumb, cur_start, len(lines)))

    chunks: list[Chunk] = []
    for crumb, start, end in sections:
        for sub_start in range(start, end + 1, MAX_CHUNK_LINES):
            sub_end = min(sub_start + MAX_CHUNK_LINES - 1, end)
            text = "\n".join(lines[sub_start - 1 : sub_end])
            if len(text.strip()) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{rel_path}:{sub_start}-{sub_end}",
                    path=rel_path,
                    heading=crumb,
                    line_start=sub_start,
                    line_end=sub_end,
                    text=text,
                )
            )
    return chunks


class DocsIndex:
    """BM25 over chunk-level postings, cached to disk, rebuilt incrementally."""

    K1 = 1.5
    B = 0.75

    def __init__(self, docs_root: str | Path, cache_path: str | Path | None = None):
        self.root = Path(docs_root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"docs root is not a directory: {self.root}")
        self.cache_path = Path(cache_path) if cache_path else self.root / ".docs_cite_index.json"
        self.chunks: list[Chunk] = []
        self.links: dict[str, list[str]] = {}   # note stem -> outgoing wikilink targets
        self._stamps: dict[str, list[float]] = {}
        self._df: Counter = Counter()
        self._tf: list[Counter] = []
        self._lens: list[int] = []
        self._avg_len: float = 0.0

    # ---------- build ----------

    def _walk(self) -> list[Path]:
        out = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            out.extend(
                Path(dirpath) / f for f in filenames if f.endswith(".md")
            )
        return sorted(out)

    def build(self, force: bool = False) -> dict:
        cached: dict = {}
        if not force and self.cache_path.exists():
            try:
                cached = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                cached = {}

        prev_stamps = cached.get("stamps", {})
        prev_by_file: dict[str, list[dict]] = {}
        for raw in cached.get("chunks", []):
            prev_by_file.setdefault(raw["path"], []).append(raw)

        chunks: list[Chunk] = []
        stamps: dict[str, list[float]] = {}
        links: dict[str, list[str]] = dict(cached.get("links", {}))
        reused = rebuilt = 0

        for path in self._walk():
            rel = path.relative_to(self.root).as_posix()
            st = path.stat()
            stamp = [st.st_mtime, float(st.st_size)]
            stamps[rel] = stamp
            if prev_stamps.get(rel) == stamp and rel in prev_by_file:
                chunks.extend(Chunk(**raw) for raw in prev_by_file[rel])
                reused += 1
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            chunks.extend(chunk_markdown(rel, source))
            links[Path(rel).stem] = sorted(set(_WIKILINK.findall(source)))
            rebuilt += 1

        self.chunks = chunks
        self.links = {k: v for k, v in links.items() if k in {Path(p).stem for p in stamps}}
        self._stamps = stamps
        self._fit()
        self._save()
        return {
            "notes_indexed": len(stamps),
            "chunks": len(chunks),
            "files_reparsed": rebuilt,
            "files_reused": reused,
        }

    def _fit(self) -> None:
        self._tf = []
        self._lens = []
        self._df = Counter()
        for chunk in self.chunks:
            toks = tokenize(f"{chunk.heading}\n{chunk.text}")
            tf = Counter(toks)
            self._tf.append(tf)
            self._lens.append(len(toks))
            self._df.update(tf.keys())
        self._avg_len = (sum(self._lens) / len(self._lens)) if self._lens else 0.0

    def _save(self) -> None:
        payload = {
            "version": 1,
            "stamps": self._stamps,
            "links": self.links,
            "chunks": [asdict(c) for c in self.chunks],
        }
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self.cache_path)

    # ---------- query ----------

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not self.chunks:
            return []
        q = [t for t in tokenize(query) if self._df.get(t)]
        if not q:
            return []
        n = len(self.chunks)
        scores = [0.0] * n
        for term in set(q):
            df = self._df[term]
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self._tf):
                f = tf.get(term)
                if not f:
                    continue
                denom = f + self.K1 * (
                    1 - self.B + self.B * self._lens[i] / (self._avg_len or 1)
                )
                scores[i] += idf * (f * (self.K1 + 1)) / denom
        ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
        return [
            {
                "chunk_id": self.chunks[i].chunk_id,
                "path": self.chunks[i].path,
                "heading": self.chunks[i].heading,
                "line_start": self.chunks[i].line_start,
                "line_end": self.chunks[i].line_end,
                "score": round(scores[i], 4),
                "text": self.chunks[i].text,
            }
            for i in ranked
            if scores[i] > 0
        ]

    def resolve(self, rel_path: str) -> Path:
        """Resolve a root-relative path, refusing anything outside the docs root."""
        candidate = (self.root / rel_path).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError(f"path escapes the docs root: {rel_path}")
        if not candidate.is_file():
            raise FileNotFoundError(rel_path)
        return candidate

    def read_span(self, rel_path: str, line_start: int, line_end: int) -> dict:
        path = self.resolve(rel_path)
        lines = path.read_text(encoding="utf-8").split("\n")
        start = max(1, int(line_start))
        end = min(len(lines), int(line_end))
        if end < start:
            raise ValueError(f"empty span {line_start}-{line_end}")
        return {
            "path": rel_path,
            "line_start": start,
            "line_end": end,
            "total_lines": len(lines),
            "text": "\n".join(lines[start - 1 : end]),
        }

    def backlinks(self, note_stem: str) -> list[str]:
        return sorted(src for src, targets in self.links.items() if note_stem in targets)
