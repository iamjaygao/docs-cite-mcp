"""
Line-accurate chunking + BM25 retrieval over a documentation tree of Markdown,
PDF and Word files.

Design rule: every chunk carries the exact 1-based line span it came from, and
that span must round-trip against the source. If it doesn't, a citation is
decoration rather than a reference. test_index.py enforces this.

Each format is read into a *view*: the lines that both indexing and read_span
see, produced by one function, so the round-trip holds for every format.

  * Markdown: the file's own lines.
  * PDF: the extracted text layer, one view per page. A citation is a page plus
    a line range within that page.
  * Word (.docx): one line per block -- a paragraph, or a table row rendered as
    "| a | b |" -- with Heading/Title styles rendered as "#" headings, so the
    Markdown chunker and breadcrumbs apply unchanged.

PDF and Word support needs pypdf / python-docx. Without them those files are
reported in files_skipped and Markdown works as before.

Keeping that true while the corpus changes underneath a running server takes
three more rules:

  * Fresh. A search re-stats the files it is about to cite and rebuilds if any
    changed, so a span never points into an older version of the file. New and
    deleted files are picked up by a cheap stat sweep, at most every
    REFRESH_INTERVAL seconds.
  * Atomic. Each build produces an immutable snapshot and swaps it in with one
    assignment. A search holds a single snapshot for its whole run, so a
    concurrent reindex can never hand it chunks and statistics that disagree.
  * Versioned. The on-disk cache records the chunker version and settings; a
    cache written by different code is discarded, not reused.

The Markdown path is stdlib only, so it runs anywhere without an install step.
"""

from __future__ import annotations

import functools
import heapq
import json
import math
import os
import re
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

# Bump whenever chunking, tokenizing or the cache layout changes: caches written
# under another version are discarded instead of silently reused.
CACHE_VERSION = 4
SUPPORTED_SUFFIXES = {".md", ".pdf", ".docx"}
SKIP_DIRS = {".obsidian", ".trash", ".git", "node_modules", ".DS_Store"}
MAX_CHUNK_LINES = 40          # long sections get split further, at a paragraph break
MIN_CHUNK_CHARS = 20          # drop near-empty fragments
REFRESH_INTERVAL = 2.0        # seconds between full stat sweeps for added/removed files

# Chunks one document may contribute to a result list. Without a cap, a long
# document matching a little in many chunks fills every slot (result crowding).
# 2 rather than 1: the gold set is document-level, so it cannot see the cost of
# hiding a document's second-best passage, which flatters a cap of 1.
DEFAULT_MAX_PER_DOC = 2

_WORD = re.compile(r"[a-z0-9]+")
_CJK_RUN = re.compile(r"[぀-ヿ一-鿿가-힯]+")  # kana, CJK unified, hangul
_WIKILINK = re.compile(r"\[\[([^\]|#]+)")
_HEADING = re.compile(r"^ {0,3}(#{1,6})\s+(.*\S)\s*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


class PathOutsideRoot(ValueError):
    """A requested path resolves outside the docs root."""


class InvalidSpan(ValueError):
    """A requested line range is empty or inverted."""


class PageRequired(ValueError):
    """read_span on a PDF was called without a page."""


class PageOutOfRange(ValueError):
    """The page does not exist, or a page was given for a non-PDF file."""


class UnsupportedFormat(ValueError):
    """The file type is not one of SUPPORTED_SUFFIXES."""


class MissingDependency(RuntimeError):
    """The optional library for this format is not installed."""


class NoTextLayer(ValueError):
    """A PDF with no extractable text, typically a scan. OCR is not supported."""


class UnreadableDocument(ValueError):
    """A PDF or Word file the parser rejected: corrupt, encrypted, not really that format."""


# ---------- corpus membership: the single definition every tool shares ----------

def iter_corpus_files(root: Path) -> list[Path]:
    """
    Every supported file that belongs to the corpus, in a stable order.

    Skips dot-directories and SKIP_DIRS, does not descend into directory
    symlinks, and drops file symlinks that resolve outside the root -- those
    could be searched but never read back, so their citations could not be
    verified. Word lock files (~$name.docx) are not documents. make_gold.py
    lists the corpus through this same function.
    """
    root = root.resolve()
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")
        )
        for name in sorted(filenames):
            if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES or name.startswith("~$"):
                continue
            path = Path(dirpath) / name
            if path.is_symlink() and not path.resolve().is_relative_to(root):
                continue
            out.append(path)
    return out


def read_markdown(path: Path) -> str:
    """
    Read a note the same way for indexing and for read_span.

    utf-8-sig drops a leading byte-order mark, which would otherwise hide a
    first-line heading. Universal newlines turn CRLF into LF on both paths.
    """
    return path.read_text(encoding="utf-8-sig")


# ---------- views: what indexing and read_span both read ----------

@dataclass(frozen=True)
class DocView:
    kind: str                              # "markdown" | "pdf" | "docx"
    pages: tuple[tuple[str, ...], ...]     # PDF: one entry per page; others: exactly one


def load_view(path: Path) -> DocView:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return DocView("markdown", (tuple(read_markdown(path).split("\n")),))
    if suffix == ".pdf":
        return DocView("pdf", _pdf_pages(path))
    if suffix == ".docx":
        return DocView("docx", (_docx_lines(path),))
    raise UnsupportedFormat(f"unsupported file type: {path.name}")


@functools.lru_cache(maxsize=32)
def _cached_view(path: str, stamp: tuple[int, int] | None) -> DocView:
    # keyed on the stamp, so an edited file is never served from the cache
    return load_view(Path(path))


def document_text(path: Path) -> str:
    """A whole view as one string, for listing and grepping (make_gold.py)."""
    view = load_view(path)
    if view.kind != "pdf":
        return "\n".join(view.pages[0])
    return "\n".join(
        line for n, page in enumerate(view.pages, start=1) for line in (f"[page {n}]", *page)
    )


def _pdf_pages(path: Path) -> tuple[tuple[str, ...], ...]:
    try:
        import pypdf
    except ImportError as exc:
        raise MissingDependency("PDF support needs pypdf: pip install pypdf") from exc
    try:
        reader = pypdf.PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise UnreadableDocument("PDF is password-protected")
        pages = tuple(
            tuple((page.extract_text() or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"))
            for page in reader.pages
        )
    except UnreadableDocument:
        raise
    except Exception as exc:  # parsers of untrusted binaries raise anything
        raise UnreadableDocument(f"{type(exc).__name__}: {exc}") from exc
    if not any(line.strip() for page in pages for line in page):
        raise NoTextLayer("no extractable text (a scanned PDF?); OCR is not supported")
    return pages


_DOCX_HEADING = re.compile(r"^Heading (\d)$")


def _docx_lines(path: Path) -> tuple[str, ...]:
    try:
        import docx
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as exc:
        raise MissingDependency("Word support needs python-docx: pip install python-docx") from exc
    try:
        document = docx.Document(str(path))
        lines: list[str] = []
        for element in document.element.body.iterchildren():
            tag = element.tag.rsplit("}", 1)[-1]
            if tag == "p":
                lines.append(_docx_paragraph(Paragraph(element, document)))
            elif tag == "tbl":
                for row in Table(element, document).rows:
                    cells, seen = [], set()
                    for cell in row.cells:           # a merged cell repeats: keep it once
                        if id(cell._tc) not in seen:
                            seen.add(id(cell._tc))
                            cells.append(" ".join(cell.text.split()))
                    lines.append("| " + " | ".join(cells) + " |")
    except Exception as exc:
        raise UnreadableDocument(f"{type(exc).__name__}: {exc}") from exc
    return tuple(lines)


def _docx_paragraph(paragraph) -> str:
    text = " ".join(paragraph.text.split("\n"))      # soft line breaks stay in one block
    style = paragraph.style.name if paragraph.style is not None else ""
    m = _DOCX_HEADING.match(style)
    level = 1 if style == "Title" else int(m.group(1)) if m else 0
    if level and text.strip():
        return "#" * min(level, 6) + " " + text.strip()
    # body text that would read as a heading or a code fence is escaped, so the
    # Markdown chunker only ever sees structure that Word's styles declared
    if _HEADING.match(text) or _FENCE.match(text):
        return "\\" + text
    return text


def _stamp(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


# ---------- tokenizing and chunking ----------

def tokenize(text: str) -> list[str]:
    """
    Latin text -> word tokens. CJK runs -> character unigrams plus bigrams.

    Whitespace tokenization silently fails on Chinese and Japanese; bigrams are
    the cheap fix that does not need a segmenter. (Same failure mode I hit on
    the ES/JP locales in the ESCI work.)
    """
    text = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    for run in _CJK_RUN.findall(text):
        tokens.extend(run)
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    tokens.extend(_WORD.findall(text))
    return tokens


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    path: str          # relative to docs root
    heading: str       # "H1 > H2 > H3" breadcrumb, "" at file top
    line_start: int    # 1-based, inclusive (within `page` for a PDF)
    line_end: int      # 1-based, inclusive
    text: str
    page: int | None = None   # PDF page, 1-based; None for other formats

    @property
    def cite(self) -> str:
        """Human-readable locator: path:12-30, path p.3:12-30, path ¶12-30 (Word blocks)."""
        if self.page is not None:
            return f"{self.path} p.{self.page}:{self.line_start}-{self.line_end}"
        if self.path.lower().endswith(".docx"):
            return f"{self.path} ¶{self.line_start}-{self.line_end}"
        return f"{self.path}:{self.line_start}-{self.line_end}"


def chunk_markdown(rel_path: str, source: str) -> list[Chunk]:
    """
    Split a note on headings, then on length, preserving line numbers.

    Lines inside fenced code blocks are never headings: a `# comment` in a
    shell or Python fence must not start a new section or enter the breadcrumb.
    An unclosed fence runs to end of file, as in CommonMark.
    """
    lines = source.split("\n")
    stack: list[tuple[int, str]] = []          # (level, title)
    sections: list[tuple[str, int, int]] = []  # (breadcrumb, start, end) 1-based
    cur_start = 1
    cur_crumb = ""
    fence: str | None = None                   # opening fence marker while inside one

    for idx, line in enumerate(lines, start=1):
        f = _FENCE.match(line)
        if fence is None and f:
            fence = f.group(1)
            continue
        if fence is not None:
            # closes on the same character, at least as long, nothing after it
            if f and f.group(1)[0] == fence[0] and len(f.group(1)) >= len(fence) \
                    and not line.strip()[len(f.group(1)):].strip():
                fence = None
            continue
        m = _HEADING.match(line)
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
        for sub_start, sub_end in _split_section(lines, start, end):
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


def chunk_view(rel_path: str, view: DocView) -> list[Chunk]:
    """Chunk any view. PDF chunks never cross a page, so a citation names one page."""
    if view.kind != "pdf":
        return chunk_markdown(rel_path, "\n".join(view.pages[0]))
    chunks: list[Chunk] = []
    for page_no, page in enumerate(view.pages, start=1):
        lines = list(page)
        for s, e in _split_section(lines, 1, len(lines)):
            text = "\n".join(lines[s - 1 : e])
            if len(text.strip()) < MIN_CHUNK_CHARS:
                continue
            chunks.append(Chunk(
                chunk_id=f"{rel_path}#p{page_no}:{s}-{e}", path=rel_path, heading="",
                line_start=s, line_end=e, text=text, page=page_no,
            ))
    return chunks


def _split_section(lines: list[str], start: int, end: int) -> list[tuple[int, int]]:
    """
    Cut [start, end] (1-based, inclusive) into spans of at most MAX_CHUNK_LINES.

    Each cut moves back to the last blank line in the second half of the
    window, so a paragraph or table is not split across two chunks; with no
    blank line there, it falls back to the hard cut.
    """
    spans = []
    s = start
    while s <= end:
        e = min(s + MAX_CHUNK_LINES - 1, end)
        if e < end:
            floor = s + MAX_CHUNK_LINES // 2
            for b in range(e, floor - 1, -1):
                if not lines[b - 1].strip():
                    e = b
                    break
        spans.append((s, e))
        s = e + 1
    return spans


def _index_text(chunk: Chunk) -> str:
    """
    The text BM25 scores: breadcrumb plus body. A chunk that opens with its own
    heading counts that heading twice. That is deliberate: removing the
    duplicate lowered MRR@5 from 0.753 to 0.715 on the gold set, because
    heading terms are the strongest signal of what a section is about.
    """
    return f"{chunk.heading}\n{chunk.text}"


# ---------- index state ----------

@dataclass(frozen=True)
class FileRecord:
    """Everything derived from one file. Reused verbatim while its stamp holds."""
    stamp: tuple[int, int]
    chunks: tuple[Chunk, ...] = ()
    term_freqs: tuple[dict[str, int], ...] = ()   # parallel to chunks
    links: tuple[str, ...] = ()                   # outgoing [[wikilink]] targets
    error: str | None = None                      # set when the file could not be read

    @classmethod
    def parse(cls, rel: str, path: Path, stamp: tuple[int, int]) -> "FileRecord":
        try:
            view = load_view(path)
        except (UnicodeDecodeError, OSError, ValueError, MissingDependency) as exc:
            # keep the stamp, so an unreadable file is not retried on every refresh
            return cls(stamp=stamp, error=f"{type(exc).__name__}: {exc}")
        chunks = tuple(chunk_view(rel, view))
        links: tuple[str, ...] = ()
        if view.kind == "markdown":                  # [[wikilinks]] are a Markdown convention
            source = "\n".join(view.pages[0])
            links = tuple(sorted({t.strip() for t in _WIKILINK.findall(source) if t.strip()}))
        return cls(
            stamp=stamp,
            chunks=chunks,
            term_freqs=tuple(dict(Counter(tokenize(_index_text(c)))) for c in chunks),
            links=links,
        )

    def to_json(self) -> dict:
        return {
            "stamp": list(self.stamp),
            "chunks": [{**asdict(c), "tf": tf} for c, tf in zip(self.chunks, self.term_freqs)],
            "links": list(self.links),
            "error": self.error,
        }

    @classmethod
    def from_json(cls, raw: dict) -> "FileRecord":
        chunks, tfs = [], []
        for c in raw["chunks"]:
            c = dict(c)
            tfs.append(c.pop("tf"))
            chunks.append(Chunk(**c))
        return cls(
            stamp=tuple(raw["stamp"]),
            chunks=tuple(chunks),
            term_freqs=tuple(tfs),
            links=tuple(raw["links"]),
            error=raw.get("error"),
        )


@dataclass(frozen=True)
class _Snapshot:
    """One consistent view of the index. Never mutated after construction."""
    files: dict[str, FileRecord] = field(default_factory=dict)
    chunks: tuple[Chunk, ...] = ()
    lens: tuple[int, ...] = ()
    avg_len: float = 0.0
    postings: dict[str, tuple[tuple[int, int], ...]] = field(default_factory=dict)

    @classmethod
    def of(cls, files: dict[str, FileRecord]) -> "_Snapshot":
        chunks: list[Chunk] = []
        tfs: list[dict[str, int]] = []
        for record in files.values():
            chunks.extend(record.chunks)
            tfs.extend(record.term_freqs)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, tf in enumerate(tfs):
            for term, freq in tf.items():
                postings[term].append((i, freq))
        lens = tuple(sum(tf.values()) for tf in tfs)
        return cls(
            files=files,
            chunks=tuple(chunks),
            lens=lens,
            avg_len=(sum(lens) / len(lens)) if lens else 0.0,
            postings={t: tuple(p) for t, p in postings.items()},
        )

    def stamps(self) -> dict[str, tuple[int, int]]:
        return {rel: rec.stamp for rel, rec in self.files.items()}


def _link_matches(target: str, note: str) -> bool:
    """[[target]] points at `note` if the names agree and any folders given agree."""
    t = target.removesuffix(".md").strip("/")
    n = note.removesuffix(".md").strip("/")
    if PurePosixPath(t).name != PurePosixPath(n).name:
        return False
    if "/" not in t or "/" not in n:
        return True
    return t == n or t.endswith("/" + n) or n.endswith("/" + t)


class DocsIndex:
    """BM25 over chunk-level postings, cached to disk, rebuilt incrementally."""

    K1 = 1.5
    B = 0.75

    def __init__(self, docs_root: str | Path, cache_path: str | Path | None = None,
                 refresh_interval: float = REFRESH_INTERVAL):
        self.root = Path(docs_root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"docs root is not a directory: {self.root}")
        self.cache_path = Path(cache_path) if cache_path else self.root / ".docs_cite_index.json"
        self.refresh_interval = refresh_interval
        self.cache_error: str | None = None
        self._snap = _Snapshot()
        self._build_lock = threading.Lock()
        self._last_sweep = float("-inf")

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._snap.chunks)

    # ---------- build ----------

    @staticmethod
    def _cache_key() -> list:
        # chunk settings are part of the key: eval_search changes MAX_CHUNK_LINES at runtime
        return [CACHE_VERSION, MAX_CHUNK_LINES, MIN_CHUNK_CHARS]

    def _load_cache(self) -> dict[str, FileRecord]:
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if raw.get("key") != self._cache_key():
                return {}
            return {rel: FileRecord.from_json(rec) for rel, rec in raw["files"].items()}
        except (OSError, ValueError, KeyError, TypeError):
            return {}

    def _save(self, files: dict[str, FileRecord]) -> bool:
        """Best effort: a corpus we cannot write to is still a corpus we can search."""
        payload = {"key": self._cache_key(), "files": {r: f.to_json() for r, f in files.items()}}
        tmp = self.cache_path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(self.cache_path)
        except OSError as exc:
            self.cache_error = f"{type(exc).__name__}: {exc}"
            return False
        self.cache_error = None
        return True

    def build(self, force: bool = False) -> dict:
        """Rescan the tree, reparsing only files whose (mtime_ns, size) changed."""
        with self._build_lock:
            previous = {} if force else (self._snap.files or self._load_cache())
            files: dict[str, FileRecord] = {}
            reparsed = reused = 0
            for path in iter_corpus_files(self.root):
                stamp = _stamp(path)
                if stamp is None:                 # vanished mid-walk
                    continue
                rel = path.relative_to(self.root).as_posix()
                cached = previous.get(rel)
                if cached is not None and cached.stamp == stamp:
                    files[rel] = cached
                    reused += 1
                else:
                    files[rel] = FileRecord.parse(rel, path, stamp)
                    reparsed += 1

            changed = force or reparsed > 0 or files.keys() != previous.keys()
            self._snap = _Snapshot.of(files)       # the single atomic swap
            self._last_sweep = time.monotonic()
            cache_written = self._save(files) if (changed or not self.cache_path.exists()) else False

            skipped = sorted(rel for rel, rec in files.items() if rec.error)
            return {
                "notes_indexed": len(files) - len(skipped),
                "chunks": len(self._snap.chunks),
                "files_reparsed": reparsed,
                "files_reused": reused,
                "files_skipped": skipped,
                "cache_written": cache_written,
                "cache_error": self.cache_error,
            }

    def refresh(self) -> bool:
        """
        Rebuild if any file was added, removed or changed. The sweep is a stat
        per file, no reads, and runs at most once per refresh_interval.
        """
        if time.monotonic() - self._last_sweep < self.refresh_interval:
            return False
        self._last_sweep = time.monotonic()
        current = {
            p.relative_to(self.root).as_posix(): _stamp(p)
            for p in iter_corpus_files(self.root)
        }
        if current == self._snap.stamps():
            return False
        self.build()
        return True

    # ---------- query ----------

    def _rank(self, snap: _Snapshot, query: str, k: int,
              max_per_doc: int | None = None) -> list[tuple[int, float]]:
        terms = {t for t in tokenize(query) if t in snap.postings}
        if not terms:
            return []
        n = len(snap.chunks)
        avg = snap.avg_len or 1
        scores: dict[int, float] = defaultdict(float)
        for term in terms:
            plist = snap.postings[term]
            idf = math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, f in plist:
                denom = f + self.K1 * (1 - self.B + self.B * snap.lens[i] / avg)
                scores[i] += idf * (f * (self.K1 + 1)) / denom
        candidates = sorted(i for i, s in scores.items() if s > 0)  # corpus order breaks ties
        if not max_per_doc:
            # nlargest is documented equal to sorted(...)[:k], so ties stay stable
            return [(i, scores[i]) for i in heapq.nlargest(k, candidates, key=scores.__getitem__)]
        # Result crowding: one long document can match a little in many chunks and
        # fill every slot. Cap chunks per document so other documents get a slot.
        ranked: list[tuple[int, float]] = []
        per_doc: Counter = Counter()
        for i in sorted(candidates, key=scores.__getitem__, reverse=True):
            path = snap.chunks[i].path
            if per_doc[path] < max_per_doc:
                per_doc[path] += 1
                ranked.append((i, scores[i]))
                if len(ranked) == k:
                    break
        return ranked

    def search(self, query: str, k: int = 5, max_per_doc: int | None = None) -> list[dict]:
        """
        Top-k chunks for `query`. max_per_doc caps how many chunks one document
        may contribute; None leaves the ranking uncapped.
        """
        self.refresh()
        snap = self._snap                          # hold one snapshot for the whole query
        ranked = self._rank(snap, query, k, max_per_doc)
        # exact check on the files about to be cited, whatever the sweep interval
        cited = {snap.chunks[i].path for i, _ in ranked}
        if any(_stamp(self.root / rel) != snap.files[rel].stamp for rel in cited):
            self.build()
            snap = self._snap
            ranked = self._rank(snap, query, k, max_per_doc)
        return [
            {
                "chunk_id": c.chunk_id,
                "path": c.path,
                "heading": c.heading,
                "page": c.page,
                "line_start": c.line_start,
                "line_end": c.line_end,
                "cite": c.cite,
                "score": round(score, 4),
                "text": c.text,
            }
            for c, score in ((snap.chunks[i], s) for i, s in ranked)
        ]

    def resolve(self, rel_path: str) -> Path:
        """Resolve a root-relative path, refusing anything outside the docs root."""
        candidate = (self.root / rel_path).resolve()
        if not candidate.is_relative_to(self.root):
            raise PathOutsideRoot(f"path escapes the docs root: {rel_path}")
        if not candidate.is_file():
            raise FileNotFoundError(rel_path)
        return candidate

    def read_span(self, rel_path: str, line_start: int, line_end: int,
                  page: int | None = None) -> dict:
        """
        Read lines back from the same view the index was built from. For a PDF,
        `page` is required and the lines are within that page.
        """
        path = self.resolve(rel_path)
        view = _cached_view(str(path), _stamp(path))
        if view.kind == "pdf":
            if page is None:
                raise PageRequired(f"{rel_path} is a PDF: pass page (1-{len(view.pages)})")
            if not 1 <= page <= len(view.pages):
                raise PageOutOfRange(f"{rel_path} has {len(view.pages)} pages, asked for {page}")
            lines = view.pages[page - 1]
        else:
            if page is not None:
                raise PageOutOfRange(f"page applies only to PDFs; {rel_path} is {view.kind}")
            lines = view.pages[0]
        start = max(1, int(line_start))
        end = min(len(lines), int(line_end))
        if end < start:
            raise InvalidSpan(f"empty span {line_start}-{line_end}")
        return {
            "path": rel_path,
            "page": page,
            "line_start": start,
            "line_end": end,
            "total_lines": len(lines),
            "total_pages": len(view.pages),
            "text": "\n".join(lines[start - 1 : end]),
        }

    def backlinks(self, note: str) -> list[str]:
        """Root-relative paths of the documents whose [[wikilinks]] point at `note`."""
        self.refresh()
        return sorted(
            rel for rel, rec in self._snap.files.items()
            if any(_link_matches(t, note) for t in rec.links)
        )
