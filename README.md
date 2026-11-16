# docs-cite-mcp

An MCP server that lets a coding agent search a documentation tree of Markdown,
PDF and Word files and cite exactly where each answer came from.

Search returns `path` + `line_start` + `line_end` for every passage (plus `page`
for a PDF), and `read_span` reads that exact range back out of the file. An agent
using this can show its source instead of producing an answer from context.

Retrieval is hybrid: BM25 for exact names, numbers and codes, fused with a small
local embedding model for paraphrases. Nothing is sent to an external service.

Works on any directory of `.md`, `.pdf` and `.docx` files: a project's `docs/`,
a tree of engineering reports, or an Obsidian vault.

## Requirements

- Python 3.10 or newer (required by the MCP SDK; macOS system Python is 3.9)
- `mcp>=2,<3`
- Optional: `pypdf` for PDF and `python-docx` for Word (BSD and MIT licensed).
  Without them those files are listed in `reindex`'s `files_skipped` and
  Markdown works as before; the Markdown path is stdlib-only.
- Optional: `fastembed` (Apache-2.0) for hybrid retrieval. It runs an ONNX model
  on the CPU, no PyTorch, no GPU. The model (~70 MB) is downloaded once into
  `~/.cache/docs-cite-mcp/models` (`DOCS_CITE_MODEL_DIR`). Without fastembed, or
  with `DOCS_CITE_EMBED_MODEL=none`, search is BM25 only and every result says so.

## Quickstart

```bash
python3.11 -m venv .venv          # any interpreter >= 3.10
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

.venv/bin/python test_index.py    # verifies the citation invariant
.venv/bin/python test_server.py   # verifies the tool contract through an MCP client

# retrieval quality on the corpus shipped in this repo, and lexical vs hybrid
.venv/bin/python eval_search.py --docs ./sample-docs --gold gold.example.jsonl --k 3
.venv/bin/python eval_search.py --docs ./sample-docs --gold gold.example.jsonl --k 3 \
  --compare mode=lexical,hybrid

# the agent loop end to end, over stdio: search, cite, read each span back
.venv/bin/python examples/cite_and_verify.py "what is the BM25 baseline NDCG"
.venv/bin/python examples/cite_and_verify.py "how far did median latency fall"   # PDF
```

Register with Claude Code. Run `--prepare` once first, so the model download and
the first embedding pass don't happen inside the agent's first search:

```bash
DOCS_ROOT="$PWD/sample-docs" .venv/bin/python server.py --prepare
claude mcp add docs-cite \
  --env DOCS_ROOT="$PWD/sample-docs" \
  -- "$PWD/.venv/bin/python" "$PWD/server.py"
```

`DOCS_ROOT` must be passed with `--env`. The server is spawned as a subprocess
and does not inherit your shell environment, so exporting the variable first has
no effect — the server silently falls back to `./sample-docs` and answers every
query from the wrong corpus.

Then, in Claude Code: *"search the docs for the BM25 baseline and cite the lines."*

## Tools

| Tool | Arguments | Returns | Hints |
|---|---|---|---|
| `search_notes` | `query`, `k` (1-20), `max_per_doc` (default 2), `mode` (`hybrid` / `lexical` / `dense`) | passages with `path`, `heading`, `page`, `line_start`, `line_end`, `cite`, `score`, `lexical_rank`, `dense_rank`; and the `mode` used | read-only |
| `read_span` | `path`, `line_start`, `line_end` (≥ 1, ≤ 400 lines), `page` (PDFs) | the exact lines, for verifying a citation | read-only |
| `list_backlinks` | `note` (name or path) | paths of documents whose `[[wikilinks]]` point at it | read-only |
| `reindex` | `force` | rescans; only changed files are reparsed | writes the cache, non-destructive |

### Contract

Errors are reported in two layers:

1. **Schema.** Bounds and descriptions for every parameter are in the published
   input schema, and every tool publishes an output schema (the TypedDicts in
   `server.py`). The SDK rejects a call that violates the input schema, such as
   `k=50`, with an `isError` result naming the field, so the agent can correct
   it and retry.
2. **Runtime.** Failures the schema cannot express return
   `{"ok": false, "error": <code>, "detail": ...}` and never a stack trace:

| `error` | Meaning |
|---|---|
| `empty_query` | query is blank after trimming whitespace |
| `span_too_large` | `read_span` asked for more than 400 lines |
| `invalid_span` | `line_end` is before `line_start` |
| `path_outside_root` | path resolves outside `DOCS_ROOT` |
| `not_found` | no such file under `DOCS_ROOT` |
| `page_required` | `read_span` on a PDF without a `page` |
| `page_out_of_range` | the PDF has no such page, or `page` was given for a non-PDF file |
| `unsupported_format` | not a `.md`, `.pdf` or `.docx` file |
| `missing_dependency` | `pypdf` / `python-docx` is not installed for this format |
| `no_text_layer` | a PDF with no extractable text, typically a scan |
| `unreadable` | bad UTF-8, a corrupt or password-protected document, or permissions |
| `internal_error` | unexpected failure; `detail` carries the exception |

`test_server.py` checks the schemas, the hints and each error code through a real
MCP client session.

## How it works

### Formats and what a line number means

Every format is first read into a *view*, a list of lines. Indexing and
`read_span` both read the view through the same function, so a citation
round-trips for every format, not only Markdown:

| Format | A citation looks like | A "line" is |
|---|---|---|
| Markdown | `guides/indexing.md:3-8` | a line of the file |
| PDF | `reports/latency-review.pdf p.1:1-4` | a line of that page's extracted text; chunks never cross a page |
| Word | `guides/deployment-runbook.docx ¶3-5` | a block: one paragraph, or one table row rendered as `\| a \| b \|` |

Word `Heading N` and `Title` styles become the breadcrumb, so a Word hit says
which section it came from. Body text that happens to start with `#` is escaped,
so only the document's real styles create sections. Scanned PDFs (no text
layer), corrupt or password-protected files, and Word lock files are reported in
`files_skipped` rather than indexed as empty.

### Chunking and ranking

Documents are split on markdown headings, then on length. A `#` inside a fenced
code block is not a heading, so a shell or Python comment does not start a new
section. Each chunk records the
1-based line span it occupies in the source file, and **that span must reproduce
the chunk text exactly when sliced back out**. `test_index.py` asserts this on
every chunk; if it fails, the server is inventing provenance rather than
reporting it.

A section longer than 40 lines is cut at a blank line, so a paragraph or table
stays in one chunk; with no blank line nearby it falls back to a hard cut.

Lexical retrieval is BM25 over an inverted index of chunks. Latin text is
tokenized on word boundaries; CJK runs are emitted as character bigrams plus
unigrams, because whitespace tokenization returns nothing useful for Chinese or
Japanese.

Dense retrieval embeds the same text BM25 scores (breadcrumb plus chunk) with
`BAAI/bge-small-en-v1.5`, adding the query instruction prefix the model was
trained with, and ranks by cosine similarity. The search is brute force over a
matrix of unit vectors: exact, and about 3 ms for 500 chunks, so an approximate
index would add a dependency without a measurable gain at this scale.

Hybrid (the default) takes the top 50 of each ranking and fuses them with
reciprocal rank fusion, `score = Σ 1 / (60 + rank)`. RRF uses only ranks, so
BM25 scores and cosine similarities never have to be put on one scale. Each
hit reports its `lexical_rank` and `dense_rank`, so a ranking can be explained.

A result list holds at most two chunks from any one document (`max_per_doc`),
so one long document that matches a little in many chunks cannot take every
slot. The cap is applied after fusion; capping each ranking first would let one
document back in twice.

Vectors live beside the BM25 statistics in each file's cache entry, so they
inherit everything below: an unchanged file is never re-embedded, an edited one
re-embeds only its own chunks, and switching models discards the old vectors.
Vectors are rounded to float16 as soon as they are computed, so a fresh index
and one loaded from cache are bit-identical and rank identically.

### Keeping citations true while the docs change

A line span is only a citation if it still matches the file when someone checks
it. The index keeps that true while a server is running:

- **Fresh.** Before returning hits, a search re-stats the files it is about to
  cite and rebuilds if any changed, so a span never points into an older version
  of a file. Added and deleted files are picked up by a stat sweep at most every
  two seconds. Calling `reindex` is never required for correctness.
- **Atomic.** Each build produces an immutable snapshot and swaps it in with one
  assignment. A search holds one snapshot for its whole run, so a concurrent
  `reindex` cannot give it chunks and statistics that disagree.
- **Versioned.** The cache records the chunker version, chunk settings, the
  embedding model, and which parsers are installed. A cache written by
  different code is discarded, so a chunking fix reaches every file rather than
  only the ones edited since; and a cache built where pypdf was installed never
  serves PDF citations where `read_span` could not open the PDF.
- **Bounded.** A symlink that resolves outside the docs root is not indexed,
  because its citations could not be read back. Indexing and `read_span` decode
  files identically (a UTF-8 byte-order mark is dropped, CRLF becomes LF), and
  files that cannot be decoded are listed in `reindex`'s `files_skipped`.

The cache stores per-chunk term counts, so an unchanged file is never
re-tokenized, and an unchanged index never rewrites its cache. The cache is an
optimisation only: on a read-only corpus the server still works, and `reindex`
reports why the cache could not be written.

## Evaluation

`eval_search.py` reports Recall@K, NDCG@K and MRR against a gold set of
`{"query": ..., "relevant": [paths]}` lines, and includes a paired bootstrap for
comparing two configurations:

```bash
.venv/bin/python eval_search.py --docs ./sample-docs --gold gold.example.jsonl \
  --compare chunk_lines=20,40
.venv/bin/python eval_search.py --docs ./sample-docs --gold gold.example.jsonl \
  --compare mode=lexical,hybrid
```

The `zero_hit_queries` field is the useful output — those are the failure modes.
The shipped example surfaces one: a Chinese-language query against English
documents, which lexical matching cannot bridge.

`--unit document` (the default) asks whether the retriever found the right file
at all. `--unit chunk` scores the list the agent actually receives, where a
second chunk from an already-found document takes a slot but earns nothing. The
gap between the two is result crowding, measured.

`--min-recall`, `--min-ndcg` and `--min-mrr` turn a run into a quality gate that
exits 1 when a metric drops below its floor. CI runs it on the sample corpus for
both lexical and hybrid retrieval, with floors just under the current scores, so
a change that loses a query fails the build. The hybrid floors sit above the
lexical scores, so hybrid silently degrading to BM25 also fails it.

### What the gold set decided

Four retrieval changes were each tried against a private 40-query gold set over
a real research-docs corpus (k=5, paired bootstrap over queries, 10,000
resamples). Only the ones the data supported shipped:

| Change | Chunk-level NDCG@5 Δ | 95% CI | p | Decision |
|---|---:|---|---:|---|
| Cap 2 chunks per document | +0.017 | [0.001, 0.036] | 0.05 | **shipped** (default) |
| Cap 1 chunk per document | +0.059 | [0.026, 0.099] | 0.004 | available via `max_per_doc=1`, not the default |
| Cut long sections at a blank line | +0.006 | [0.000, 0.019] | 0.25 | **shipped**: no query got worse |
| Singularise plurals | +0.008 | [−0.024, 0.047] | 0.68 | rejected: helped some queries, hurt others |
| Count a section's heading once, not twice | −0.006 | [−0.035, 0.019] | 0.67 | rejected: MRR@5 fell 0.753 → 0.715 |

The two shipped changes together: chunk-level NDCG@5 0.618 → 0.640 (p = 0.02),
Recall@5 0.654 → 0.688 (p = 0.04), and the share of the top five held by a single
document 0.505 → 0.355. Document-level scores are unchanged or slightly up.

A cap of 1 scores higher, but the gold set labels documents, not passages, so it
cannot see the cost of hiding a document's second-best passage. 2 is the
default for that reason.

### Lexical, dense, or hybrid

Dense retrieval was added with its acceptance criteria written down before the
experiment ran: hybrid would become the default only if, on the 40-query set,
chunk-level NDCG@5 improved with a 95% interval excluding zero, document-level
Recall@5 did not fall, and median search latency stayed under 50 ms. Two small
CPU models were compared (k=5, `max_per_doc=2`, paired bootstrap against
lexical):

| Retrieval | Chunk NDCG@5 | Δ vs lexical, 95% CI | p | Chunk R@5 | Doc R@5 | Search p50 |
|---|---:|---|---:|---:|---:|---:|
| Lexical (BM25) | 0.640 | | | 0.687 | 0.729 | 0.3 ms |
| Dense, bge-small-en-v1.5 | 0.729 | +0.089 [+0.012, +0.174] | 0.03 | 0.758 | 0.808 | 3.3 ms |
| **Hybrid, bge-small-en-v1.5** | **0.740** | **+0.100 [+0.039, +0.165]** | **0.002** | **0.792** | **0.808** | **3.7 ms** |
| Dense, multilingual MiniLM-L12 | 0.635 | −0.005 [−0.089, +0.087] | 0.91 | 0.650 | 0.683 | 3.0 ms |
| Hybrid, multilingual MiniLM-L12 | 0.701 | +0.061 [+0.015, +0.111] | 0.01 | 0.758 | 0.792 | 3.4 ms |

Hybrid with bge-small met all three criteria and is the default. Dense alone
gained about as much on average but with an interval twice as wide: it wins big
on some queries and loses on others, and fusion keeps BM25's wins on exact
terms. Of the five queries lexical retrieval missed entirely, hybrid recovered
two and lost one that lexical had found, so it still misses four. One of those
asks for a model's "architecture" and "baseline" while its document uses
neither word: the vocabulary gap dense retrieval is meant to bridge, and this
small model did not.

The cost is indexing time: embedding 488 chunks takes about 17 s on a laptop CPU
against 0.03 s for BM25, paid once and then only for edited files.

bge-small is English-only. For other languages set
`DOCS_CITE_EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`;
on this English corpus it was weaker, and it truncates at 128 tokens.

## Known limitations

- **Relevance is labelled per document.** The gold format names files, not line
  spans, so the eval cannot tell whether the cited passage within a correct file
  is the one that answers the question.
- **The quality gate is only as broad as its gold set.** CI gates on ten
  queries over five sample documents (Markdown, PDF, Word), which catches gross
  regressions but cannot resolve small ranking changes. The ranking decisions
  above were made on the 40-query set, which is Markdown only.
- **PDF line numbers are of the extracted text.** They match what `read_span`
  returns, not necessarily the visual lines on the page: multi-column layouts,
  running headers and footers come out in pypdf's reading order. There is no
  OCR, so scanned PDFs are skipped.
- **Word citations are block numbers.** Word does not display them; the `cite`
  string and breadcrumb are how a person finds the passage. Text boxes,
  footnotes and comments are not extracted, and legacy `.doc` is not supported.
- **Small models, no reranker.** bge-small reads at most 512 tokens of a chunk
  and is English-only; the multilingual alternative reads 128. There is no
  cross-encoder reranking stage after fusion.
- **Dense retrieval was chosen on one corpus.** The 40-query set is one team's
  English research docs; other corpora may favour another model or lexical
  retrieval, which is why `mode` is a per-query parameter and the comparison is
  one command (`--compare mode=lexical,hybrid`).
- **The whole index lives in memory.** Fine for thousands of documents; untested
  well beyond that.

## Prior art

Several MCP servers already expose Obsidian vaults. Most offer CRUD over the
vault (read / create / update / search), and many depend on the Obsidian Local
REST API plugin or the 1.12 CLI, so the app has to be running.

This one is narrower on purpose:

- **Line-level provenance, not file-level.** Existing servers return a matching
  document; this returns the passage plus the exact line span it occupies, and a
  tool to verify it.
- **Retrieval quality is measured.** The gold set and corpus in this repo are
  public, so the numbers are reproducible.
- **Not Obsidian-specific.** Reads any tree of Markdown, PDF and Word files off disk. No plugin, no
  running app, no Node.

## Do not commit private corpora

`.docs_cite_index.json` stores the **full text** of every chunk of every
document, as do the eval caches under `.cache/`. Both are in `.gitignore` here —
but the cache is written next to the corpus by default, so if you point this at a
private tree that is itself a git repository, add `.docs_cite_index.json` to
*that* repository's `.gitignore` as well.

`gold.jsonl` is gitignored for the same reason: real queries describe the corpus
they were written against. `gold.example.jsonl` is the shareable stand-in.

## License

MIT
