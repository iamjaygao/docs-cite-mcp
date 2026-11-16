# docs-cite-mcp

An MCP server that lets a coding agent search a markdown documentation tree and
cite exactly where each answer came from.

Search returns `path` + `line_start` + `line_end` for every passage, and
`read_span` reads that exact range back out of the file. An agent using this can
show its source instead of producing an answer from context.

Works on any directory of `.md` files: a project's `docs/`, a tree of engineering
reports, or an Obsidian vault.

## Requirements

- Python 3.10 or newer (required by the MCP SDK; macOS system Python is 3.9)
- `mcp>=2,<3` — the only external dependency. Everything else is stdlib.

## Quickstart

```bash
python3.11 -m venv .venv          # any interpreter >= 3.10
.venv/bin/pip install --upgrade pip
.venv/bin/pip install "mcp>=2,<3"

.venv/bin/python test_index.py    # verifies the citation invariant
.venv/bin/python test_server.py   # verifies the tool contract through an MCP client

# retrieval quality on the corpus shipped in this repo
.venv/bin/python eval_search.py --docs ./sample-docs --gold gold.example.jsonl --k 3

# the agent loop end to end, over stdio: search, cite, read each span back
.venv/bin/python examples/cite_and_verify.py "what is the BM25 baseline NDCG"
```

Register with Claude Code:

```bash
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
| `search_notes` | `query`, `k` (1-20), `max_per_doc` (default 2) | passages with `path`, `heading`, `line_start`, `line_end`, `score` | read-only |
| `read_span` | `path`, `line_start`, `line_end` (≥ 1, ≤ 400 lines) | the exact lines, for verifying a citation | read-only |
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
| `unreadable` | file exists but is not valid UTF-8 or cannot be read |
| `internal_error` | unexpected failure; `detail` carries the exception |

`test_server.py` checks the schemas, the hints and each error code through a real
MCP client session.

## How it works

Documents are split on markdown headings, then on length. A `#` inside a fenced
code block is not a heading, so a shell or Python comment does not start a new
section. Each chunk records the
1-based line span it occupies in the source file, and **that span must reproduce
the chunk text exactly when sliced back out**. `test_index.py` asserts this on
every chunk; if it fails, the server is inventing provenance rather than
reporting it.

A section longer than 40 lines is cut at a blank line, so a paragraph or table
stays in one chunk; with no blank line nearby it falls back to a hard cut.

Retrieval is BM25 over an inverted index of chunks. Latin text is tokenized on
word boundaries; CJK runs are emitted as character bigrams plus unigrams, because
whitespace tokenization returns nothing useful for Chinese or Japanese.

A result list holds at most two chunks from any one document (`max_per_doc`),
so one long document that matches a little in many chunks cannot take every
slot.

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
- **Versioned.** The cache records the chunker version and chunk settings. A
  cache written by different code is discarded, so a chunking fix reaches every
  file rather than only the ones edited since.
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
```

The `zero_hit_queries` field is the useful output — those are the failure modes.
The shipped example surfaces one: a Chinese-language query against English
documents, which lexical matching cannot bridge.

`--unit document` (the default) asks whether the retriever found the right file
at all. `--unit chunk` scores the list the agent actually receives, where a
second chunk from an already-found document takes a slot but earns nothing. The
gap between the two is result crowding, measured.

`--min-recall`, `--min-ndcg` and `--min-mrr` turn a run into a quality gate that
exits 1 when a metric drops below its floor. CI runs it on the sample corpus with
floors just under the current scores, so a change that loses a query fails the
build.

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

## Known limitations

- **Relevance is labelled per document.** The gold format names files, not line
  spans, so the eval cannot tell whether the cited passage within a correct file
  is the one that answers the question.
- **The quality gate is only as broad as its gold set.** CI gates on seven
  queries over three sample documents, which catches gross regressions but
  cannot resolve small ranking changes.
- **Lexical only.** No dense retrieval, so paraphrases and cross-language queries
  miss.
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
- **Not Obsidian-specific.** Reads any markdown tree off disk. No plugin, no
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
