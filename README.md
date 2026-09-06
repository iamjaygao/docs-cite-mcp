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

# retrieval quality on the corpus shipped in this repo
.venv/bin/python eval_search.py --docs ./sample-docs --gold gold.example.jsonl --k 3
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

| Tool | Arguments | Returns |
|---|---|---|
| `search_notes` | `query`, `k` (1-20) | passages with `path`, `heading`, `line_start`, `line_end`, `score` |
| `read_span` | `path`, `line_start`, `line_end` | the exact lines, for verifying a citation |
| `list_backlinks` | `note` | documents whose `[[wikilinks]]` point at it |
| `reindex` | `force` | rescans; only changed files are reparsed |

Every tool clamps its inputs and returns `{"ok": false, "error": ...}` instead of
raising into the agent.

## How it works

Documents are split on markdown headings, then on length. Each chunk records the
1-based line span it occupies in the source file, and **that span must reproduce
the chunk text exactly when sliced back out**. `test_index.py` asserts this on
every chunk; if it fails, the server is inventing provenance rather than
reporting it.

Retrieval is BM25 over chunks. Latin text is tokenized on word boundaries; CJK
runs are emitted as character bigrams plus unigrams, because whitespace
tokenization returns nothing useful for Chinese or Japanese.

The index is cached next to the corpus and rebuilt per changed file.

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

## Known limitations

- **Long documents can dominate results.** A large file becomes many chunks, each
  matching a little, and can crowd out a shorter, more authoritative document.
  There is no per-document cap on results yet.
- **`search_notes` returns chunks; `eval_search.py` scores documents.** The eval
  deduplicates chunks to their source file before computing metrics, so the
  numbers describe document-level recall, not what an agent actually sees in its
  result list.
- **Code fences are not parsed.** A `#` at the start of a line inside a fenced
  block is treated as a heading, which can distort chunk boundaries and the
  heading breadcrumb.
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
