# docs-cite-mcp

An MCP server that lets a coding agent search a markdown documentation tree and
cite exactly where each answer came from.

Works on any directory of `.md` files: a project's `docs/`, a tree of engineering
reports, or an Obsidian vault.

Search returns `path` + `line_start` + `line_end` for every passage, and
`read_span` lets the agent go read that exact range back out of the file. An
agent using this can show its source instead of producing an answer from
context.

## Why these four tools

The Nokia JD asks for agents "connected to engineering systems — issue tracking,
documentation, source control, and test data — so they fetch facts instead of
inventing them", and for "source traceability, tests, and review gates". The
mapping:

| JD phrase | What implements it |
|---|---|
| MCP servers and Python helpers | `server.py` — four tools over stdio |
| search documentation | `search_notes` |
| fetch facts instead of inventing them | line spans on every hit; `read_span` verifies |
| source traceability | the round-trip invariant in `test_index.py` |
| guardrails | result caps, span caps, path-escape guard, structured error returns |
| measure whether the tools improve quality | `eval_search.py` |

## Setup

```bash
pip install mcp                    # only external dependency; everything else is stdlib
python test_index.py               # verify the citation invariant first

# try it on the corpus shipped in this repo
python eval_search.py --docs ./sample-docs --gold gold.example.jsonl --k 3

export DOCS_ROOT="$PWD/sample-docs"   # or your own docs tree
claude mcp add docs-cite -- python "$PWD/server.py"
```

Then in Claude Code: *"search the docs for the BM25 baseline and cite the lines."*

`sample-docs/` is a small corpus committed here so the tests, the quickstart and
CI all run with no setup. Point `--docs` at a real tree to get a meaningful
benchmark.

## Files

- `docs_index.py` — heading-aware chunker that preserves 1-based line spans, plus
  BM25. CJK runs are tokenized as character bigrams, since whitespace
  tokenization silently returns nothing for Chinese text.
- `server.py` — the MCP server. Every tool clamps its inputs and returns
  `{"ok": false, "error": ...}` rather than raising into the agent.
- `test_index.py` — the invariant that makes provenance real: every chunk's
  recorded span must reproduce its text when sliced out of the source file.
  Also covers path escapes, incremental rebuild, and CJK retrieval.
- `eval_search.py` — Recall@K / NDCG@K / MRR over a hand-written gold set, with
  a paired bootstrap for comparing two configs.

## Build order

**Phase 1 (a few hours).** Point it at a real corpus, run `test_index.py`, wire
it into Claude Code, use it for a day. Fix whatever breaks on real documents —
frontmatter, code fences, generated files, very large files.

**Phase 2 (an afternoon).** Write 30-50 gold queries in `gold.jsonl` from
questions you have actually asked the corpus, and run `eval_search.py`. The
`zero_hit_queries` list is the interesting output: those are the failure modes.
The shipped example already surfaces one: a Chinese-language query against
English documents, which lexical matching cannot bridge.

**Phase 3 (optional).** Add a dense channel with sentence-transformers + FAISS
and fuse with RRF. Only do this if Phase 2 shows lexical search actually missing
things — and report the gain from the paired bootstrap, not from eyeballing.

## Prior art

Several MCP servers already expose Obsidian vaults. Most offer CRUD over the vault
(read / create / update / search), and many depend on the Obsidian Local REST API
plugin or the 1.12 CLI, so the app has to be running.

This one is narrower on purpose:

- **Line-level provenance, not file-level.** Existing servers return a matching
  document; this returns the passage plus the exact line span it occupies, and a
  `read_span` tool to verify it.
- **Retrieval quality is measured.** `eval_search.py` reports Recall@K / NDCG@K
  against a gold set, with a paired bootstrap for config changes. The gold set and
  corpus in this repo are public, so the numbers are reproducible.
- **Not Obsidian-specific.** Reads any markdown tree off disk. No plugin, no
  running app, no Node; stdlib only apart from `mcp`.

## Do not commit private corpora

`.docs_cite_index.json` stores the **full text** of every chunk of every document,
as do the eval caches under `.cache/`. Both are in `.gitignore` here — but the
cache is written next to the corpus by default, so if you point this at a private
tree that is itself a git repository, add `.docs_cite_index.json` to *that*
repository's `.gitignore` as well.

`gold.jsonl` is gitignored for the same reason: real queries describe the corpus
they were written against. `gold.example.jsonl` is the shareable stand-in.
