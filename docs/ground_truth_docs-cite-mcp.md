# Résumé ground truth: docs-cite-mcp

Every number below comes from a saved output file in this repository. If a number is not in a file listed here, it does not go on a résumé.

**Evidence files** (generated 2026-11-19 UTC, from code at commit `8749f8e`, with no uncommitted changes to tracked files):

| Key | File | Produced by |
|---|---|---|
| R | `results/retrieval_experiments.json` | `bench/retrieval_experiments.py --docs <private corpus> --gold gold.jsonl` |
| P | `results/perf.json` | `bench/perf.py --old d7629a2` |
| S | `results/sample_eval_lexical.json`, `results/sample_eval_hybrid.json` | `eval_search.py --docs ./sample-docs --gold gold.example.jsonl --k 3 --mode {lexical,hybrid}` |
| CI | GitHub Actions runs 43792620563, 43793787156, 43887132657, 43889994510 | `.github/workflows/test.yml` on commits `6d2a82d`, `9f4a60b`, `483ce19`, `8749f8e` |
| Code | `server.py`, `docs_index.py`, `dense.py` at `8749f8e` | — |

**Populations**:

- **G40**: the private gold set `gold.jsonl` (sha256 `53cfa5b92e68…`, recorded in R). It has 41 lines, 40 of them labeled. Relevance is document-level and binary. The set was labeled by the project owner with `make_gold.py`, which hides the retriever's ranking until a query is labeled. The corpus is one research project's documentation: 33 Markdown files (content hash `848a62c9a793…` in R), cut into 488 chunks with the shipped chunker and 487 with hard 40-line cuts. Settings: k=5, retrieval depth 20. The paired bootstrap resamples queries 10,000 times with seed 0. Each arm was run once.
- **S10**: the public sample in `sample-docs/` (3 Markdown files, 1 PDF, 1 Word file) with `gold.example.jsonl` (10 queries), k=3.
- **P300**: a synthetic corpus of 300 documents and 1,200 chunks, generated from seed 0 with a vocabulary of 20,000 tokens. Lexical retrieval only. Two code versions, `d7629a2` (old) and `8749f8e` (current), were timed in one session on one machine (macOS arm64, 16 CPUs, Python 3.11.15). Each timing is the median of 5 repeats.

---

## 1. What the project is

docs-cite-mcp is a Model Context Protocol server, written in Python, that indexes a directory of Markdown, PDF and Word files. It answers search queries with passages that each carry a file path, a line range, and for PDFs a page number. A second tool, `read_span`, returns that line range from the file so the citation can be checked. Each search hit also carries a hash of the file's text. If the file changes before the citation is checked, `read_span` returns the error `stale_citation` instead of different text. Retrieval combines BM25 with a local embedding model (`BAAI/bge-small-en-v1.5`, run through fastembed on the CPU) using reciprocal rank fusion. A result list holds at most two passages from any one document. The repository includes an evaluation script (Recall@k, NDCG@k, MRR@k at document and passage level, with a paired bootstrap), a labeling tool, tests, and CI on Python 3.10, 3.11 and 3.12.

## 2. Ownership

At the owner's instruction, this file does not track authorship line by line. Git records a single author for every commit. The experiment design, the acceptance criteria, and the decision to ship or reject each change are described under each claim. Be ready to explain any line of the code in an interview.

There are no teammates on this project.

## 3. Claims

Status: **VERIFIED** means a saved artifact produces this exact number. **INFERRED** means it follows from the code or the design, but no artifact shows the number. **RETRACTED** means it appeared in docs, the README or earlier notes, but no artifact produces it (listed in §5).

### VERIFIED

| # | Claim | Exact value | Résumé form | Metric | Population and setup | Source |
|---|---|---|---|---|---|---|
| V1 | Hybrid retrieval vs shipped BM25, passage level | 0.640127 → 0.740019; Δ +0.09989, 95% CI [0.03872, 0.16476], p = 0.0022 | NDCG@5 0.64 → 0.74 (p = 0.002) | chunk-level NDCG@5 | G40; both arms max_per_doc=2, blank-line chunking; bge-small-en-v1.5; single run | R `arms.lexical_shipped/hybrid_bge.units.chunk.ndcg@5`; `comparisons[7].units.chunk` |
| V2 | Same, recall | 0.687500 → 0.791667; Δ +0.10417, p = 0.0021 | Recall@5 0.69 → 0.79 | chunk-level Recall@5 | as V1 | R same keys, `recall@5` |
| V3 | Same, document level | Recall 0.729167 → 0.808333 (Δ +0.07917, p = 0.0078); NDCG 0.682946 → 0.772063 (Δ +0.08912, p = 0.0035) | document Recall@5 0.73 → 0.81 | document-level Recall@5, NDCG@5 | as V1 | R `units.document`; `comparisons[7].units.document` |
| V4 | Dense-only (bge-small) vs shipped BM25 | 0.729054; Δ +0.08893, 95% CI [0.01156, 0.17363], p = 0.0314 | — (supporting, not headline) | chunk-level NDCG@5 | as V1 | R `arms.dense_bge`; `comparisons[6]` |
| V5 | Multilingual MiniLM-L12 | dense Δ −0.00477, CI [−0.08878, 0.08742], p = 0.912; hybrid 0.701493, Δ +0.06137, CI [0.01521, 0.11142], p = 0.0124 | — | chunk-level NDCG@5 | as V1, model `paraphrase-multilingual-MiniLM-L12-v2` | R `comparisons[8]`, `[9]` |
| V6 | Queries with no relevant document in the top 5 | 5 of 40 (lexical) → 4 of 40 (hybrid); 2 rescued, 1 newly lost | — | document-level zero-hit count | as V1 | R `units.document.zero_hit_count`; `comparisons[7].units.document.zero_hit_rescued/lost` |
| V7 | Search latency | hybrid 3.702 ms; lexical 0.308 ms | 3.7 ms per query (p50) | median wall time per `search()` call, including query embedding | G40 queries, one call each, warm model, same process, Apple-silicon laptop | R `arms.hybrid_bge.search_ms_p50`, `arms.lexical_shipped.search_ms_p50` |
| V8 | Full index build | hybrid 16.989 s (embeds 488 chunks); lexical 0.043 s | — | wall time of one forced build | G40 corpus, one run | R `arms.*.build_seconds` |
| V9 | Result crowding before the fix | document Recall@5 0.720833 vs chunk Recall@5 0.654167; top-5 share held by one document 0.505 | — | Recall@5 by unit; `top_doc_share` | G40; baseline arm (hard 40-line cuts, no cap) | R `arms.base` |
| V10 | Cap of 2 chunks per document | Δ +0.01652, CI [0.00106, 0.03570], p = 0.0527; top-5 share 0.505 → 0.355 | — | chunk-level NDCG@5 | G40, against the baseline arm | R `comparisons[0]`; `arms.cap2.units.chunk.top_doc_share` |
| V11 | Cap of 1 chunk per document | Δ +0.05853, CI [0.02647, 0.09852], p = 0.0039 | — | chunk-level NDCG@5 | as V10 | R `comparisons[1]` |
| V12 | Cut long sections at a blank line | Δ +0.00557, CI [0, 0.01619], p = 0.2196; 0 queries worse, 2 better | — | chunk-level NDCG@5; per-query counts | as V10 | R `comparisons[2]`; `arms.{base,split}.units.chunk.per_query.ndcg` |
| V13 | Plural stemming (rejected) | Δ +0.00745, CI [−0.02394, 0.04654], p = 0.675 | — | chunk-level NDCG@5 | as V10 | R `comparisons[3]` |
| V14 | Count a section heading once (rejected) | chunk NDCG Δ −0.00606, CI [−0.03537, 0.01946], p = 0.668; document MRR@5 0.752917 → 0.714583 | — | chunk NDCG@5; document MRR@5 | as V10 | R `comparisons[4]`; `arms.{base,heading_once}.units.document.mrr@5` |
| V15 | Both shipped ranking changes together | chunk NDCG 0.618040 → 0.640127 (Δ +0.02209, CI [0.00557, 0.04311], p = 0.02); chunk Recall 0.654167 → 0.687500 (p = 0.0425); 0 queries worse | NDCG@5 0.618 → 0.640 | chunk-level NDCG@5, Recall@5 | as V10 | R `comparisons[5]` |
| V16 | Full index build, old vs current code | 0.9046 s → 0.1772 s | — | median of 5 | P300; lexical only | P `results.{old,current}.full_build_s_median` |
| V17 | Rebuild with no file changed | 0.8813 s → 0.0644 s | — | median of 5 | P300 | P `noop_rebuild_s_median` |
| V18 | New index instance loading an existing cache | 0.8749 s → 0.0911 s | — | median of 5 | P300 | P `fresh_instance_from_cache_s_median` |
| V19 | BM25 search | 0.2731 ms → 0.0471 ms | — | mean of 50 calls, median of 5 repeats | P300, k=10 | P `search_ms_mean_median` |
| V20 | Search racing a forced reindex | old code: 9 `IndexError`; current: 0 | — | error count in one run | P300; 4 search threads during 3 forced rebuilds; the count varies run to run | P `results.*.race` |
| V21 | Public sample set | lexical Recall@3 0.9, NDCG@3 0.863093, MRR@3 0.85; hybrid 1.0, 0.963093, 0.95 | — | document-level @3 | S10 | S |
| V22 | CI passes on three Python versions | 3.10, 3.11, 3.12 for each of 4 commits | — | job conclusion | CI | CI run IDs above |
| V23 | Tool contract size | 13 error codes; 4 tools; 3 formats (.md, .pdf, .docx) | — | count | Code | `server.ERROR_CODES`; `@mcp.tool` functions; `docs_index.SUPPORTED_SUFFIXES` |
| V24 | Corpus size behind G40 | 33 files; 488 chunks | — | count | G40 | R `corpus.files`; `arms.lexical_shipped.chunks` |

### INFERRED

| # | Claim | Basis | Why it is not VERIFIED |
|---|---|---|---|
| I1 | Embedding model is ~70 MB | fastembed's model list reports 0.067 GB for bge-small-en-v1.5 | Library metadata, not saved |
| I2 | One query that hybrid still misses asks for "architecture" and "baseline", and its document contains neither word | A grep run during the experiments | Grep output was not saved |
| I3 | Every chunk's line span reproduces its text, in any corpus | `test_index.py` asserts it on its fixtures and on the sample corpus; CI passes | Tested on fixtures; holding on every corpus is a design property |
| I4 | A citation is never stale without being detected | `test_server.py` runs search → edit → stale → re-search → verify through an MCP client; CI passes | Behavior is tested; there is no count to report |

## 4. Safe résumé sentences

Each sentence keeps the qualifiers that make it true. Combine at most two in one bullet.

- **V1, V2**: "Added hybrid retrieval (BM25 fused with a local embedding model via reciprocal rank fusion) under acceptance criteria fixed before the experiment; on a 40-query, self-labeled gold set, passage-level NDCG@5 rose from 0.64 to 0.74 against the shipped BM25 ranking (paired bootstrap, p = 0.002)."
- **V3**: "…and document-level Recall@5 from 0.73 to 0.81 on the same 40 queries."
- **V7**: "Hybrid search runs at 3.7 ms median per query on a laptop CPU, including query embedding, over a 488-passage corpus."
- **V6**: "Hybrid retrieval recovered 2 of the 5 queries that BM25 missed entirely, and lost 1 that BM25 had found."
- **V9, V10**: "Measured result crowding that document-level metrics hid: on the baseline ranking, the share of top-5 slots held by a single document averaged 0.505 over 40 queries; capping chunks per document lowered it to 0.355."
- **V15**: "Shipped two ranking changes only after paired bootstrap tests; together they raised passage-level NDCG@5 from 0.618 to 0.640 (p = 0.02) with no query scoring worse, and two other candidate changes were measured and rejected."
- **V13, V14**: "Rejected plural stemming and heading de-duplication after measuring them: no significant gain, and heading de-duplication lowered document-level MRR@5 from 0.753 to 0.715."
- **V16–V19**: "Measured indexing on a 300-document synthetic corpus: rebuilds with no file changed went from 0.88 s to 0.064 s between two code versions, and BM25 queries from 0.27 ms to 0.047 ms." (The verb is "measured": the two versions differ by several changes, so neither number belongs to one fix.)
- **V20**: "Found and fixed a race between search and reindex; a reproduction raised 9 IndexErrors in one run on the old code and none on the fixed code." (Say "in one run": the count varies.)
- **V21**: "CI gates retrieval quality on a public 10-query sample (hybrid NDCG@3 0.963 vs 0.863 for BM25), so a ranking regression fails the build." (State the size: 10 queries can catch gross regressions only.)
- **V22**: "CI runs the test suite on Python 3.10, 3.11 and 3.12."
- **V23**: "Typed MCP tool contract: input and output schemas, 4 tools, 13 stable error codes."

## 5. Do-not-use list

| Number or statement | Where it appeared | Reason |
|---|---|---|
| "0.92 s → 0.056 s" no-op rebuild; "0.9 s → 0.17 s" build; "0.3 ms → 0.04 ms" search | Earlier session notes; interview handbook | Ad-hoc measurement, never saved. Use V16–V19, which differ in the second decimal. |
| "5 IndexErrors" | Earlier session notes; handbook | Never saved. The saved run (V20) shows 9, and the count varies by run. |
| "Dense alone … with an interval twice as wide" | README (corrected) | False. The CI widths are 0.162 (dense) and 0.126 (hybrid). |
| "MiniLM truncates at 128 tokens" | README (corrected) | The tokenizer config fastembed loads sets `model_max_length` 512. There is no evidence for 128 in this setup. |
| "Brute-force search, about 3 ms for 500 chunks" | README (corrected) | 3.49 ms (dense) is the whole search call including query embedding. The matrix product was never timed on its own. |
| "BM25 search p50 0.27 ms" on the real corpus | Handbook | Unsaved run. The saved value is 0.308 ms (V7). |
| Split row "CI [0.000, 0.019], p = 0.25" labeled chunk-level | README (corrected) | Those are the document-level values. Chunk-level is CI [0, 0.016], p = 0.22 (V12). |
| "Dense, bge 3.3 ms", "Dense, MiniLM 3.0 ms", lexical chunk R@5 "0.687", hybrid MiniLM "0.701" | README table (corrected) | Earlier unsaved run or rounding. Saved: 3.49 ms, 3.24 ms, 0.6875 (0.688), 0.7015 (0.702). |
| "The gold set and corpus in this repo are public, so the numbers are reproducible" | README Prior art (corrected) | Only the 10-query sample is public. The decision numbers come from the private 40-query set. |
| Anything in `baseline_28q_document.json` / `baseline_28q_chunk.json` | Committed in `d7629a2` | Saved outputs, but a re-run at `d7629a2` on today's corpus and gold set does not reproduce them (only `top_doc_share` 0.5357 matches). The labels or documents changed since, and the 28-query population cannot be compared with G40. |
| Any "×-fold faster" or percentage speedup | — | No artifact contains a ratio. Quote the two measured values instead. |
| "Improved retrieval by N%" | — | No artifact contains a percentage change. Quote the before and after values with the metric and population. |
| Old-vs-current perf presented as one fix's effect | — | V16–V19 compare two commits several changes apart. |
| Comparing V21 (10 queries, k=3) with V1–V3 (40 queries, k=5) | — | Different populations and cutoffs. |

## 6. Interview depth

**Setup limits an interviewer may probe**

- **One corpus and 40 queries.** The gold set was labeled by one person on their own research docs, at document level, with binary relevance. The confidence intervals are wide: V1's interval spans 0.039 to 0.165.
- **Document-level labels.** The eval cannot tell whether the cited passage inside a correct file is the one that answers the question. The cap of 2 was chosen over a cap of 1 by reasoning, not by measurement: the cap of 1 scores higher (V11) because a document-level metric cannot see a hidden second passage.
- **Multiple comparisons.** Ten paired comparisons were run. V1 (p = 0.002) stays significant under a Bonferroni threshold of 0.05/10 = 0.005; V10 (p = 0.053) and V15 (p = 0.02) do not.
- **Single run.** Lexical results are deterministic. Dense results come from one ONNX run on one machine, and latency is one call per query.
- **Synthetic performance corpus.** P300 uses random tokens, so the timings show relative cost, not latency on real text.
- **Unreproducible early baselines.** The 28-query baselines from September cannot be regenerated, so the project's history before `6d2a82d` has no usable numbers.

**What I would do differently**

- Label evidence passages, not just documents, from the start. This is the most important change: it would let the eval test whether hybrid ranks the evidence higher, and whether a cap of 2 helps an agent collect all the evidence for multi-passage questions.
- Save every experiment's output when it runs, with the code version, instead of reconstructing it later.
- Score the list the agent actually receives (chunk level) from the first eval, not after the fact.

**Open questions**

- Does hybrid retrieval help a real agent answer more questions correctly, or only rank documents better? There is no agent-level evaluation yet: answer accuracy, citation support, and correct "not in the corpus" answers.
- Would a cross-encoder reranker fix the misses where the query's words are in the right document but it ranks too low?
- How does the system behave above a few thousand documents? The whole index is in memory, and nothing larger was tested.
- Prompt injection: document text is returned to the agent verbatim, and nothing guards against instructions embedded in a document.
