# Ranking Baselines

Reference numbers for the retrieval stack. Every figure here carries the
candidate pool it was computed on; figures from different pools are not
comparable.

## Full-list NDCG, test split

| Model | NDCG | Notes |
|---|---|---|
| Random floor | 0.7503 | full-list, no cutoff |
| BM25 | 0.8198 | lexical only |
| Two-Tower | 0.8245 | dense, English-only encoder |
| LambdaMART | 0.8429 | 16 usable features |

## Known caveats

The dense retriever is trained on US-locale pairs only, so Japanese queries are
served zero-shot by an English-only encoder. The gap there is expected behaviour,
not a diagnosed defect: no ablation isolates the cause.

See [[evaluation-protocol]] for how these were produced.
