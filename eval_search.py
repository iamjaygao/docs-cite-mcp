"""
Retrieval evaluation for the docs index.

Without this the server is a demo. With it you can say what the retrieval
quality actually is, and whether a config change helped or was noise.

Gold format (gold.jsonl), one object per line:
    {"query": "why does JP retrieval degrade", "relevant": ["research/two-tower.md"]}

Relevance is judged at note level, so writing 30-50 of these by hand from
questions you have actually asked this corpus is an afternoon, not a project.

Usage:
    python eval_search.py --docs ./sample-docs --gold gold.jsonl
    python eval_search.py --docs ./sample-docs --gold gold.jsonl --compare chunk_lines=20,40
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import docs_index
from docs_index import DocsIndex


# ---------- metrics ----------

def recall_at_k(ranked_paths: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked_paths[:k]) & relevant) / len(relevant)


def ndcg_at_k(ranked_paths: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, path in enumerate(ranked_paths[:k], start=1)
        if path in relevant
    )
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def rr_at_k(ranked_paths: list[str], relevant: set[str], k: int) -> float:
    for rank, path in enumerate(ranked_paths[:k], start=1):
        if path in relevant:
            return 1.0 / rank
    return 0.0


# ---------- runner ----------

def dedup_paths(hits: list[dict]) -> list[str]:
    """Chunk hits -> document ranking, first occurrence wins."""
    seen: list[str] = []
    for hit in hits:
        if hit["path"] not in seen:
            seen.append(hit["path"])
    return seen


def run(index: DocsIndex, gold: list[dict], k: int, depth: int) -> dict:
    per_query = []
    for row in gold:
        relevant = set(row["relevant"])
        ranked = dedup_paths(index.search(row["query"], k=depth))
        per_query.append(
            {
                "query": row["query"],
                "recall": recall_at_k(ranked, relevant, k),
                "ndcg": ndcg_at_k(ranked, relevant, k),
                "rr": rr_at_k(ranked, relevant, k),
                "found": bool(set(ranked[:k]) & relevant),
            }
        )
    n = len(per_query) or 1
    return {
        "queries": len(per_query),
        f"recall@{k}": sum(q["recall"] for q in per_query) / n,
        f"ndcg@{k}": sum(q["ndcg"] for q in per_query) / n,
        f"mrr@{k}": sum(q["rr"] for q in per_query) / n,
        "zero_hit_queries": [q["query"] for q in per_query if not q["found"]],
        "per_query": per_query,
    }


def paired_bootstrap(
    a: list[float], b: list[float], iterations: int = 10000, seed: int = 0
) -> dict:
    """
    Resample queries (not observations) to compare two configs on the same gold
    set. Reports the mean difference, a 95% interval, and a two-sided p-value.
    """
    if len(a) != len(b):
        raise ValueError("paired bootstrap needs the same queries in both runs")
    rng = random.Random(seed)
    n = len(a)
    observed = sum(b) / n - sum(a) / n
    diffs = [b[i] - a[i] for i in range(n)]
    centered = [d - observed for d in diffs]
    samples = []
    extreme = 0
    for _ in range(iterations):
        idx = [rng.randrange(n) for _ in range(n)]
        samples.append(sum(diffs[i] for i in idx) / n)
        if abs(sum(centered[i] for i in idx) / n) >= abs(observed):
            extreme += 1
    samples.sort()
    lo = samples[int(0.025 * iterations)]
    hi = samples[int(0.975 * iterations) - 1]
    return {
        "delta": round(observed, 5),
        "ci95": [round(lo, 5), round(hi, 5)],
        "p_value": round((extreme + 1) / (iterations + 1), 5),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--depth", type=int, default=20, help="chunks retrieved before dedup")
    ap.add_argument(
        "--compare",
        help="A/B one knob, e.g. chunk_lines=20,40 -- reindexes and paired-bootstraps",
    )
    args = ap.parse_args()

    gold = [
        json.loads(line)
        for line in Path(args.gold).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not gold:
        print("gold set is empty")
        return 1

    if not args.compare:
        index = DocsIndex(args.docs)
        index.build()
        result = run(index, gold, args.k, args.depth)
        print(json.dumps({k: v for k, v in result.items() if k != "per_query"}, indent=2,
                         ensure_ascii=False))
        return 0

    knob, raw = args.compare.split("=", 1)
    if knob != "chunk_lines":
        print(f"unsupported knob: {knob}")
        return 1
    left, right = (int(v) for v in raw.split(","))

    runs = {}
    for value in (left, right):
        docs_index.MAX_CHUNK_LINES = value
        cache_dir = Path(".cache")
        cache_dir.mkdir(exist_ok=True)
        index = DocsIndex(args.docs, cache_path=cache_dir / f"eval_index_{value}.json")
        index.build(force=True)
        runs[value] = run(index, gold, args.k, args.depth)

    for metric in (f"recall@{args.k}", f"ndcg@{args.k}"):
        key = "recall" if metric.startswith("recall") else "ndcg"
        stats = paired_bootstrap(
            [q[key] for q in runs[left]["per_query"]],
            [q[key] for q in runs[right]["per_query"]],
        )
        print(
            f"{metric}: chunk_lines={left} {runs[left][metric]:.4f} -> "
            f"chunk_lines={right} {runs[right][metric]:.4f}  {stats}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
