"""
Re-run every retrieval comparison the README reports, on one code version, one
corpus and one gold set, and save the results.

Each comparison is paired: both arms rank the same queries over the same corpus
with the same code, differing only in the setting under test. Settings the
shipped code no longer exposes (hard 40-line cuts, plural stemming, counting a
heading once) are restored here by patching docs_index for the duration of one
arm; the product code is not changed.

Saved per arm and unit: aggregate metrics, per-query scores in gold-file order
(no query text), zero-hit counts, index build time and search latency; per
comparison: paired bootstrap of NDCG@k and Recall@k. Corpus and gold set are
identified by content hashes, not paths, so a private corpus is not named.

    python bench/retrieval_experiments.py --docs PATH --gold gold.jsonl \
        --out results/retrieval_experiments.json
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import docs_index  # noqa: E402
from docs_index import DocsIndex, iter_corpus_files, load_view, view_version  # noqa: E402
from eval_search import paired_bootstrap, run  # noqa: E402

K, DEPTH = 5, 20
MODELS = {"bge": "BAAI/bge-small-en-v1.5",
          "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"}


# ---------- settings the shipped code no longer exposes ----------

def _hard_cut(lines, start, end):
    """Pre-change chunking: cut every MAX_CHUNK_LINES lines, ignoring blank lines."""
    return [(s, min(s + docs_index.MAX_CHUNK_LINES - 1, end))
            for s in range(start, end + 1, docs_index.MAX_CHUNK_LINES)]


def _stem_plural(word: str) -> str:
    if len(word) <= 3 or not word.isascii() or not word.isalpha():
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _heading_once(chunk):
    crumb = chunk.heading
    if crumb:
        m = docs_index._HEADING.match(chunk.text.split("\n", 1)[0])
        if m and m.group(2) == crumb.rsplit(" > ", 1)[-1]:
            crumb = crumb.rsplit(" > ", 1)[0] if " > " in crumb else ""
    return f"{crumb}\n{chunk.text}"


@contextlib.contextmanager
def patched(hard_cut=False, stem=False, heading_once=False):
    saved = (docs_index._split_section, docs_index.tokenize, docs_index._index_text)
    if hard_cut:
        docs_index._split_section = _hard_cut
    if stem:
        original = saved[1]
        docs_index.tokenize = lambda text: [_stem_plural(t) for t in original(text)]
    if heading_once:
        docs_index._index_text = _heading_once
    try:
        yield
    finally:
        docs_index._split_section, docs_index.tokenize, docs_index._index_text = saved


# ---------- arms and comparisons ----------

# name: (patches, max_per_doc, mode, model)
ARMS = {
    "base":            (dict(hard_cut=True), None, "lexical", None),
    "cap2":            (dict(hard_cut=True), 2, "lexical", None),
    "cap1":            (dict(hard_cut=True), 1, "lexical", None),
    "split":           (dict(), None, "lexical", None),
    "stem":            (dict(hard_cut=True, stem=True), None, "lexical", None),
    "heading_once":    (dict(hard_cut=True, heading_once=True), None, "lexical", None),
    "lexical_shipped": (dict(), 2, "lexical", None),
    "dense_bge":       (dict(), 2, "dense", "bge"),
    "hybrid_bge":      (dict(), 2, "hybrid", "bge"),
    "dense_minilm":    (dict(), 2, "dense", "minilm"),
    "hybrid_minilm":   (dict(), 2, "hybrid", "minilm"),
}

# (round, baseline arm, treatment arm)
COMPARISONS = [
    ("ranking", "base", "cap2"),
    ("ranking", "base", "cap1"),
    ("ranking", "base", "split"),
    ("ranking", "base", "stem"),
    ("ranking", "base", "heading_once"),
    ("ranking", "base", "lexical_shipped"),
    ("retrieval", "lexical_shipped", "dense_bge"),
    ("retrieval", "lexical_shipped", "hybrid_bge"),
    ("retrieval", "lexical_shipped", "dense_minilm"),
    ("retrieval", "lexical_shipped", "hybrid_minilm"),
]


def run_arm(name, docs, gold, embedders):
    patches, cap, mode, model = ARMS[name]
    with patched(**patches):
        idx = DocsIndex(docs, cache_path=Path(tempfile.mkdtemp()) / "idx.json",
                        refresh_interval=3600, embedder=embedders.get(model))
        t = time.perf_counter()
        stats = idx.build(force=True)
        build_s = time.perf_counter() - t
        latencies = []
        for row in gold:
            t = time.perf_counter()
            idx.search(row["query"], k=K, max_per_doc=cap, mode=mode)
            latencies.append((time.perf_counter() - t) * 1000)
        units = {}
        for unit in ("document", "chunk"):
            r = run(idx, gold, K, DEPTH, unit, cap, mode)
            zero = {i for i, q in enumerate(r["per_query"]) if not q["found"]}
            units[unit] = {
                "queries": r["queries"],
                f"recall@{K}": r[f"recall@{K}"],
                f"ndcg@{K}": r[f"ndcg@{K}"],
                f"mrr@{K}": r[f"mrr@{K}"],
                "top_doc_share": r["top_doc_share"],
                "zero_hit_count": len(zero),
                "zero_hit_query_indices": sorted(zero),
                "per_query": {m: [q[m] for q in r["per_query"]] for m in ("recall", "ndcg", "rr")},
            }
    return {
        "setting": {**patches, "max_per_doc": cap, "mode": mode,
                    "model": MODELS.get(model)},
        "chunks": stats["chunks"],
        "build_seconds": build_s,
        "search_ms_p50": statistics.median(latencies),
        "units": units,
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corpus_fingerprint(root: Path) -> dict:
    files = iter_corpus_files(root)
    h = hashlib.sha256()
    for p in files:
        try:
            h.update(view_version(load_view(p)).encode())
        except Exception as exc:                      # unreadable files still count
            h.update(type(exc).__name__.encode())
    return {"files": len(files), "content_sha256": h.hexdigest()}


def git_state() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           cwd=REPO, capture_output=True, text=True).stdout.strip()
    return {"commit": head, "uncommitted_changes": bool(dirty)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iterations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gold_path = Path(args.gold)
    gold = [json.loads(l) for l in gold_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    gold = [g for g in gold if g.get("relevant")]

    from dense import FastEmbedder
    embedders = {short: FastEmbedder(model) for short, model in MODELS.items()}
    for e in embedders.values():
        e.embed(["warm up"], "query")                  # model load outside the timings

    arms = {}
    for name in ARMS:
        print(f"arm {name} ...", file=sys.stderr, flush=True)
        arms[name] = run_arm(name, args.docs, gold, embedders)

    comparisons = []
    for round_, a, b in COMPARISONS:
        entry = {"round": round_, "baseline": a, "treatment": b, "units": {}}
        for unit in ("document", "chunk"):
            pa, pb = arms[a]["units"][unit]["per_query"], arms[b]["units"][unit]["per_query"]
            za = set(arms[a]["units"][unit]["zero_hit_query_indices"])
            zb = set(arms[b]["units"][unit]["zero_hit_query_indices"])
            entry["units"][unit] = {
                f"ndcg@{K}": paired_bootstrap(pa["ndcg"], pb["ndcg"], args.iterations, args.seed),
                f"recall@{K}": paired_bootstrap(pa["recall"], pb["recall"], args.iterations, args.seed),
                "zero_hit_rescued": len(za - zb),
                "zero_hit_lost": len(zb - za),
            }
        comparisons.append(entry)

    import importlib.metadata as md
    out = {
        "script": "bench/retrieval_experiments.py",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": {p: md.version(p) for p in ("fastembed", "onnxruntime", "numpy", "pypdf", "python-docx")},
        },
        "corpus": corpus_fingerprint(Path(args.docs)),
        "gold": {"sha256": sha256_file(gold_path), "labeled_queries": len(gold),
                 "relevance": "document-level, binary"},
        "setup": {"k": K, "depth": DEPTH, "bootstrap_iterations": args.iterations,
                  "bootstrap_seed": args.seed, "bootstrap_unit": "query, paired",
                  "latency": "single run per query, warm model, same process, laptop CPU"},
        "arms": arms,
        "comparisons": comparisons,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
