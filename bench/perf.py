"""
Indexing and search timings, and the reindex/search race, for two code versions
on one synthetic corpus in one session, and save the results.

The corpus is generated from a fixed seed so every run times the same input.
Each code version runs in its own subprocess (its own docs_index import). The
older version is extracted with `git archive`; the current one is the working tree.

    python bench/perf.py --old d7629a2 --out results/perf.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS, LINES, VOCAB, SEED, REPEATS = 300, 60, 20000, 0, 5
QUERY = "w17 w4242 w999 w12345"


def make_corpus(root: Path) -> None:
    rng = random.Random(SEED)
    words = [f"w{i}" for i in range(VOCAB)]
    for d in range(DOCS):
        body = "\n".join(" ".join(rng.choices(words, k=12)) for _ in range(LINES))
        (root / f"d{d}.md").write_text(f"# Doc {d}\n{body}\n## Part two\n{body}\n", encoding="utf-8")


def worker(code_dir: str, corpus: str) -> dict:
    """Runs inside a subprocess with code_dir first on sys.path."""
    sys.path.insert(0, code_dir)
    import docs_index
    assert Path(docs_index.__file__).parent == Path(code_dir), docs_index.__file__

    def new_index(cache):
        try:
            return docs_index.DocsIndex(corpus, cache_path=cache, refresh_interval=3600)
        except TypeError:                                  # older versions: no refresh_interval
            return docs_index.DocsIndex(corpus, cache_path=cache)

    full, noop, cold, search = [], [], [], []
    chunks = None
    for _ in range(REPEATS):
        cache = Path(tempfile.mkdtemp()) / "idx.json"
        idx = new_index(cache)
        t = time.perf_counter(); stats = idx.build(force=True); full.append(time.perf_counter() - t)
        chunks = stats["chunks"]
        t = time.perf_counter(); idx.build(); noop.append(time.perf_counter() - t)
        fresh = new_index(cache)
        t = time.perf_counter(); fresh.build(); cold.append(time.perf_counter() - t)
        t = time.perf_counter()
        for _ in range(50):
            idx.search(QUERY, k=10)
        search.append((time.perf_counter() - t) / 50 * 1000)

    # race: 4 threads search while the main thread forces 3 rebuilds
    errors: list[str] = []
    stop = threading.Event()

    def searcher():
        while not stop.is_set():
            try:
                idx.search("w17 w4242", k=5)
            except Exception as exc:  # noqa: BLE001
                errors.append(type(exc).__name__)

    threads = [threading.Thread(target=searcher) for _ in range(4)]
    for t in threads:
        t.start()
    for _ in range(3):
        idx.build(force=True)
    stop.set()
    for t in threads:
        t.join()

    med = statistics.median
    return {
        "chunks": chunks,
        "full_build_s_median": med(full), "full_build_s_runs": full,
        "noop_rebuild_s_median": med(noop), "noop_rebuild_s_runs": noop,
        "fresh_instance_from_cache_s_median": med(cold), "fresh_instance_from_cache_s_runs": cold,
        "search_ms_mean_median": med(search), "search_ms_mean_runs": search,
        "race": {"threads": 4, "forced_rebuilds": 3, "errors": len(errors),
                 "error_types": sorted(set(errors))},
    }


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        print(json.dumps(worker(sys.argv[2], sys.argv[3])))
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="git revision of the older code")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    corpus = Path(tempfile.mkdtemp())
    make_corpus(corpus)
    old_dir = Path(tempfile.mkdtemp())
    subprocess.run(f"git archive {args.old} | tar -x -C {old_dir}", shell=True, cwd=REPO, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    old_full = subprocess.run(["git", "rev-parse", args.old], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                                capture_output=True, text=True).stdout.strip())

    results = {}
    for label, code_dir in (("old", old_dir), ("current", REPO)):
        print(f"timing {label} ...", file=sys.stderr, flush=True)
        env = {**os.environ, "DOCS_CITE_EMBED_MODEL": "none"}
        proc = subprocess.run([sys.executable, __file__, "--worker", str(code_dir), str(corpus)],
                              capture_output=True, text=True, env=env, check=True)
        results[label] = json.loads(proc.stdout.strip().splitlines()[-1])

    out = {
        "script": "bench/perf.py",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"old": old_full, "current": head, "current_uncommitted_changes": dirty},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "machine": platform.machine(), "cpu_count": os.cpu_count()},
        "setup": {"synthetic_corpus": {"docs": DOCS, "lines_per_section": LINES, "vocabulary": VOCAB,
                                       "seed": SEED, "sections_per_doc": 2},
                  "repeats": REPEATS, "search": f"query {QUERY!r}, k=10, mean of 50 calls per repeat",
                  "retrieval": "lexical (BM25) only; no embedding model"},
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
