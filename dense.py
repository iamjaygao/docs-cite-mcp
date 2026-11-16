"""
Dense embeddings for hybrid retrieval: optional, local, CPU-only.

Nothing leaves the machine. Models run through fastembed (ONNX Runtime, no
PyTorch) and are downloaded once into DOCS_CITE_MODEL_DIR. Without fastembed
installed, the index stays lexical (BM25) and says so in every search result.

Vectors are rounded to float16 as soon as they are computed. The on-disk cache
stores float16, so a freshly built index and one loaded from cache hold
bit-identical vectors and rank identically across restarts.
"""

from __future__ import annotations

import base64
import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

Kind = Literal["query", "passage"]

# Chosen on the 40-query gold set: hybrid with bge-small raised chunk-level NDCG@5
# 0.640 -> 0.740 (p = 0.002); the multilingual MiniLM gained less (+0.061) and,
# used alone, nothing. MiniLM remains the choice for corpora that are not English.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
MODEL_DIR = Path(os.environ.get("DOCS_CITE_MODEL_DIR", Path.home() / ".cache" / "docs-cite-mcp" / "models"))


@dataclass(frozen=True)
class ModelSpec:
    # Retrieval models are trained with these prefixes; leaving them off costs accuracy.
    query_prefix: str = ""
    passage_prefix: str = ""


MODELS = {
    "BAAI/bge-small-en-v1.5": ModelSpec(
        query_prefix="Represent this sentence for searching relevant passages: "),
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": ModelSpec(),
    "intfloat/multilingual-e5-large": ModelSpec(query_prefix="query: ", passage_prefix="passage: "),
}


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str], kind: Kind) -> np.ndarray:
        """Unit-length float16-rounded vectors, one row per text."""
        ...


def _normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms == 0, 1, norms)
    return vectors.astype(np.float16).astype(np.float32)   # what the cache will hold


class FastEmbedder:
    """A fastembed model, loaded on first use so importing this module is cheap."""

    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: Path = MODEL_DIR):
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}; choose one of {sorted(MODELS)}")
        from fastembed import TextEmbedding           # ImportError -> caller falls back to BM25
        self.name = model
        self.spec = MODELS[model]
        self._cache_dir = cache_dir
        self._model_cls = TextEmbedding
        self._model = None
        self._load_lock = threading.Lock()     # tool handlers run on worker threads
        self.dim = next(m["dim"] for m in TextEmbedding.list_supported_models() if m["model"] == model)

    def _load(self):
        with self._load_lock:
            if self._model is None:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                self._model = self._model_cls(self.name, cache_dir=str(self._cache_dir))
            return self._model

    def embed(self, texts: list[str], kind: Kind) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prefix = self.spec.query_prefix if kind == "query" else self.spec.passage_prefix
        return _normalise(list(self._load().embed([prefix + t for t in texts], batch_size=64)))


class HashEmbedder:
    """
    Deterministic stand-in for tests: hashed bag of words. It exercises fusion,
    caching and freshness without downloading a model; it is not a model.
    """

    def __init__(self, dim: int = 64):
        self.name = f"hash-{dim}"
        self.dim = dim

    def embed(self, texts: list[str], kind: Kind) -> np.ndarray:
        from docs_index import tokenize
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "little")
                out[row, h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        return _normalise(out)


def default_embedder() -> Embedder | None:
    """
    The embedder the server uses: DOCS_CITE_EMBED_MODEL, or DEFAULT_MODEL.
    Returns None (lexical only) if it is set to "none" or fastembed is missing.
    """
    model = os.environ.get("DOCS_CITE_EMBED_MODEL", DEFAULT_MODEL)
    if model.lower() == "none":
        return None
    try:
        return FastEmbedder(model)
    except ImportError:
        return None


def pack(vectors: np.ndarray) -> str:
    return base64.b64encode(np.asarray(vectors, dtype=np.float16).tobytes()).decode("ascii")


def unpack(data: str, dim: int) -> np.ndarray:
    return np.frombuffer(base64.b64decode(data), dtype=np.float16).reshape(-1, dim).astype(np.float32)
