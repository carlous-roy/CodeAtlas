"""Retrieval strategies over one set of chunks.

``bm25``
    Lexical only, over the chunk text.
``dense``
    Cosine similarity of embeddings. Each chunk is embedded as
    ``path :: name`` followed by its text.
``hybrid``
    BM25 and dense rankings fused with Reciprocal Rank Fusion.
``hybrid+cap``
    Hybrid, then a per-file cap: no file contributes more than ``per_file``
    chunks to the head of the list. Chunks over the cap are pushed below every
    survivor, in their original order, never discarded.
``hybrid+rerank``
    Hybrid, then a cross-encoder re-scores the top ``depth`` candidates. It
    reads the chunk text alone (``prefixed=False``) or the same ``path :: name``
    payload the embedder sees (``prefixed=True``).
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import snowballstemmer
from rank_bm25 import BM25Okapi

from codeatlas.chunking import Chunk
from codeatlas.models import EMBEDDER

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer

STRATEGIES = ("bm25", "dense", "hybrid", "hybrid+cap", "hybrid+rerank", "hybrid+rerank+prefix")

RRF_K = 60
FUSION_DEPTH = 60
CAP_DEPTH = 60
RERANK_DEPTH = 30
DEFAULT_PER_FILE = 1

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+")

# English function words. Questions are written in natural language, and
# without this list "how", "does" and "it" carry positive weight against code.
STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at back be
    because been before being below between both but by can could did do does
    doing done down during each either else ever every few for from further get
    gets got had has have having he her here hers herself him himself his how i
    if in into is it its itself just let like may me might more most much must
    my myself no nor not now of off on once one only onto or other ought our ours
    ourselves out over own per s same shall she should so some such t than that
    the their theirs them themselves then there these they this those through to
    too under until up upon us very was we were what when where whether which
    while who whom whose why will with within without would you your yours
    yourself yourselves
    """.split()
)

_stemmer = snowballstemmer.stemmer("english")


def split_identifier(token: str) -> list[str]:
    """``RateLimitFilter`` -> ``['ratelimitfilter', 'rate', 'limit', 'filter']``.
    The whole identifier is kept so an exact mention still matches."""
    low = token.lower()
    out = [low]
    for part in token.split("_"):
        if not part:
            continue
        pieces = _CAMEL.findall(part)
        if len(pieces) > 1:
            out.extend(p.lower() for p in pieces)
        elif part.lower() != low:
            out.append(part.lower())
    return out


def tokenize(text: str) -> list[str]:
    """Words and identifier parts, lower-cased, without stopwords, stemmed."""
    words = [w for t in _TOKEN.findall(text) for w in split_identifier(t)]
    return _stemmer.stemWords([w for w in words if w not in STOPWORDS])


def rrf(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[int]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank).
    Ties keep the order of first appearance, so the result is deterministic."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: -scores[d])


def apply_file_cap(ranked: Sequence[int], paths: Sequence[str], per_file: int) -> list[int]:
    """Keep the first ``per_file`` chunks of every file in rank order and move
    the rest below them, in their original order. Nothing is discarded, so
    ``len(result) == len(ranked)``."""
    if per_file < 1:
        raise ValueError("per_file must be at least 1")
    seen: Counter[str] = Counter()
    keep: list[int] = []
    overflow: list[int] = []
    for doc in ranked:
        path = paths[doc]
        if seen[path] < per_file:
            seen[path] += 1
            keep.append(doc)
        else:
            overflow.append(doc)
    return keep + overflow


def _argsort_desc(scores: np.ndarray) -> np.ndarray:
    # Stable so that equal scores keep index order across runs and platforms.
    return np.argsort(-scores, kind="stable")


def embedding_cache_key(chunks: Sequence[Chunk]) -> str:
    """Hash of every embedded payload plus the model pin. A changed chunk
    file or a changed model gives a different key, so a stale cache can never
    be reused."""
    h = hashlib.sha256()
    h.update(f"{EMBEDDER.name}@{EMBEDDER.revision}\n".encode())
    for c in chunks:
        h.update(c.payload().encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def embed_chunks(
    chunks: Sequence[Chunk],
    embedder: SentenceTransformer,
    cache_dir: Path | None = None,
    label: str = "chunks",
) -> np.ndarray:
    """Normalised embeddings of every chunk payload, cached under ``cache_dir``
    by :func:`embedding_cache_key`."""
    cache = None
    if cache_dir is not None:
        cache = cache_dir / f"emb_{label}.{embedding_cache_key(chunks)}.npy"
        if cache.exists():
            emb = np.load(cache)
            if emb.shape[0] == len(chunks):
                return emb
    payload = [c.payload() for c in chunks]
    emb = embedder.encode(payload, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    emb = np.asarray(emb, dtype=np.float32)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb)
    return emb


class Index:
    """BM25 and dense indexes over one chunking, with the fused strategies."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: SentenceTransformer | None = None,
        embeddings: np.ndarray | None = None,
        reranker: CrossEncoder | None = None,
    ):
        self.chunks = list(chunks)
        self.paths = [c.path for c in self.chunks]
        self.texts = [c.text for c in self.chunks]
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        self.embedder = embedder
        self.emb = embeddings
        self.reranker = reranker
        self._query_vectors: dict[str, np.ndarray] = {}
        if self.emb is not None and self.emb.shape[0] != len(self.chunks):
            raise ValueError("embeddings do not match the chunk list")

    @classmethod
    def build(
        cls,
        chunks: Sequence[Chunk],
        embedder: SentenceTransformer,
        cache_dir: Path | None = None,
        label: str = "chunks",
        reranker: CrossEncoder | None = None,
    ) -> Index:
        emb = embed_chunks(chunks, embedder, cache_dir, label)
        return cls(chunks, embedder=embedder, embeddings=emb, reranker=reranker)

    # ---------------------------------------------------------- rankings
    def bm25_rank(self, query: str, k: int) -> list[int]:
        scores = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float64)
        return [int(i) for i in _argsort_desc(scores)[:k]]

    def dense_rank(self, query: str, k: int) -> list[int]:
        if self.emb is None or self.embedder is None:
            raise RuntimeError("dense retrieval needs an embedder and embeddings")
        qv = self._query_vectors.get(query)
        if qv is None:
            qv = np.asarray(self.embedder.encode([query], normalize_embeddings=True)[0], dtype=np.float32)
            self._query_vectors[query] = qv
        scores = self.emb @ qv
        return [int(i) for i in _argsort_desc(scores)[:k]]

    def hybrid_rank(self, query: str, k: int, depth: int = FUSION_DEPTH, rrf_k: int = RRF_K) -> list[int]:
        fused = rrf([self.bm25_rank(query, depth), self.dense_rank(query, depth)], k=rrf_k)
        return fused[:k]

    def capped_rank(self, query: str, k: int, per_file: int = DEFAULT_PER_FILE, depth: int = CAP_DEPTH) -> list[int]:
        ranked = self.hybrid_rank(query, depth)
        return apply_file_cap(ranked, self.paths, per_file)[:k]

    def rerank(self, query: str, k: int, prefixed: bool, depth: int = RERANK_DEPTH) -> list[int]:
        if self.reranker is None:
            raise RuntimeError("reranking needs a cross-encoder")
        candidates = self.hybrid_rank(query, depth)
        pairs = [(query, self.chunks[i].payload() if prefixed else self.texts[i]) for i in candidates]
        scores = np.asarray(self.reranker.predict(pairs, show_progress_bar=False), dtype=np.float64)
        return [candidates[int(j)] for j in _argsort_desc(scores)[:k]]

    def search(self, query: str, strategy: str, k: int = 10, per_file: int = DEFAULT_PER_FILE) -> list[int]:
        if strategy == "bm25":
            return self.bm25_rank(query, k)
        if strategy == "dense":
            return self.dense_rank(query, k)
        if strategy == "hybrid":
            return self.hybrid_rank(query, k)
        if strategy == "hybrid+cap":
            return self.capped_rank(query, k, per_file=per_file)
        if strategy == "hybrid+rerank":
            return self.rerank(query, k, prefixed=False)
        if strategy == "hybrid+rerank+prefix":
            return self.rerank(query, k, prefixed=True)
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
