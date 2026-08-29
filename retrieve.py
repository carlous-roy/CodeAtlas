"""
Four retrieval strategies over the same chunks, so they can be compared on the
same questions.

  bm25    lexical only. Fast, no model, and strong whenever the question happens
          to share vocabulary with the code.
  dense   embeddings only. Handles the questions where the asker does not know
          the codebase's words, which is most real questions.
  hybrid  both, fused with Reciprocal Rank Fusion. RRF is used rather than a
          weighted score blend because BM25 scores and cosine similarities are
          not on a comparable scale, and normalising them introduces a knob
          nobody can tune honestly on 36 questions.
  rerank  hybrid, then a cross-encoder re-scores the top 30. The cross-encoder
          reads query and chunk together instead of comparing two independently
          produced vectors, which is why it is worth the latency.
"""
import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def tokenize(text: str):
    """Split identifiers as well as words: a question asking about 'rate limit'
    should be able to match a symbol called RateLimitFilter."""
    out = []
    for t in TOKEN.findall(text):
        low = t.lower()
        out.append(low)
        parts = [p for p in re.split(r"_", t) if p]
        for p in parts:
            camel = re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+", p)
            if len(camel) > 1:
                out.extend(c.lower() for c in camel)
            elif p.lower() != low:
                out.append(p.lower())
    return out


class Index:
    def __init__(self, chunks, embedder=None, cache: Path | None = None):
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        self.emb = None
        if embedder is not None:
            if cache and cache.exists():
                self.emb = np.load(cache)
            else:
                # Prefix the declaration path onto the embedded text. The name is
                # signal: "RelayController.set_relay" carries meaning that the
                # body alone may not.
                payload = [f"{c['path']} :: {c['name']}\n{c['text'][:2000]}" for c in chunks]
                self.emb = embedder.encode(payload, batch_size=64,
                                           show_progress_bar=False,
                                           normalize_embeddings=True)
                if cache:
                    np.save(cache, self.emb)

    def bm25_rank(self, q, k):
        s = self.bm25.get_scores(tokenize(q))
        return list(np.argsort(-s)[:k])

    def dense_rank(self, q, k, embedder):
        qv = embedder.encode([q], normalize_embeddings=True)[0]
        s = self.emb @ qv
        return list(np.argsort(-s)[:k])

    def hybrid_rank(self, q, k, embedder, depth=60, rrf_k=60):
        a = self.bm25_rank(q, depth)
        b = self.dense_rank(q, depth, embedder)
        scores = {}
        for rank, i in enumerate(a):
            scores[i] = scores.get(i, 0) + 1.0 / (rrf_k + rank + 1)
        for rank, i in enumerate(b):
            scores[i] = scores.get(i, 0) + 1.0 / (rrf_k + rank + 1)
        return [i for i, _ in sorted(scores.items(), key=lambda kv: -kv[1])][:k]

    def rerank(self, q, k, embedder, cross, depth=30):
        cand = self.hybrid_rank(q, depth, embedder)
        pairs = [(q, self.texts[i][:1500]) for i in cand]
        s = cross.predict(pairs, show_progress_bar=False)
        order = np.argsort(-np.asarray(s))
        return [cand[j] for j in order][:k]


def load_chunks(strategy):
    return [json.loads(l) for l in open(f"index/chunks_{strategy}.jsonl", encoding="utf-8")]
