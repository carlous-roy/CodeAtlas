"""Retrieval metrics at file level, and the statistics reported with them.

Relevance is judged per file: a retrieved chunk is relevant when its source
file is one of the labelled answer files. The ranked list is a list of chunk
paths, so one file can appear several times; the metrics below count a file
once, at the rank of its first chunk.

``hit_rate@k``
    1 if any relevant file appears in the top ``k`` chunks, else 0. This is
    what the first version of this project called ``recall@k``.
``recall@k``
    Share of the labelled files that appear in the top ``k`` chunks.
``mrr``
    1 / rank of the first relevant chunk, 0 when none is retrieved.
``ndcg@k``
    Discounted cumulative gain with one binary gain per relevant file at the
    rank of its first chunk, normalised by the ideal gain of the labelled set
    (every labelled file at the top, up to ``k``).
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Iterable, Sequence

import numpy as np

K_LIST = (1, 3, 5, 10)
METRICS = tuple(f"hit_rate@{k}" for k in K_LIST) + tuple(f"recall@{k}" for k in K_LIST) + ("mrr", "ndcg@10")
BOOTSTRAP_RESAMPLES = 10_000
SEED = 20260917


def first_hit_rank(paths: Sequence[str], relevant: Iterable[str]) -> int | None:
    rel = set(relevant)
    for i, p in enumerate(paths, start=1):
        if p in rel:
            return i
    return None


def _first_ranks(paths: Sequence[str], relevant: Iterable[str]) -> dict[str, int]:
    """Rank of the first chunk of every relevant file that was retrieved."""
    rel = set(relevant)
    ranks: dict[str, int] = {}
    for i, p in enumerate(paths, start=1):
        if p in rel and p not in ranks:
            ranks[p] = i
    return ranks


def hit_rate_at_k(paths: Sequence[str], relevant: Iterable[str], k: int) -> float:
    first = first_hit_rank(paths, relevant)
    return 1.0 if first is not None and first <= k else 0.0


def recall_at_k(paths: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    found = {p for p in paths[:k] if p in rel}
    return len(found) / len(rel)


def reciprocal_rank(paths: Sequence[str], relevant: Iterable[str]) -> float:
    first = first_hit_rank(paths, relevant)
    return 1.0 / first if first is not None else 0.0


def ndcg_at_k(paths: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    ranks = _first_ranks(paths[:k], rel)
    dcg = sum(1.0 / math.log2(r + 1) for r in ranks.values())
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / idcg


def score_question(paths: Sequence[str], relevant: Iterable[str]) -> dict[str, float]:
    rel = set(relevant)
    out: dict[str, float] = {}
    for k in K_LIST:
        out[f"hit_rate@{k}"] = hit_rate_at_k(paths, rel, k)
    for k in K_LIST:
        out[f"recall@{k}"] = recall_at_k(paths, rel, k)
    out["mrr"] = reciprocal_rank(paths, rel)
    out["ndcg@10"] = ndcg_at_k(paths, rel, 10)
    return out


# ------------------------------------------------------------- statistics
def derived_seed(base: int, label: str) -> int:
    """A seed per statistic, so adding a configuration to the evaluation
    leaves every other interval unchanged."""
    return (base + zlib.crc32(label.encode("utf-8"))) % (2**32)


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean over questions."""
    arr = np.asarray(values, dtype=np.float64)
    n = arr.shape[0]
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return (float(lo), float(hi))


def sign_test_p(better: int, worse: int) -> float:
    """Exact two-sided sign test on the discordant questions."""
    m = better + worse
    if m == 0:
        return 1.0
    tail = sum(math.comb(m, i) for i in range(0, min(better, worse) + 1))
    return min(1.0, 2.0 * tail / 2**m)


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
    level: float = 0.95,
) -> dict[str, float | int | list[float]]:
    """Paired comparison of two systems on the same questions: mean of
    ``a - b`` with a percentile bootstrap interval, the count of questions on
    which ``a`` is better, worse or the same, and an exact sign test."""
    if len(a) != len(b):
        raise ValueError("paired comparison needs the same questions in the same order")
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    lo, hi = bootstrap_ci(diff, n_resamples, seed, level)
    better = int(np.sum(diff > 1e-12))
    worse = int(np.sum(diff < -1e-12))
    return {
        "diff": float(diff.mean()),
        "ci95": [lo, hi],
        "better": better,
        "worse": worse,
        "same": int(len(diff) - better - worse),
        "sign_test_p": sign_test_p(better, worse),
    }
