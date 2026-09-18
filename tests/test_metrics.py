from __future__ import annotations

import math

import pytest

from codeatlas.metrics import (
    bootstrap_ci,
    derived_seed,
    hit_rate_at_k,
    ndcg_at_k,
    paired_bootstrap,
    recall_at_k,
    reciprocal_rank,
    score_question,
    sign_test_p,
)

REL = {"a", "b", "c", "d"}


def test_hit_rate_is_not_recall():
    paths = ["x", "x", "y", "z", "a", "w", "q", "r", "s", "t"]
    assert hit_rate_at_k(paths, REL, 5) == 1.0
    assert recall_at_k(paths, REL, 5) == 0.25
    assert hit_rate_at_k(paths, REL, 3) == 0.0
    assert recall_at_k(paths, REL, 3) == 0.0


def test_recall_counts_distinct_files_once():
    paths = ["a", "a", "a", "b", "b"]
    assert recall_at_k(paths, REL, 5) == 0.5
    assert recall_at_k(paths, {"a"}, 1) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "y", "a"], REL) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "y"], REL) == 0.0
    assert reciprocal_rank(["a"], REL) == 1.0


def test_ndcg_normalises_by_the_labelled_set():
    # one of three relevant files at rank 1 is not a perfect ranking
    paths = ["a", "x", "x", "x", "x", "x", "x", "x", "x", "x"]
    ideal = sum(1 / math.log2(i + 1) for i in range(1, 4))
    assert ndcg_at_k(paths, {"a", "b", "c"}, 10) == pytest.approx(1.0 / ideal)
    assert ndcg_at_k(paths, {"a"}, 10) == 1.0


def test_ndcg_counts_a_file_once():
    repeated = ["a"] * 10
    once = ["a"] + ["x"] * 9
    rel = {"a", "b"}
    assert ndcg_at_k(repeated, rel, 10) == ndcg_at_k(once, rel, 10)
    assert ndcg_at_k(repeated, rel, 10) < 1.0


def test_ndcg_perfect_and_empty():
    assert ndcg_at_k(["a", "b", "c", "d"], REL, 10) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], REL, 10) == 0.0
    assert ndcg_at_k(["b", "a"], {"a", "b"}, 1) == pytest.approx(1.0)


def test_score_question_keys():
    scores = score_question(["a", "x", "b"], {"a", "b"})
    assert scores["hit_rate@1"] == 1.0
    assert scores["recall@1"] == 0.5
    assert scores["recall@3"] == 1.0
    assert scores["mrr"] == 1.0
    assert set(scores) == {
        "hit_rate@1", "hit_rate@3", "hit_rate@5", "hit_rate@10",
        "recall@1", "recall@3", "recall@5", "recall@10",
        "mrr", "ndcg@10",
    }  # fmt: skip


def test_bootstrap_is_deterministic_and_brackets_the_mean():
    values = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    lo, hi = bootstrap_ci(values, n_resamples=2000, seed=1)
    assert (lo, hi) == bootstrap_ci(values, n_resamples=2000, seed=1)
    assert lo < sum(values) / len(values) < hi
    assert bootstrap_ci([1.0, 1.0, 1.0], n_resamples=100, seed=3) == (1.0, 1.0)


def test_paired_bootstrap_counts_and_sign_test():
    a = [1, 1, 1, 1, 0, 0, 0, 0]
    b = [0, 0, 0, 0, 0, 0, 0, 0]
    stats = paired_bootstrap(a, b, n_resamples=2000, seed=2)
    assert stats["better"] == 4 and stats["worse"] == 0 and stats["same"] == 4
    assert stats["diff"] == pytest.approx(0.5)
    assert stats["sign_test_p"] == pytest.approx(sign_test_p(4, 0)) == pytest.approx(0.125)
    assert stats["ci95"][0] > 0.0


def test_sign_test_symmetry():
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(3, 3) == 1.0
    assert sign_test_p(5, 1) == sign_test_p(1, 5)


def test_derived_seed_is_stable_and_label_dependent():
    assert derived_seed(1, "x/mrr") == derived_seed(1, "x/mrr")
    assert derived_seed(1, "x/mrr") != derived_seed(1, "y/mrr")
