from __future__ import annotations

import numpy as np

from codeatlas.chunking import Chunk, build_chunks
from codeatlas.retrieval import (
    STOPWORDS,
    Index,
    apply_file_cap,
    embed_chunks,
    embedding_cache_key,
    rrf,
    split_identifier,
    tokenize,
)


# ---------------------------------------------------------------- tokenizer
def test_tokenize_removes_stopwords_and_stems():
    tokens = tokenize("How does it decide a function is too complicated?")
    assert not any(t in STOPWORDS for t in tokens)
    assert "how" not in tokens and "does" not in tokens
    assert "decid" in tokens and "function" in tokens and "complic" in tokens


def test_tokenize_matches_inflections():
    assert tokenize("retries") == tokenize("retry")
    assert tokenize("rate limiting") == tokenize("rate limit")


def test_identifier_splitting_keeps_whole_and_parts():
    assert split_identifier("RateLimitFilter") == ["ratelimitfilter", "rate", "limit", "filter"]
    assert split_identifier("set_relay") == ["set_relay", "set", "relay"]
    assert split_identifier("HTTPServer") == ["httpserver", "http", "server"]
    assert split_identifier("relay") == ["relay"]


def test_tokenize_query_reaches_camel_case_symbol():
    doc = tokenize("class RateLimitFilter: ...")
    query = tokenize("rate limit")
    assert set(query) <= set(doc)


# ---------------------------------------------------------------- fusion
def test_rrf_scores_and_tie_order():
    fused = rrf([[1, 2, 3], [3, 1, 4]], k=60)
    # 1: 1/61 + 1/62 ; 3: 1/63 + 1/61 ; 2: 1/62 ; 4: 1/63
    assert fused == [1, 3, 2, 4]


def test_rrf_ties_keep_first_appearance():
    assert rrf([[7, 8], [9, 10]], k=60) == [7, 9, 8, 10]


# ---------------------------------------------------------------- per-file cap
def test_cap_pushes_overflow_below_survivors_and_discards_nothing():
    paths = ["a", "a", "b", "a", "c", "b", "d"]
    ranked = list(range(len(paths)))
    capped = apply_file_cap(ranked, paths, per_file=1)
    assert capped == [0, 2, 4, 6, 1, 3, 5]
    assert sorted(capped) == ranked
    capped2 = apply_file_cap(ranked, paths, per_file=2)
    assert capped2 == [0, 1, 2, 4, 5, 6, 3]


def test_cap_overflow_reaches_the_top_k_when_few_files_remain():
    paths = ["a"] * 5 + ["b"] * 5
    ranked = list(range(10))
    top3 = apply_file_cap(ranked, paths, per_file=1)[:3]
    assert top3 == [0, 5, 1]


def test_cap_rejects_zero():
    import pytest

    with pytest.raises(ValueError):
        apply_file_cap([0], ["a"], per_file=0)


# ---------------------------------------------------------------- index
def _chunks(fixture_corpus, count):
    return build_chunks(fixture_corpus, "structural", count)


def test_strategies_return_k_distinct_chunks(fixture_corpus, count, embedder):
    index = Index.build(_chunks(fixture_corpus, count), embedder)
    for strategy in ("bm25", "dense", "hybrid", "hybrid+cap"):
        got = index.search("rate limit per client", strategy, k=5)
        assert len(got) == 5 and len(set(got)) == 5


def test_hybrid_is_rrf_of_bm25_and_dense(fixture_corpus, count, embedder):
    index = Index.build(_chunks(fixture_corpus, count), embedder)
    q = "how long between retry attempts"
    expected = rrf([index.bm25_rank(q, 60), index.dense_rank(q, 60)])[:10]
    assert index.hybrid_rank(q, 10) == expected


def test_capped_rank_matches_apply_file_cap(fixture_corpus, count, embedder):
    index = Index.build(_chunks(fixture_corpus, count), embedder)
    q = "retry backoff"
    expected = apply_file_cap(index.hybrid_rank(q, 60), index.paths, 1)[:10]
    assert index.capped_rank(q, 10, per_file=1) == expected
    top_paths = [index.paths[i] for i in index.capped_rank(q, 3, per_file=1)]
    assert len(set(top_paths)) == 3


def test_bm25_finds_lexical_match(fixture_corpus, count, embedder):
    index = Index.build(_chunks(fixture_corpus, count), embedder)
    top = index.bm25_rank("presign ttl seconds", 1)[0]
    assert index.chunks[top].path.endswith("config.yml")


# ---------------------------------------------------------------- cache
def _chunk(text: str, name: str = "n") -> Chunk:
    return Chunk("id", "p", "corpus/p/f.py", "function", name, 1, 1, text)


def test_cache_key_changes_with_content():
    a = embedding_cache_key([_chunk("alpha"), _chunk("beta")])
    b = embedding_cache_key([_chunk("alpha"), _chunk("gamma")])
    c = embedding_cache_key([_chunk("alpha"), _chunk("beta", name="other")])
    assert a != b and a != c


def test_stale_cache_is_not_reused(tmp_path, embedder):
    first = [_chunk("alpha one"), _chunk("beta two")]
    emb1 = embed_chunks(first, embedder, cache_dir=tmp_path, label="t")
    files = list(tmp_path.glob("emb_t.*.npy"))
    assert len(files) == 1
    reused = embed_chunks(first, embedder, cache_dir=tmp_path, label="t")
    assert np.array_equal(emb1, reused)
    changed = [_chunk("alpha one"), _chunk("gamma three"), _chunk("delta")]
    emb2 = embed_chunks(changed, embedder, cache_dir=tmp_path, label="t")
    assert emb2.shape[0] == 3
    assert len(list(tmp_path.glob("emb_t.*.npy"))) == 2
    assert np.array_equal(emb2, embedder.encode([c.payload() for c in changed]))
