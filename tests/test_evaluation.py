"""The evaluation loop end to end on the fixture corpus, with the stand-in
token counter and embedder from conftest and a stand-in cross-encoder, so
the run needs no model download."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import codeatlas.evaluation as evaluation
from codeatlas.chunking import CHUNKINGS
from codeatlas.evaluation import EvalSettings, Evaluation, results_digest
from codeatlas.metrics import METRICS
from codeatlas.render import BLOCKS, render_all

FIXTURES = Path(__file__).parent / "fixtures"
BASE = ("bm25", "dense", "hybrid")
RERANK = ("hybrid+rerank", "hybrid+rerank+prefix")


class OverlapReranker:
    """Stand-in cross-encoder: a pair scores the number of words the query and the text share."""

    def predict(self, pairs, show_progress_bar=False):
        return np.asarray([len(set(q.lower().split()) & set(t.lower().split())) for q, t in pairs], dtype=np.float32)


@pytest.fixture
def stand_in_models(monkeypatch, count, embedder):
    monkeypatch.setattr(evaluation, "load_token_counter", lambda: count)
    monkeypatch.setattr(evaluation, "load_embedder", lambda: embedder)
    monkeypatch.setattr(evaluation, "load_reranker", lambda: OverlapReranker())


def settings(out: Path, **overrides) -> EvalSettings:
    fields = dict(
        corpus_dir=FIXTURES / "corpus",
        golden_path=FIXTURES / "golden.json",
        split_path=FIXTURES / "split.json",
        out_dir=out / "results",
        index_dir=out / "index",
        manifest_path=FIXTURES / "corpus_manifest.json",
        n_resamples=50,
    )
    fields.update(overrides)
    return EvalSettings(**fields)


def load(out: Path, name: str) -> dict:
    return json.loads((out / "results" / name).read_text(encoding="utf-8"))


def render_from(out: Path) -> tuple[str, str]:
    """Render every README block and the case study from a results directory."""
    readme = out / "README.md"
    readme.write_text("".join(f"<!-- codeatlas:table:{b} -->\n<!-- /codeatlas:table:{b} -->\n" for b in BLOCKS))
    page = out / "case-study.html"
    assert render_all(out / "results", readme, page) == 0
    return readme.read_text(encoding="utf-8"), page.read_text(encoding="utf-8")


def test_run_scores_every_configuration_and_writes_consistent_results(tmp_path, stand_in_models):
    Evaluation(settings(tmp_path)).run()
    metrics = load(tmp_path, "metrics.json")
    cap = load(tmp_path, "cap_ablation.json")
    assert metrics["n_questions"] == 6
    assert cap["chunking"] == "structural_merged"
    assert [row["per_file"] for row in cap["rows"]] == [None, 3, 2, 1]
    chosen = cap["chosen_per_file"]
    strategies = BASE + (("hybrid+cap",) if chosen is not None else ()) + RERANK
    keys = [c["key"] for c in metrics["configurations"]]
    assert set(keys) == {f"{c}/{s}" for c in CHUNKINGS for s in strategies}
    assert keys[0].startswith("structural_merged/")
    for c in metrics["configurations"]:
        assert set(c["metrics"]) == set(METRICS)
        for cell in c["metrics"].values():
            lo, hi = cell["ci95"]
            assert 0.0 <= lo <= cell["value"] <= hi <= 1.0
        assert c["metrics"]["hit_rate@5"]["value"] >= c["metrics"]["recall@5"]["value"]
        assert c["per_file"] == (chosen if c["strategy"] == "hybrid+cap" else None)
        assert c["dev"]["n"] == 3 and c["test"]["n"] == 3

    per_question = load(tmp_path, "per_question.json")
    retrievals = load(tmp_path, "retrievals.json")
    for key in keys:
        assert [p["id"] for p in per_question[key]] == [f"fx0{i}" for i in range(1, 7)]
        for row, scores in zip(retrievals[key], per_question[key], strict=True):
            assert 1 <= len(row["retrieved"]) <= 10
            hits = [x["path"] in set(row["relevant"]) for x in row["retrieved"]]
            assert [x["hit"] for x in row["retrieved"]] == hits
            assert scores["hit_rate@10"] == (1.0 if any(hits) else 0.0)

    comparisons = load(tmp_path, "comparisons.json")
    assert {c["label"] for c in comparisons} >= {"hybrid vs bm25, merged", "reranker (text only) vs hybrid, merged"}
    for c in comparisons:
        assert c["a"] in keys and c["b"] in keys
        assert c["better"] + c["worse"] + c["same"] == 6
        assert c["within_noise"] == (c["ci95"][0] <= 0.0 <= c["ci95"][1])

    failures = load(tmp_path, "failure_analysis.json")
    shipped = failures["shipped"]
    assert shipped == ("structural_merged/hybrid+cap" if chosen is not None else "structural_merged/hybrid")
    entry = next(c for c in metrics["configurations"] if c["key"] == shipped)
    assert failures["configurations"][shipped]["misses_at_5"] == entry["misses_at_5"]

    metadata = load(tmp_path, "metadata.json")
    assert metadata["corpus"]["verified"] is True
    assert metadata["corpus"]["files"] == 5 and metadata["corpus"]["projects"] == {"demo": None}
    assert metadata["settings"]["per_file"] == chosen
    assert set(metadata["chunkings"]) == set(CHUNKINGS)
    assert all(info["n_over_budget"] == 0 for info in metadata["chunkings"].values())


def test_two_runs_produce_the_same_digest(tmp_path, stand_in_models):
    Evaluation(settings(tmp_path / "a")).run()
    Evaluation(settings(tmp_path / "b")).run()
    first = results_digest(tmp_path / "a" / "results")
    assert set(first) >= {"metrics.json", "comparisons.json", "metadata.json"}
    assert first == results_digest(tmp_path / "b" / "results")


def test_docs_render_from_a_run_whose_manifest_is_not_in_the_results_directory(tmp_path, stand_in_models):
    Evaluation(settings(tmp_path)).run()
    readme, page = render_from(tmp_path)
    assert "| `demo` | none | unpinned | 5 | 125 |" in readme
    assert "corpus commits demo unpinned." in page
    assert "None" not in page


def test_run_refuses_a_corpus_that_differs_from_the_manifest(tmp_path, stand_in_models):
    corpus = tmp_path / "corpus"
    shutil.copytree(FIXTURES / "corpus", corpus)
    (corpus / "demo" / "config.yml").write_text("queue:\n  name: other\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        Evaluation(settings(tmp_path, corpus_dir=corpus)).run()
    assert "content differs: demo/config.yml" in str(exc.value)
    assert not (tmp_path / "results").exists()


def test_fixed_cap_without_the_reranker(tmp_path, stand_in_models):
    Evaluation(settings(tmp_path, per_file=1, with_rerank=False)).run()
    metrics = load(tmp_path, "metrics.json")
    keys = {c["key"] for c in metrics["configurations"]}
    assert keys == {f"{c}/{s}" for c in CHUNKINGS for s in (*BASE, "hybrid+cap")}
    capped = next(c for c in metrics["configurations"] if c["key"] == "structural_merged/hybrid+cap")
    assert capped["per_file"] == 1
    cap = load(tmp_path, "cap_ablation.json")
    assert cap["chosen_per_file"] == 1 and cap["rows"] == [] and cap["dev_n"] == 3
    assert load(tmp_path, "metadata.json")["models"]["reranker"] is None
    readme, page = render_from(tmp_path)
    assert "structural, merged / hybrid + per-file cap" in readme
    assert "The reranker was not run in this evaluation." in page
