"""Score every chunking and retrieval strategy on the golden questions and
write the results, with confidence intervals, to a results directory.

Files written:

``metrics.json``
    One entry per configuration with every metric on all questions (value and
    95% bootstrap interval) and on the dev and test halves.
``per_question.json``
    Per-question metric values for every configuration.
``retrievals.json``
    The top-10 chunks for every question and configuration, with hit flags.
``comparisons.json``
    Paired bootstrap comparisons between configurations.
``cap_ablation.json``
    Per-file cap values tried on the dev half, and the value chosen.
``failure_analysis.json``
    Misses of the shipped configuration and what fills their top 5.
``metadata.json``
    Date, versions, model revisions, corpus manifest hash and settings.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import random
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np

from codeatlas import __version__
from codeatlas.chunking import (
    CHUNKINGS,
    WINDOW_OVERLAP,
    Chunk,
    build_chunks,
    project_of,
    size_summary,
    write_chunks,
)
from codeatlas.corpus import load_manifest, manifest_hash, sha256_of, verify_manifest
from codeatlas.metrics import (
    BOOTSTRAP_RESAMPLES,
    METRICS,
    SEED,
    bootstrap_ci,
    derived_seed,
    first_hit_rank,
    paired_bootstrap,
    score_question,
)
from codeatlas.models import (
    EMBEDDER,
    RERANKER,
    TokenCounter,
    load_embedder,
    load_reranker,
    load_token_counter,
    payload_budget,
)
from codeatlas.retrieval import (
    CAP_DEPTH,
    FUSION_DEPTH,
    RERANK_DEPTH,
    RRF_K,
    Index,
)

log = logging.getLogger("codeatlas")

DEPTH = 10
DOC_SUFFIXES = (".md", ".txt")
CAP_CANDIDATES: tuple[int | None, ...] = (None, 3, 2, 1)
SHIPPED_CHUNKING = "structural_merged"

# Paired comparisons reported alongside the tables. (label, a, b, metrics)
COMPARISONS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        "structural boundaries vs windows, hybrid",
        "structural_merged/hybrid",
        "window/hybrid",
        ("hit_rate@1", "mrr", "ndcg@10"),
    ),
    ("structural boundaries vs windows, dense", "structural_merged/dense", "window/dense", ("hit_rate@1", "mrr")),
    ("merged vs unmerged structural, hybrid", "structural_merged/hybrid", "structural/hybrid", ("hit_rate@1", "mrr")),
    (
        "hybrid vs dense, merged",
        "structural_merged/hybrid",
        "structural_merged/dense",
        ("hit_rate@1", "hit_rate@5", "mrr", "ndcg@10"),
    ),
    ("hybrid vs dense, window", "window/hybrid", "window/dense", ("hit_rate@1", "mrr")),
    ("hybrid vs bm25, merged", "structural_merged/hybrid", "structural_merged/bm25", ("hit_rate@5", "mrr")),
    (
        "per-file cap vs none, merged hybrid",
        "structural_merged/hybrid+cap",
        "structural_merged/hybrid",
        ("hit_rate@3", "hit_rate@5", "recall@5", "mrr", "ndcg@10"),
    ),
    (
        "reranker (text only) vs hybrid, merged",
        "structural_merged/hybrid+rerank",
        "structural_merged/hybrid",
        ("hit_rate@1", "mrr"),
    ),
    (
        "reranker (prefixed) vs hybrid, merged",
        "structural_merged/hybrid+rerank+prefix",
        "structural_merged/hybrid",
        ("hit_rate@1", "mrr"),
    ),
    (
        "reranker prefixed vs text only, merged",
        "structural_merged/hybrid+rerank+prefix",
        "structural_merged/hybrid+rerank",
        ("hit_rate@1", "mrr"),
    ),
    ("reranker (text only) vs hybrid, window", "window/hybrid+rerank", "window/hybrid", ("hit_rate@1", "mrr")),
    ("reranker (prefixed) vs hybrid, window", "window/hybrid+rerank+prefix", "window/hybrid", ("hit_rate@1", "mrr")),
    (
        "reranker prefixed vs text only, window",
        "window/hybrid+rerank+prefix",
        "window/hybrid+rerank",
        ("hit_rate@1", "mrr"),
    ),
)


@dataclass(frozen=True)
class Question:
    """One golden question: its id, text, labelled answer files and difficulty label."""

    id: str
    q: str
    relevant: tuple[str, ...]
    difficulty: str


def load_golden(path: Path) -> tuple[dict, list[Question]]:
    """The protocol block and the questions of a golden file; ids must be unique."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    questions = [Question(q["id"], q["q"], tuple(q["relevant"]), q.get("difficulty", "")) for q in data["questions"]]
    ids = [q.id for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("golden set has duplicate question ids")
    return data.get("protocol", {}), questions


def make_split(questions: list[Question], seed: int = SEED) -> dict:
    """Provisional 50/50 dev/test split, stratified by project: within each
    project the ids are shuffled with ``seed`` and the first half goes to dev."""
    rng = random.Random(seed)
    by_project: dict[str, list[str]] = {}
    for q in questions:
        by_project.setdefault(project_of(q.relevant[0]), []).append(q.id)
    dev: list[str] = []
    test: list[str] = []
    for project in sorted(by_project):
        ids = sorted(by_project[project])
        rng.shuffle(ids)
        half = len(ids) // 2
        dev.extend(sorted(ids[:half]))
        test.extend(sorted(ids[half:]))
    return {
        "method": "stratified by project, shuffled with a fixed seed, first half dev",
        "seed": seed,
        "dev": dev,
        "test": test,
    }


def load_split(path: Path, questions: list[Question]) -> dict[str, list[str]]:
    """The dev and test id lists of a split file, which must partition ``questions``."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    ids = {q.id for q in questions}
    dev, test = list(data["dev"]), list(data["test"])
    if set(dev) & set(test):
        raise ValueError("dev and test halves overlap")
    if set(dev) | set(test) != ids:
        raise ValueError("split does not cover exactly the golden questions")
    return {"dev": dev, "test": test}


def _round(x: float) -> float:
    return round(float(x), 4)


def _installed(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _git_head(repo: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def _cap_strategy(index: Index, per_file: int | None) -> Callable[[str], list[int]]:
    if per_file is None:
        return lambda q: index.hybrid_rank(q, DEPTH)
    return lambda q: index.capped_rank(q, DEPTH, per_file=per_file)


def _strategies(index: Index, per_file: int | None, with_rerank: bool) -> dict[str, Callable[[str], list[int]]]:
    """The ranking function of every strategy evaluated on one index."""
    strategies: dict[str, Callable[[str], list[int]]] = {
        "bm25": lambda q: index.bm25_rank(q, DEPTH),
        "dense": lambda q: index.dense_rank(q, DEPTH),
        "hybrid": lambda q: index.hybrid_rank(q, DEPTH),
    }
    if per_file is not None:
        strategies["hybrid+cap"] = _cap_strategy(index, per_file)
    if with_rerank:
        strategies["hybrid+rerank"] = lambda q: index.rerank(q, DEPTH, prefixed=False)
        strategies["hybrid+rerank+prefix"] = lambda q: index.rerank(q, DEPTH, prefixed=True)
    return strategies


def _group_difference_ci(
    values: dict[str, float], group_a: list[str], group_b: list[str], n_resamples: int, seed: int
) -> list[float]:
    """Bootstrap interval of mean(group_a) - mean(group_b), resampling each group."""
    if not group_a or not group_b:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    a = np.asarray([values[i] for i in group_a])
    b = np.asarray([values[i] for i in group_b])
    diffs = a[rng.integers(0, len(a), size=(n_resamples, len(a)))].mean(axis=1) - b[
        rng.integers(0, len(b), size=(n_resamples, len(b)))
    ].mean(axis=1)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return [_round(lo), _round(hi)]


def _group_mean(values: dict[str, float], ids: list[str]) -> float:
    return _round(np.mean([values[i] for i in ids])) if ids else 0.0


@dataclass
class EvalSettings:
    """Inputs, outputs and knobs of one evaluation run. ``per_file`` is a cap
    value or ``"auto"`` to choose one on the dev half; ``manifest_path=None``
    skips the corpus check and records that in metadata.json."""

    corpus_dir: Path
    golden_path: Path
    split_path: Path
    out_dir: Path
    index_dir: Path
    manifest_path: Path | None
    chunkings: tuple[str, ...] = CHUNKINGS
    with_rerank: bool = True
    per_file: int | str = "auto"
    n_resamples: int = BOOTSTRAP_RESAMPLES
    seed: int = SEED
    repo_dir: Path | None = None


class Evaluation:
    """One run over every chunking and strategy in ``settings``. :meth:`run`
    checks the corpus, builds each index, chooses the per-file cap on the dev
    half of the shipped chunking, scores every strategy and writes the results
    files after each chunking."""

    def __init__(self, settings: EvalSettings):
        self.s = settings
        self.protocol, self.questions = load_golden(settings.golden_path)
        self.split = load_split(settings.split_path, self.questions)
        self.ids = [q.id for q in self.questions]
        self.results: list[dict] = []
        self.per_question: dict[str, list[dict]] = {}
        self.retrievals: dict[str, list[dict]] = {}
        self.chunking_info: dict[str, dict] = {}
        self.cap_ablation: dict | None = None
        self.chosen_per_file: int | None = None
        self.corpus_info: dict = {}
        self.started = time.time()

    # ---------------------------------------------------------- corpus
    def check_corpus(self) -> None:
        s = self.s
        if not s.corpus_dir.is_dir():
            raise FileNotFoundError(f"corpus directory {s.corpus_dir} does not exist")
        if s.manifest_path is None:
            log.warning("corpus manifest check skipped; results will say the corpus was not verified")
            self.corpus_info = {"manifest": None, "manifest_sha256": None, "verified": False}
            return
        manifest = load_manifest(s.manifest_path)
        problems = verify_manifest(s.corpus_dir, manifest)
        if problems:
            shown = "\n  ".join(problems[:20])
            more = f"\n  ... {len(problems) - 20} more" if len(problems) > 20 else ""
            raise SystemExit(
                f"corpus at {s.corpus_dir} does not match {s.manifest_path} "
                f"({len(problems)} problems):\n  {shown}{more}\n"
                "Run `codeatlas corpus fetch` to check out the pinned commits."
            )
        self.corpus_info = {
            "manifest": str(s.manifest_path),
            "manifest_sha256": manifest_hash(s.manifest_path),
            "verified": True,
            "projects": {name: info.get("commit") for name, info in manifest["projects"].items()},
            "files": manifest["totals"]["files"],
            "lines": manifest["totals"]["lines"],
        }
        log.info(
            "corpus matches manifest: %d files, %d lines", manifest["totals"]["files"], manifest["totals"]["lines"]
        )

    # ---------------------------------------------------------- scoring
    def _score(self, index: Index, fn: Callable[[str], list[int]], key: str) -> None:
        per_q: list[dict] = []
        rows: list[dict] = []
        for q in self.questions:
            idxs = fn(q.q)[:DEPTH]
            paths = [index.chunks[i].path for i in idxs]
            scores = score_question(paths, q.relevant)
            first = first_hit_rank(paths, q.relevant)
            per_q.append({"id": q.id, "first_hit_rank": first, **{m: _round(v) for m, v in scores.items()}})
            rel = set(q.relevant)
            rows.append(
                {
                    "id": q.id,
                    "q": q.q,
                    "relevant": list(q.relevant),
                    "retrieved": [
                        {
                            "path": index.chunks[i].path,
                            "name": index.chunks[i].name,
                            "kind": index.chunks[i].kind,
                            "lines": [index.chunks[i].start_line, index.chunks[i].end_line],
                            "hit": index.chunks[i].path in rel,
                        }
                        for i in idxs
                    ],
                }
            )
        self.per_question[key] = per_q
        self.retrievals[key] = rows

    def _aggregate(self, key: str, chunking: str, strategy: str, n_chunks: int, per_file: int | None) -> dict:
        per_q = self.per_question[key]
        by_id = {p["id"]: p for p in per_q}
        metrics: dict[str, dict] = {}
        for m in METRICS:
            values = [by_id[i][m] for i in self.ids]
            lo, hi = bootstrap_ci(values, self.s.n_resamples, derived_seed(self.s.seed, f"{key}/{m}"))
            metrics[m] = {"value": _round(np.mean(values)), "ci95": [_round(lo), _round(hi)]}
        splits = {}
        for name, ids in self.split.items():
            splits[name] = {m: _round(np.mean([by_id[i][m] for i in ids])) for m in METRICS}
            splits[name]["n"] = len(ids)
        misses = [i for i in self.ids if by_id[i]["hit_rate@5"] == 0.0]
        entry = {
            "key": key,
            "chunking": chunking,
            "strategy": strategy,
            "n_chunks": n_chunks,
            "per_file": per_file,
            "n_questions": len(self.ids),
            "metrics": metrics,
            "dev": splits["dev"],
            "test": splits["test"],
            "misses_at_5": misses,
        }
        return entry

    def _log_entry(self, entry: dict) -> None:
        m = entry["metrics"]
        log.info(
            "  %-22s H@1 %.2f  H@5 %.2f  R@5 %.2f  MRR %.3f [%.3f, %.3f]  nDCG %.3f",
            entry["strategy"],
            m["hit_rate@1"]["value"],
            m["hit_rate@5"]["value"],
            m["recall@5"]["value"],
            m["mrr"]["value"],
            m["mrr"]["ci95"][0],
            m["mrr"]["ci95"][1],
            m["ndcg@10"]["value"],
        )

    # ---------------------------------------------------------- cap choice
    def _choose_cap(self, index: Index, chunking: str) -> int | None:
        """Try every cap value on the dev half and keep the best by MRR, then
        recall@5, preferring no cap and then the larger (weaker) cap on ties."""
        if isinstance(self.s.per_file, int):
            self.chosen_per_file = self.s.per_file
            self.cap_ablation = {
                "chunking": chunking,
                "base_strategy": "hybrid",
                "rule": "fixed by --per-file",
                "candidates": [self.s.per_file],
                "chosen_per_file": self.s.per_file,
                "dev_n": len(self.split["dev"]),
                "test_n": len(self.split["test"]),
                "rows": [],
            }
            return self.s.per_file
        dev_ids = set(self.split["dev"])
        rows: list[dict] = []
        best_key: tuple[float, float] | None = None
        chosen: int | None = None
        for per_file in CAP_CANDIDATES:
            fn = _cap_strategy(index, per_file)
            scored = {q.id: score_question([index.chunks[i].path for i in fn(q.q)], q.relevant) for q in self.questions}
            row = {"per_file": per_file}
            for name, ids in (("dev", self.split["dev"]), ("test", self.split["test"]), ("all", self.ids)):
                row[name] = {
                    m: _round(np.mean([scored[i][m] for i in ids]))
                    for m in ("hit_rate@1", "hit_rate@3", "hit_rate@5", "recall@5", "mrr", "ndcg@10")
                }
            rows.append(row)
            key = (
                round(float(np.mean([scored[i]["mrr"] for i in dev_ids])), 6),
                round(float(np.mean([scored[i]["recall@5"] for i in dev_ids])), 6),
            )
            if best_key is None or key > best_key:
                best_key, chosen = key, per_file
        self.chosen_per_file = chosen
        self.cap_ablation = {
            "chunking": chunking,
            "base_strategy": "hybrid",
            "rule": "highest dev MRR, then dev recall@5; ties prefer no cap, then the larger cap",
            "candidates": list(CAP_CANDIDATES),
            "chosen_per_file": chosen,
            "dev_n": len(self.split["dev"]),
            "test_n": len(self.split["test"]),
            "rows": rows,
        }
        log.info("per-file cap chosen on the dev half of %s: %s", chunking, chosen)
        return chosen

    # ---------------------------------------------------------- main loop
    def run(self) -> None:
        """Score every configuration and write the results directory."""
        s = self.s
        self.check_corpus()
        s.out_dir.mkdir(parents=True, exist_ok=True)
        log.info("loading models")
        count_tokens = load_token_counter()
        embedder = load_embedder()
        reranker = load_reranker() if s.with_rerank else None
        budget = payload_budget()

        # The shipped chunking goes first so that the per-file cap is chosen on it.
        order = sorted(s.chunkings, key=lambda c: c != SHIPPED_CHUNKING)
        for chunking in order:
            chunks = build_chunks(s.corpus_dir, chunking, count_tokens)
            write_chunks(chunks, s.index_dir / f"chunks_{chunking}.jsonl")
            info = size_summary(chunks)
            info.update(self._payload_check(chunks, count_tokens, budget))
            self.chunking_info[chunking] = info
            log.info("%s: %s", chunking, info)
            index = Index.build(chunks, embedder, cache_dir=s.index_dir, label=chunking, reranker=reranker)

            if self.chosen_per_file is None and self.cap_ablation is None:
                self._choose_cap(index, chunking)

            strategies = _strategies(index, self.chosen_per_file, reranker is not None)

            for name, fn in strategies.items():
                key = f"{chunking}/{name}"
                self._score(index, fn, key)
                per_file = self.chosen_per_file if name == "hybrid+cap" else None
                entry = self._aggregate(key, chunking, name, len(chunks), per_file)
                self.results.append(entry)
                self._log_entry(entry)
            self.write()
        self.write()

    @staticmethod
    def _payload_check(chunks: list[Chunk], count_tokens: TokenCounter, budget: int) -> dict:
        """Count every payload with the real tokenizer: nothing may exceed the
        window, otherwise the chunker's accounting is wrong."""
        counts = [count_tokens(c.payload()) for c in chunks]
        over = sum(1 for n in counts if n > budget)
        if over:
            raise RuntimeError(f"{over} chunk payloads exceed the {budget}-token budget")
        return {
            "max_payload_tokens": max(counts) if counts else 0,
            "median_payload_tokens": int(np.median(counts)) if counts else 0,
            "payload_budget": budget,
            "n_over_budget": over,
        }

    # ---------------------------------------------------------- outputs
    def comparisons(self) -> list[dict]:
        """Every paired comparison in ``COMPARISONS`` whose two configurations
        were scored, plus the shipped configuration against the best other
        cell by MRR (BM25 alone excluded)."""
        out: list[dict] = []
        have = {r["key"]: r for r in self.results}
        pairs = list(COMPARISONS)
        shipped = f"{SHIPPED_CHUNKING}/hybrid+cap" if self.chosen_per_file is not None else f"{SHIPPED_CHUNKING}/hybrid"
        if shipped in have:
            others = [k for k in have if k != shipped and not k.endswith("bm25")]
            if others:
                best = max(others, key=lambda k: have[k]["metrics"]["mrr"]["value"])
                pairs.append(
                    (
                        "shipped configuration vs best other cell by MRR",
                        shipped,
                        best,
                        ("hit_rate@1", "hit_rate@5", "mrr"),
                    )
                )
        for label, a, b, metrics in pairs:
            if a not in have or b not in have:
                continue
            pa = {p["id"]: p for p in self.per_question[a]}
            pb = {p["id"]: p for p in self.per_question[b]}
            for m in metrics:
                va = [pa[i][m] for i in self.ids]
                vb = [pb[i][m] for i in self.ids]
                stats = paired_bootstrap(va, vb, self.s.n_resamples, derived_seed(self.s.seed, f"{a}|{b}|{m}"))
                out.append(
                    {
                        "label": label,
                        "a": a,
                        "b": b,
                        "metric": m,
                        "a_value": _round(np.mean(va)),
                        "b_value": _round(np.mean(vb)),
                        "diff": _round(stats["diff"]),
                        "ci95": [_round(stats["ci95"][0]), _round(stats["ci95"][1])],
                        "better": stats["better"],
                        "worse": stats["worse"],
                        "same": stats["same"],
                        "sign_test_p": _round(stats["sign_test_p"]),
                        "within_noise": stats["ci95"][0] <= 0.0 <= stats["ci95"][1],
                    }
                )
        return out

    def failure_analysis(self) -> dict:
        """What fills the top 5 on the questions the shipped configuration
        misses, against the questions it hits."""
        shipped = f"{SHIPPED_CHUNKING}/hybrid+cap" if self.chosen_per_file is not None else f"{SHIPPED_CHUNKING}/hybrid"
        out: dict = {"configurations": {}}
        for key in (shipped, f"{SHIPPED_CHUNKING}/hybrid"):
            if key not in self.retrievals:
                continue
            rows = self.retrievals[key]
            per = {p["id"]: p for p in self.per_question[key]}
            doc_share: dict[str, float] = {}
            wrong_project: dict[str, float] = {}
            for r in rows:
                top5 = r["retrieved"][:5]
                doc_share[r["id"]] = sum(1 for x in top5 if x["path"].endswith(DOC_SUFFIXES)) / 5
                want = {project_of(p) for p in r["relevant"]}
                wrong_project[r["id"]] = sum(1 for x in top5 if project_of(x["path"]) not in want) / 5
            misses = [i for i in self.ids if per[i]["hit_rate@5"] == 0.0]
            hits = [i for i in self.ids if per[i]["hit_rate@5"] == 1.0]
            seed = derived_seed(self.s.seed, f"{key}/failure")
            out["configurations"][key] = {
                "misses_at_5": misses,
                "n_misses": len(misses),
                "n_hits": len(hits),
                "doc_share_top5": {
                    "misses": _group_mean(doc_share, misses),
                    "hits": _group_mean(doc_share, hits),
                    "all": _group_mean(doc_share, self.ids),
                    "diff_ci95": _group_difference_ci(doc_share, misses, hits, self.s.n_resamples, seed),
                },
                "wrong_project_share_top5": {
                    "misses": _group_mean(wrong_project, misses),
                    "hits": _group_mean(wrong_project, hits),
                    "all": _group_mean(wrong_project, self.ids),
                    "diff_ci95": _group_difference_ci(wrong_project, misses, hits, self.s.n_resamples, seed + 1),
                },
                "misses_detail": [
                    {
                        "id": r["id"],
                        "q": r["q"],
                        "relevant": r["relevant"],
                        "top5": [f"{x['path']} :: {x['name']}" for x in r["retrieved"][:5]],
                    }
                    for r in rows
                    if r["id"] in misses
                ],
            }
        out["shipped"] = shipped
        return out

    def metadata(self) -> dict:
        """Provenance of the run: date, versions, model pins, corpus and settings."""
        s = self.s
        packages = {}
        for name in (
            "numpy",
            "torch",
            "sentence-transformers",
            "transformers",
            "tokenizers",
            "tree-sitter",
            "tree-sitter-python",
            "tree-sitter-java",
            "rank-bm25",
            "snowballstemmer",
        ):
            packages[name] = _installed(name)
        return {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "codeatlas_version": __version__,
            "codeatlas_commit": _git_head(s.repo_dir) if s.repo_dir else None,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": packages,
            "models": {
                "embedder": {"name": EMBEDDER.name, "revision": EMBEDDER.revision, "max_tokens": EMBEDDER.max_tokens},
                "reranker": {"name": RERANKER.name, "revision": RERANKER.revision, "max_tokens": RERANKER.max_tokens}
                if s.with_rerank
                else None,
            },
            "corpus": self.corpus_info,
            "questions": {
                "golden": str(s.golden_path),
                "golden_sha256": sha256_of(s.golden_path),
                "n": len(self.ids),
                "split": str(s.split_path),
                "dev_n": len(self.split["dev"]),
                "test_n": len(self.split["test"]),
            },
            "settings": {
                "depth": DEPTH,
                "fusion_depth": FUSION_DEPTH,
                "rrf_k": RRF_K,
                "cap_depth": CAP_DEPTH,
                "rerank_depth": RERANK_DEPTH,
                "per_file": self.chosen_per_file,
                "payload_budget_tokens": payload_budget(),
                "window_overlap": WINDOW_OVERLAP,
                "bootstrap_resamples": s.n_resamples,
                "seed": s.seed,
            },
            "chunkings": self.chunking_info,
            "duration_seconds": round(time.time() - self.started, 1),
        }

    def write(self) -> None:
        """Write every results file from what has been scored so far."""
        out = self.s.out_dir
        _dump(out / "metrics.json", {"n_questions": len(self.ids), "configurations": self.results})
        _dump(out / "per_question.json", self.per_question)
        _dump(out / "retrievals.json", self.retrievals)
        _dump(out / "comparisons.json", self.comparisons())
        if self.cap_ablation is not None:
            _dump(out / "cap_ablation.json", self.cap_ablation)
        _dump(out / "failure_analysis.json", self.failure_analysis())
        _dump(out / "metadata.json", self.metadata())


def _dump(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


VOLATILE_METADATA = ("generated_at", "duration_seconds", "codeatlas_commit")


def results_digest(out_dir: Path, ignore_volatile: bool = True) -> dict[str, str]:
    """sha256 of every results file. ``metadata.json`` is hashed without the
    fields that change between two runs of the same code on the same machine
    (timestamp, run time, commit the run was made at), so two runs compare."""
    digest: dict[str, str] = {}
    for p in sorted(out_dir.glob("*.json")):
        data = p.read_bytes()
        if ignore_volatile and p.name == "metadata.json":
            meta = json.loads(data)
            for key in VOLATILE_METADATA:
                meta.pop(key, None)
            data = json.dumps(meta, sort_keys=True).encode("utf-8")
        digest[p.name] = hashlib.sha256(data).hexdigest()
    return digest
