"""Render the README tables and docs/case-study.html from results/.

Numbers in the README live between marker comments::

    <!-- codeatlas:table:all -->
    ... generated ...
    <!-- /codeatlas:table:all -->

``render_all`` replaces every block from the JSON files in ``results/`` and
writes the case study page. With ``check=True`` it only reports whether the
files on disk match what it would write, which is what CI runs.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Callable
from pathlib import Path

from codeatlas.evaluation import SHIPPED_CHUNKING

CHUNKING_LABEL = {
    "structural_merged": "structural, merged",
    "structural": "structural, unmerged",
    "window": "window",
}
STRATEGY_LABEL = {
    "bm25": "BM25",
    "dense": "dense",
    "hybrid": "hybrid (RRF)",
    "hybrid+cap": "hybrid + per-file cap",
    "hybrid+rerank": "hybrid + reranker (text only)",
    "hybrid+rerank+prefix": "hybrid + reranker (prefixed)",
}
METRIC_LABEL = {
    "hit_rate@1": "hit rate@1",
    "hit_rate@3": "hit rate@3",
    "hit_rate@5": "hit rate@5",
    "hit_rate@10": "hit rate@10",
    "recall@1": "recall@1",
    "recall@3": "recall@3",
    "recall@5": "recall@5",
    "recall@10": "recall@10",
    "mrr": "MRR",
    "ndcg@10": "nDCG@10",
}
_BLOCK = re.compile(r"(<!-- codeatlas:table:(\w+) -->\n)(.*?)(<!-- /codeatlas:table:\2 -->)", re.S)


class Results:
    """The results directory, loaded once."""

    def __init__(self, results_dir: Path):
        self.dir = results_dir
        self.metrics = self._load("metrics.json")
        self.comparisons = self._load("comparisons.json")
        self.cap = self._load("cap_ablation.json")
        self.failures = self._load("failure_analysis.json")
        self.metadata = self._load("metadata.json")
        self.manifest = self._load_manifest()
        self.by_key = {c["key"]: c for c in self.metrics["configurations"]}
        self.n = self.metrics["n_questions"]
        per_file = self.metadata["settings"].get("per_file")
        self.shipped_key = f"{SHIPPED_CHUNKING}/hybrid+cap" if per_file is not None else f"{SHIPPED_CHUNKING}/hybrid"
        self.shipped = self.by_key[self.shipped_key]

    def _load(self, name: str) -> dict:
        with (self.dir / name).open(encoding="utf-8") as f:
            return json.load(f)

    def _load_manifest(self) -> dict:
        """The manifest next to the results, else the one the evaluation
        recorded, else an empty one (a run with --allow-unverified-corpus)."""
        candidates = [self.dir / "corpus_manifest.json"]
        recorded = (self.metadata.get("corpus") or {}).get("manifest")
        if recorded:
            candidates.append(Path(recorded))
        for p in candidates:
            if p.exists():
                with p.open(encoding="utf-8") as f:
                    return json.load(f)
        return {"projects": {}, "totals": {"files": 0, "lines": 0}, "files": []}

    def get(self, key: str) -> dict | None:
        """The metrics entry of a ``chunking/strategy`` key, or None if it was not scored."""
        return self.by_key.get(key)

    def comparison(self, a: str, b: str, metric: str) -> dict | None:
        """The paired comparison of ``a`` against ``b`` on ``metric``, if one was made."""
        for c in self.comparisons:
            if c["a"] == a and c["b"] == b and c["metric"] == metric:
                return c
        return None


# ------------------------------------------------------------- formatting
def decimals(metric: str) -> int:
    return 3 if metric in ("mrr", "ndcg@10") else 2


def fmt(value: float, metric: str) -> str:
    return f"{value:.{decimals(metric)}f}"


def fmt_ci(cell: dict, metric: str) -> str:
    d = decimals(metric)
    lo, hi = cell["ci95"]
    return f"{cell['value']:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def fmt_diff(c: dict) -> str:
    d = decimals(c["metric"])
    lo, hi = c["ci95"]
    return f"{c['diff']:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def config_label(key: str) -> str:
    chunking, strategy = key.split("/", 1)
    return f"{CHUNKING_LABEL[chunking]} / {STRATEGY_LABEL[strategy]}"


def md_table(header: list[str], rows: list[list[str]], align_first_left: bool = True) -> str:
    sep = ["---"] * len(header)
    if align_first_left:
        sep = [":---"] + ["---:"] * (len(header) - 1)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------- README blocks
def block_headline(r: Results) -> str:
    cols = ["hit_rate@1", "hit_rate@3", "hit_rate@5", "recall@5", "mrr", "ndcg@10"]
    row = [f"**{config_label(r.shipped_key)}**"] + [fmt_ci(r.shipped["metrics"][m], m) for m in cols]
    return md_table(["Configuration"] + [METRIC_LABEL[m] for m in cols], [row])


def block_all(r: Results) -> str:
    cols = ["hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10"]
    rows = []
    for c in r.metrics["configurations"]:
        bold = c["key"] == r.shipped_key
        cells = [CHUNKING_LABEL[c["chunking"]], STRATEGY_LABEL[c["strategy"]], str(c["n_chunks"])]
        cells += [fmt_ci(c["metrics"][m], m) for m in cols]
        rows.append([f"**{x}**" if bold else x for x in cells])
    return md_table(["Chunking", "Strategy", "Chunks"] + [METRIC_LABEL[m] for m in cols], rows)


def block_matched(r: Results) -> str:
    cols = ["hit_rate@1", "hit_rate@5", "mrr", "ndcg@10"]
    rows = []
    for strategy in ("hybrid", "dense"):
        for chunking in ("structural_merged", "window", "structural"):
            c = r.get(f"{chunking}/{strategy}")
            if c is None:
                continue
            rows.append(
                [CHUNKING_LABEL[chunking], STRATEGY_LABEL[strategy], str(c["n_chunks"])]
                + [fmt_ci(c["metrics"][m], m) for m in cols]
            )
    return md_table(["Chunking", "Strategy", "Chunks"] + [METRIC_LABEL[m] for m in cols], rows)


def block_reranker(r: Results) -> str:
    cols = ["hit_rate@1", "hit_rate@5", "mrr", "ndcg@10"]
    rows = []
    for chunking in ("structural_merged", "window", "structural"):
        for strategy in ("hybrid", "hybrid+rerank", "hybrid+rerank+prefix"):
            c = r.get(f"{chunking}/{strategy}")
            if c is None:
                continue
            rows.append(
                [CHUNKING_LABEL[chunking], STRATEGY_LABEL[strategy]] + [fmt_ci(c["metrics"][m], m) for m in cols]
            )
    return md_table(["Chunking", "Strategy"] + [METRIC_LABEL[m] for m in cols], rows)


def block_cap(r: Results) -> str:
    cap = r.cap
    rows = []
    for row in cap["rows"]:
        pf = row["per_file"]
        label = "no cap" if pf is None else f"cap {pf}"
        if pf == cap["chosen_per_file"]:
            label = f"**{label}** (chosen)"
        rows.append(
            [
                label,
                fmt(row["dev"]["mrr"], "mrr"),
                fmt(row["dev"]["hit_rate@5"], "hit_rate@5"),
                fmt(row["dev"]["recall@5"], "recall@5"),
                fmt(row["test"]["mrr"], "mrr"),
                fmt(row["test"]["hit_rate@5"], "hit_rate@5"),
                fmt(row["test"]["recall@5"], "recall@5"),
                fmt(row["all"]["mrr"], "mrr"),
                fmt(row["all"]["hit_rate@5"], "hit_rate@5"),
                fmt(row["all"]["recall@5"], "recall@5"),
            ]
        )
    header = [
        "Per-file cap",
        f"dev MRR (n={cap['dev_n']})",
        "dev hit rate@5",
        "dev recall@5",
        f"test MRR (n={cap['test_n']})",
        "test hit rate@5",
        "test recall@5",
        f"all MRR (n={r.n})",
        "all hit rate@5",
        "all recall@5",
    ]
    return md_table(header, rows)


def block_comparisons(r: Results) -> str:
    rows = []
    for c in r.comparisons:
        verdict = "within noise" if c["within_noise"] else "interval excludes zero"
        rows.append(
            [
                c["label"],
                METRIC_LABEL[c["metric"]],
                fmt(c["a_value"], c["metric"]),
                fmt(c["b_value"], c["metric"]),
                fmt_diff(c),
                f"{c['better']} / {c['worse']} / {c['same']}",
                f"{c['sign_test_p']:.3f}",
                verdict,
            ]
        )
    header = [
        "Comparison (A vs B)",
        "Metric",
        "A",
        "B",
        "A - B [95% CI]",
        "better / worse / same",
        "sign test p",
        "Reading",
    ]
    return md_table(header, rows)


def block_failures(r: Results) -> str:
    f = r.failures["configurations"][r.failures["shipped"]]
    rows = []
    for label, key in (
        ("share of the top 5 that is documentation", "doc_share_top5"),
        ("share of the top 5 from the wrong project", "wrong_project_share_top5"),
    ):
        d = f[key]
        lo, hi = d["diff_ci95"]
        rows.append(
            [label, f"{d['misses']:.2f}", f"{d['hits']:.2f}", f"{d['misses'] - d['hits']:+.2f} [{lo:+.2f}, {hi:+.2f}]"]
        )
    table = md_table(
        [
            f"{config_label(r.failures['shipped'])}",
            f"on misses (n={f['n_misses']})",
            f"on hits (n={f['n_hits']})",
            "difference [95% CI]",
        ],
        rows,
    )
    misses = ", ".join(f"`{m}`" for m in f["misses_at_5"]) or "none"
    return f"{table}\nQuestions missed at rank 5: {misses}.\n"


def block_corpus(r: Results) -> str:
    rows = []
    for name, info in r.manifest["projects"].items():
        repo = info.get("repository") or "none"
        commit = info.get("commit")
        rows.append(
            [
                f"`{name}`",
                repo,
                f"`{commit[:12]}`" if commit else "unpinned",
                str(info["files"]),
                f"{info['lines']:,}",
            ]
        )
    t = r.manifest["totals"]
    rows.append(["**total**", "", "", f"**{t['files']}**", f"**{t['lines']:,}**"])
    return md_table(["Project", "Repository", "Commit", "Files", "Lines"], rows)


def block_chunkings(r: Results) -> str:
    rows = []
    for chunking, info in r.metadata["chunkings"].items():
        rows.append(
            [
                CHUNKING_LABEL[chunking],
                str(info["n_chunks"]),
                str(info["n_files"]),
                str(info["median_lines"]),
                str(info["p95_lines"]),
                str(info["max_lines"]),
                str(info["median_payload_tokens"]),
                str(info["max_payload_tokens"]),
            ]
        )
    header = ["Chunking", "Chunks", "Files", "Median lines", "p95 lines", "Max lines", "Median tokens", "Max tokens"]
    return md_table(header, rows)


def block_metadata(r: Results) -> str:
    m = r.metadata
    models = m["models"]
    pkgs = m["packages"]
    lines = [
        f"- Generated: {m['generated_at']} with codeatlas {m['codeatlas_version']}"
        + (f" at commit `{m['codeatlas_commit'][:12]}`" if m.get("codeatlas_commit") else ""),
        f"- Python {m['python']}; torch {pkgs.get('torch')}, sentence-transformers {pkgs.get('sentence-transformers')}, "
        f"transformers {pkgs.get('transformers')}, tree-sitter {pkgs.get('tree-sitter')}, rank-bm25 {pkgs.get('rank-bm25')}, "
        f"snowballstemmer {pkgs.get('snowballstemmer')}",
        f"- Embedder: `{models['embedder']['name']}` at revision `{models['embedder']['revision'][:12]}` "
        f"({models['embedder']['max_tokens']}-token window)",
    ]
    if models.get("reranker"):
        lines.append(
            f"- Reranker: `{models['reranker']['name']}` at revision `{models['reranker']['revision'][:12]}` "
            f"({models['reranker']['max_tokens']}-token window)"
        )
    corpus = m["corpus"]
    lines.append(
        f"- Corpus manifest sha256 `{(corpus.get('manifest_sha256') or '')[:12]}`, verified: {corpus.get('verified')}"
    )
    s = m["settings"]
    lines.append(
        f"- Settings: top-{s['depth']} scored, RRF k={s['rrf_k']} over depth {s['fusion_depth']}, cap depth {s['cap_depth']}, "
        f"rerank depth {s['rerank_depth']}, per-file cap {s['per_file']}, payload budget {s['payload_budget_tokens']} tokens, "
        f"bootstrap {s['bootstrap_resamples']:,} resamples, seed {s['seed']}"
    )
    lines.append(f"- Run time: {m['duration_seconds']:.0f} s on CPU")
    return "\n".join(lines) + "\n"


BLOCKS: dict[str, Callable[[Results], str]] = {
    "headline": block_headline,
    "all": block_all,
    "matched": block_matched,
    "reranker": block_reranker,
    "cap": block_cap,
    "comparisons": block_comparisons,
    "failures": block_failures,
    "corpus": block_corpus,
    "chunkings": block_chunkings,
    "metadata": block_metadata,
}


def render_readme(text: str, r: Results) -> str:
    """The README text with every marked block replaced by its rendering."""

    def replace(m: re.Match[str]) -> str:
        name = m.group(2)
        if name not in BLOCKS:
            raise KeyError(f"unknown README block {name!r}")
        return f"{m.group(1)}{BLOCKS[name](r)}{m.group(4)}"

    out, n = _BLOCK.subn(replace, text)
    if n == 0:
        raise ValueError("README has no codeatlas table markers")
    return out


# ------------------------------------------------------------- case study
def render_case_study(r: Results) -> str:
    """The complete case study page."""
    from codeatlas.case_study import render

    return render(r)


def render_all(results_dir: Path, readme: Path, case_study: Path, check: bool = False) -> int:
    """Rewrite the README blocks and the case study from ``results_dir``, or
    with ``check`` only report whether they are current. Returns the exit code."""
    r = Results(results_dir)
    outputs = {readme: render_readme(readme.read_text(encoding="utf-8"), r), case_study: render_case_study(r)}
    stale = [p for p, content in outputs.items() if not p.exists() or p.read_text(encoding="utf-8") != content]
    if check:
        if stale:
            print("out of date: " + ", ".join(str(p) for p in stale))
            return 1
        print("rendered docs match results/")
        return 0
    for p, content in outputs.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        print(f"wrote {p}")
    return 0


def esc(s: object) -> str:
    """HTML-escape a value for the case study."""
    return html.escape(str(s), quote=True)
