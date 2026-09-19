"""The case study page, rendered from results/ by ``codeatlas render-docs``.

Everything numeric on the page comes from the JSON files; the prose around
the numbers is written here, and the sentences that state a direction or a
verdict are built from the paired comparisons so that they cannot disagree
with the data.
"""

from __future__ import annotations

from codeatlas.render import (
    CHUNKING_LABEL,
    METRIC_LABEL,
    STRATEGY_LABEL,
    Results,
    config_label,
    decimals,
    esc,
    fmt,
    fmt_ci,
)

SITE = "https://roycarlous.com"
PAGE_URL = f"{SITE}/case-studies/codeatlas.html"
REPO_URL = "https://github.com/carlous-roy/CodeAtlas"
TITLE = "CodeAtlas: Case Study · Roy Carlous Christudass"
DESCRIPTION = (
    "Semantic code search over four pinned projects, and the evaluation harness "
    "that measures it: 36 labelled questions, three chunkers, six retrieval "
    "strategies, bootstrap intervals on every number."
)

STYLE = """
  :root{
    color-scheme: light;
    --bg:#ffffff; --surface:#f6f6f7; --border:rgba(0,0,0,0.12);
    --ink:#0b0b0b; --ink-2:#333338; --ink-3:#5b5b66;
    --accent:#b91c1c; --accent-ink:#ffffff;
    --series-1:#1d5fb4; --series-2:#b4501d;
    --good-bg:rgba(34,197,94,0.12);
  }
  @media (prefers-color-scheme: dark){
    :root:not([data-theme="light"]){
      color-scheme: dark;
      --bg:#0b0b10; --surface:#15151c; --border:rgba(255,255,255,0.16);
      --ink:#f4f4f5; --ink-2:#d4d4dc; --ink-3:#a5a5b3;
      --accent:#f87171; --accent-ink:#0b0b10;
      --series-1:#6ea3f0; --series-2:#f0a06e;
      --good-bg:rgba(34,197,94,0.16);
    }
  }
  :root[data-theme="dark"]{
    color-scheme: dark;
    --bg:#0b0b10; --surface:#15151c; --border:rgba(255,255,255,0.16);
    --ink:#f4f4f5; --ink-2:#d4d4dc; --ink-3:#a5a5b3;
    --accent:#f87171; --accent-ink:#0b0b10;
    --series-1:#6ea3f0; --series-2:#f0a06e;
    --good-bg:rgba(34,197,94,0.16);
  }
  *{box-sizing:border-box}
  html{scroll-behavior:smooth}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif;
    -webkit-font-smoothing:antialiased}
  .wrap{max-width:820px;margin:0 auto;padding:clamp(24px,5vw,64px) 16px 80px}
  a{color:var(--accent);text-decoration:underline;text-underline-offset:2px}
  a:focus-visible,[tabindex]:focus-visible,summary:focus-visible{outline:3px solid var(--accent);outline-offset:3px}
  .skip{position:absolute;left:-999px;top:8px;background:var(--accent);color:var(--accent-ink);padding:8px 12px;border-radius:8px}
  .skip:focus{left:8px;z-index:10}
  nav.crumbs{font-size:14px;margin-bottom:28px}
  .eyebrow{font:600 12px/1 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);margin-bottom:12px}
  h1{font-size:clamp(30px,5.5vw,44px);line-height:1.1;letter-spacing:-.02em;font-weight:800;margin:0 0 12px}
  .lede{font-size:clamp(17px,2.2vw,20px);line-height:1.55;color:var(--ink-2);margin:0 0 22px;max-width:62ch}
  .meta{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:30px;padding:0;list-style:none}
  .meta li{font-size:13px;padding:6px 12px;border:1px solid var(--border);border-radius:999px;color:var(--ink-2);background:var(--surface)}
  h2{font-size:clamp(21px,3vw,27px);letter-spacing:-.01em;font-weight:700;margin:48px 0 8px}
  h3{font-size:17px;font-weight:650;margin:28px 0 6px}
  p,li{color:var(--ink-2)}
  p{margin:0 0 14px;max-width:70ch}
  ul{padding-left:20px;max-width:70ch}
  strong{color:var(--ink);font-weight:650}
  code{font:.92em ui-monospace,SFMono-Regular,Menlo,monospace}
  .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:22px 0 6px;padding:0;list-style:none}
  .stats li{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 16px}
  .stats .n{display:block;font-size:26px;font-weight:750;letter-spacing:-.02em;line-height:1.1;color:var(--ink)}
  .stats .ci{display:block;font-size:12px;color:var(--ink-3);margin-top:2px}
  .stats .l{display:block;font-size:13px;color:var(--ink-3);margin-top:6px;line-height:1.35}
  .scroll{overflow-x:auto;margin:18px 0;border:1px solid var(--border);border-radius:12px;background:var(--surface)}
  table{border-collapse:collapse;width:100%;font-size:14px}
  caption{text-align:left;font-size:13px;color:var(--ink-3);padding:10px 14px 6px;caption-side:top}
  th,td{padding:9px 12px;text-align:right;border-bottom:1px solid var(--border);white-space:nowrap;vertical-align:top}
  th[scope=row],td:first-child,th:first-child{text-align:left}
  thead th{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--ink-3);font-weight:600}
  tbody tr:last-child td,tbody tr:last-child th{border-bottom:0}
  tr.shipped th,tr.shipped td{font-weight:700;color:var(--ink);background:var(--good-bg)}
  .chart{list-style:none;padding:0;margin:16px 0}
  .chart li{display:grid;grid-template-columns:minmax(150px,1fr) 3fr;gap:6px 12px;align-items:center;padding:6px 0;border-bottom:1px solid var(--border)}
  .chart li:last-child{border-bottom:0}
  .chart .lab{font-size:13px;color:var(--ink-2)}
  .chart .lab small{display:block;color:var(--ink-3)}
  .bars{display:grid;gap:3px}
  .bar{position:relative;height:14px;background:var(--surface);border-radius:3px;overflow:hidden}
  .bar span{position:absolute;left:0;top:0;bottom:0;border-radius:3px}
  .bar.h span{background:var(--series-1)}
  .bar.m span{background:var(--series-2)}
  .bar em{position:absolute;right:6px;top:-1px;font:600 11px/14px ui-monospace,SFMono-Regular,Menlo,monospace;font-style:normal;color:var(--ink)}
  .legend{display:flex;gap:16px;font-size:13px;color:var(--ink-2);margin:6px 0}
  .legend i{display:inline-block;width:18px;height:10px;border-radius:2px;vertical-align:middle;margin-right:6px}
  @media (max-width:520px){
    .chart li{grid-template-columns:1fr}
    .stats{grid-template-columns:1fr 1fr}
  }
  .callout{border-left:3px solid var(--accent);background:var(--surface);border-radius:0 10px 10px 0;padding:14px 18px;margin:20px 0}
  .callout p:last-child{margin-bottom:0}
  pre{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px;overflow-x:auto;
    font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink-2)}
  footer{margin-top:56px;padding-top:20px;border-top:1px solid var(--border);font-size:14px;color:var(--ink-3)}
  footer p{margin:0 0 6px}
  .neg{color:var(--series-2)}
"""


# ------------------------------------------------------------- helpers
def _ci_text(cell: dict, metric: str) -> str:
    d = decimals(metric)
    lo, hi = cell["ci95"]
    return f"{lo:.{d}f} to {hi:.{d}f}"


def _n(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _comparison_sentence(r: Results, a: str, b: str, metric: str, a_label: str, b_label: str) -> str:
    """One plain sentence stating direction, size, interval and verdict."""
    c = r.comparison(a, b, metric)
    if c is None:
        return ""
    d = decimals(metric)
    lo, hi = c["ci95"]
    name = METRIC_LABEL[metric]
    if c["diff"] == 0:
        head = f"{a_label} and {b_label} tie on {name} at {c['a_value']:.{d}f}"
    else:
        head = f"{a_label} scores {name} {c['a_value']:.{d}f} against {c['b_value']:.{d}f} for {b_label}"
    body = (
        f" (difference {c['diff']:+.{d}f}, 95% interval {lo:+.{d}f} to {hi:+.{d}f}; "
        f"better on {_n(c['better'], 'question')}, worse on {c['worse']}, same on {c['same']})."
    )
    verdict = (
        " The interval includes zero, so this is within noise at this sample size."
        if c["within_noise"]
        else " The interval excludes zero."
    )
    return esc(head + body + verdict)


def _table(caption: str, header: list[str], rows: list[tuple[list[str], bool]], region_label: str) -> str:
    head = "".join(f'<th scope="col">{esc(h)}</th>' for h in header)
    body = []
    for cells, shipped in rows:
        first, rest = cells[0], cells[1:]
        tds = "".join(f"<td>{c}</td>" for c in rest)
        cls = ' class="shipped"' if shipped else ""
        body.append(f'<tr{cls}><th scope="row">{first}</th>{tds}</tr>')
    return (
        f'<div class="scroll" tabindex="0" role="region" aria-label="{esc(region_label)}">'
        f"<table><caption>{esc(caption)}</caption><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _stat(n: str, label: str, ci: str = "") -> str:
    ci_html = f'<span class="ci">95% interval {esc(ci)}</span>' if ci else ""
    return f'<li><span class="n">{esc(n)}</span>{ci_html}<span class="l">{esc(label)}</span></li>'


# ------------------------------------------------------------- sections
def _stats(r: Results) -> str:
    s = r.shipped["metrics"]
    return (
        '<ul class="stats" aria-label="Headline numbers">'
        + _stat(str(r.n), "questions with labelled answer files")
        + _stat(
            fmt(s["hit_rate@5"]["value"], "hit_rate@5"),
            "hit rate@5, shipped configuration",
            _ci_text(s["hit_rate@5"], "hit_rate@5"),
        )
        + _stat(
            fmt(s["recall@5"]["value"], "recall@5"),
            "recall@5, shipped configuration",
            _ci_text(s["recall@5"], "recall@5"),
        )
        + _stat(fmt(s["mrr"]["value"], "mrr"), "mean reciprocal rank", _ci_text(s["mrr"], "mrr"))
        + _stat(str(len(r.metrics["configurations"])), "configurations scored on the same questions")
        + "</ul>"
    )


def _corpus_table(r: Results) -> str:
    rows = []
    for name, info in r.manifest["projects"].items():
        repo = info.get("repository") or ""
        link = (
            f'<a href="{esc(repo.removesuffix(".git"))}">{esc(repo.removesuffix(".git").split("/")[-1])}</a>'
            if repo
            else "—"
        )
        rows.append(
            (
                [
                    f"<code>{esc(name)}</code>",
                    link,
                    f"<code>{esc((info.get('commit') or '')[:12])}</code>",
                    str(info["files"]),
                    f"{info['lines']:,}",
                ],
                False,
            )
        )
    t = r.manifest["totals"]
    rows.append(
        (
            ["<strong>total</strong>", "", "", f"<strong>{t['files']}</strong>", f"<strong>{t['lines']:,}</strong>"],
            False,
        )
    )
    return _table(
        "The four projects, at the commits recorded in results/corpus_manifest.json",
        ["Project", "Repository", "Commit", "Files", "Lines"],
        rows,
        "Corpus table",
    )


def _headline_table(r: Results) -> str:
    cols = ["hit_rate@1", "hit_rate@3", "hit_rate@5", "recall@5", "mrr", "ndcg@10"]
    row = [esc(config_label(r.shipped_key))] + [esc(fmt_ci(r.shipped["metrics"][m], m)) for m in cols]
    return _table(
        f"Shipped configuration on all {r.n} questions; value and 95% bootstrap interval",
        ["Configuration"] + [METRIC_LABEL[m] for m in cols],
        [(row, True)],
        "Shipped configuration table",
    )


def _chart(r: Results) -> str:
    items = []
    for c in r.metrics["configurations"]:
        h5 = c["metrics"]["hit_rate@5"]["value"]
        mrr = c["metrics"]["mrr"]["value"]
        items.append(
            f'<li><span class="lab">{esc(STRATEGY_LABEL[c["strategy"]])}<small>{esc(CHUNKING_LABEL[c["chunking"]])}</small></span>'
            f'<span class="bars"><span class="bar h" title="hit rate@5"><span style="width:{h5 * 100:.1f}%"></span><em>{h5:.2f}</em></span>'
            f'<span class="bar m" title="MRR"><span style="width:{mrr * 100:.1f}%"></span><em>{mrr:.3f}</em></span></span></li>'
        )
    return (
        '<div class="legend" aria-hidden="true"><span><i style="background:var(--series-1)"></i>hit rate@5</span>'
        '<span><i style="background:var(--series-2)"></i>MRR</span></div>'
        f'<ul class="chart" aria-label="Hit rate at 5 and MRR for every configuration; the table below has the same numbers with intervals">{"".join(items)}</ul>'
    )


def _all_table(r: Results) -> str:
    cols = ["hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10"]
    rows = []
    for c in r.metrics["configurations"]:
        cells = [esc(CHUNKING_LABEL[c["chunking"]]), esc(STRATEGY_LABEL[c["strategy"]]), str(c["n_chunks"])]
        cells += [esc(fmt_ci(c["metrics"][m], m)) for m in cols]
        rows.append((cells, c["key"] == r.shipped_key))
    return _table(
        f"Every configuration on the same {r.n} questions; value and 95% bootstrap interval",
        ["Chunking", "Strategy", "Chunks"] + [METRIC_LABEL[m] for m in cols],
        rows,
        "All configurations table",
    )


def _comparisons_table(r: Results) -> str:
    rows = []
    for c in r.comparisons:
        d = decimals(c["metric"])
        lo, hi = c["ci95"]
        diff = f"{c['diff']:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"
        reading = "within noise" if c["within_noise"] else "interval excludes zero"
        rows.append(
            (
                [
                    esc(c["label"]),
                    esc(METRIC_LABEL[c["metric"]]),
                    f"{c['a_value']:.{d}f}",
                    f"{c['b_value']:.{d}f}",
                    esc(diff),
                    f"{c['better']} / {c['worse']} / {c['same']}",
                    f"{c['sign_test_p']:.3f}",
                    esc(reading),
                ],
                False,
            )
        )
    return _table(
        "Paired comparisons on the same questions: mean difference with a 95% paired-bootstrap interval, question counts and an exact sign test",
        [
            "Comparison (A vs B)",
            "Metric",
            "A",
            "B",
            "A - B [95% CI]",
            "better / worse / same",
            "sign test p",
            "Reading",
        ],
        rows,
        "Comparisons table",
    )


def _cap_table(r: Results) -> str:
    cap = r.cap
    rows = []
    for row in cap["rows"]:
        pf = row["per_file"]
        label = "no cap" if pf is None else f"cap {pf}"
        chosen = pf == cap["chosen_per_file"]
        rows.append(
            (
                [
                    esc(label + (" (chosen)" if chosen else "")),
                    fmt(row["dev"]["mrr"], "mrr"),
                    fmt(row["dev"]["recall@5"], "recall@5"),
                    fmt(row["test"]["mrr"], "mrr"),
                    fmt(row["test"]["recall@5"], "recall@5"),
                    fmt(row["all"]["mrr"], "mrr"),
                    fmt(row["all"]["hit_rate@5"], "hit_rate@5"),
                    fmt(row["all"]["recall@5"], "recall@5"),
                ],
                chosen,
            )
        )
    return _table(
        f"Per-file cap values tried on the dev half (n={cap['dev_n']}) of the merged structural chunking with hybrid retrieval; the test half (n={cap['test_n']}) and all questions shown for reference",
        [
            "Per-file cap",
            "dev MRR",
            "dev recall@5",
            "test MRR",
            "test recall@5",
            "all MRR",
            "all hit rate@5",
            "all recall@5",
        ],
        rows,
        "Per-file cap table",
    )


def _failures_table(r: Results) -> str:
    f = r.failures["configurations"][r.failures["shipped"]]
    rows = []
    for label, key in (
        ("share of the top 5 that is documentation", "doc_share_top5"),
        ("share of the top 5 from the wrong project", "wrong_project_share_top5"),
    ):
        d = f[key]
        lo, hi = d["diff_ci95"]
        rows.append(
            (
                [
                    esc(label),
                    f"{d['misses']:.2f}",
                    f"{d['hits']:.2f}",
                    f"{d['misses'] - d['hits']:+.2f} [{lo:+.2f}, {hi:+.2f}]",
                ],
                False,
            )
        )
    return _table(
        f"What fills the top 5 for the {f['n_misses']} questions the shipped configuration misses at rank 5, against the {f['n_hits']} it hits",
        ["Share", "On misses", "On hits", "Difference [95% CI]"],
        rows,
        "Failure analysis table",
    )


def _chunking_table(r: Results) -> str:
    rows = []
    for chunking, info in r.metadata["chunkings"].items():
        rows.append(
            (
                [
                    esc(CHUNKING_LABEL[chunking]),
                    str(info["n_chunks"]),
                    str(info["median_lines"]),
                    str(info["p95_lines"]),
                    str(info["max_lines"]),
                    str(info["median_payload_tokens"]),
                    str(info["max_payload_tokens"]),
                ],
                False,
            )
        )
    return _table(
        f"Chunk statistics; every payload fits the {r.metadata['settings']['payload_budget_tokens']}-token budget",
        ["Chunking", "Chunks", "Median lines", "p95 lines", "Max lines", "Median tokens", "Max tokens"],
        rows,
        "Chunking statistics table",
    )


def _reranker_paragraphs(r: Results) -> str:
    s = "structural_merged"
    text_only = r.get(f"{s}/hybrid+rerank")
    prefixed = r.get(f"{s}/hybrid+rerank+prefix")
    if text_only is None or prefixed is None:
        return "<p>The reranker was not run in this evaluation.</p>"
    out = []
    out.append(
        "<p>"
        + _comparison_sentence(
            r,
            f"{s}/hybrid+rerank",
            f"{s}/hybrid",
            "mrr",
            "Reranking the chunk text alone",
            "hybrid retrieval without it",
        )
        + " "
        + _comparison_sentence(
            r,
            f"{s}/hybrid+rerank+prefix",
            f"{s}/hybrid",
            "mrr",
            "Reranking the same path-and-name payload the embedder sees",
            "hybrid retrieval",
        )
        + "</p>"
    )
    out.append(
        "<p>"
        + _comparison_sentence(
            r, f"{s}/hybrid+rerank+prefix", f"{s}/hybrid+rerank", "mrr", "The prefixed variant", "the text-only variant"
        )
        + " "
        + _comparison_sentence(
            r,
            "window/hybrid+rerank",
            "window/hybrid",
            "mrr",
            "On window chunks, the text-only reranker",
            "hybrid retrieval",
        )
        + "</p>"
    )
    return "".join(out)


def _generated_line(r: Results) -> str:
    m = r.metadata
    commit = m.get("codeatlas_commit") or ""
    date = m["generated_at"][:10]
    projects = (m.get("corpus") or {}).get("projects") or {}
    corpus = ", ".join(f"{name} {sha[:7] if sha else 'unpinned'}" for name, sha in projects.items())
    return (
        f"Results generated {esc(date)} by codeatlas {esc(m['codeatlas_version'])}"
        + (f" at commit <code>{esc(commit[:12])}</code>" if commit else "")
        + (f"; corpus commits {esc(corpus)}." if corpus else "; corpus not pinned.")
    )


# ------------------------------------------------------------- page
def render(r: Results) -> str:
    s = r.shipped["metrics"]
    n = r.n
    cap_value = r.metadata["settings"].get("per_file")
    files = r.manifest["totals"]["files"]
    lines = r.manifest["totals"]["lines"]
    date = r.metadata["generated_at"][:10]
    embedder = r.metadata["models"]["embedder"]
    reranker = r.metadata["models"].get("reranker") or {}
    budget = r.metadata["settings"]["payload_budget_tokens"]
    f = r.failures["configurations"][r.failures["shipped"]]
    misses = ", ".join(f"<code>{esc(m)}</code>" for m in f["misses_at_5"]) or "none"
    doc = f["doc_share_top5"]
    wrong = f["wrong_project_share_top5"]
    doc_verdict = (
        "includes zero, so it is within noise" if doc["diff_ci95"][0] <= 0 <= doc["diff_ci95"][1] else "excludes zero"
    )
    wrong_verdict = (
        "includes zero, so it is within noise"
        if wrong["diff_ci95"][0] <= 0 <= wrong["diff_ci95"][1]
        else "excludes zero"
    )
    best_other = next((c for c in r.comparisons if c["label"].startswith("shipped configuration")), None)

    parts: list[str] = []
    parts.append(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(TITLE)}</title>
<meta name="description" content="{esc(DESCRIPTION)}">
<link rel="canonical" href="{PAGE_URL}">
<meta name="theme-color" content="#ffffff" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#0b0b10" media="(prefers-color-scheme: dark)">
<meta property="og:type" content="article">
<meta property="og:title" content="CodeAtlas: semantic code search with a measured evaluation">
<meta property="og:description" content="{esc(DESCRIPTION)}">
<meta property="og:url" content="{PAGE_URL}">
<meta property="og:site_name" content="roycarlous.com">
<meta property="article:modified_time" content="{esc(r.metadata["generated_at"])}">
<meta name="twitter:card" content="summary">
<meta name="twitter:title" content="CodeAtlas: semantic code search with a measured evaluation">
<meta name="twitter:description" content="{esc(DESCRIPTION)}">
<link rel="icon" href="/favicon.png">
<style>{STYLE}</style>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<div class="wrap">
<header>
  <nav class="crumbs" aria-label="Breadcrumb"><a href="/">&larr; roycarlous.com</a></nav>
  <p class="eyebrow">Case study</p>
  <h1>CodeAtlas</h1>
  <p class="lede">Semantic search over four of my own projects, and the evaluation harness that measures it. The numbers on this page are rendered from the committed results files by a script; a rerun on the pinned corpus regenerates them.</p>
  <ul class="meta" aria-label="Project facts">
    <li><a href="{REPO_URL}">Code on GitHub</a></li>
    <li>Python &middot; sentence-transformers &middot; Tree-sitter &middot; BM25</li>
    <li>{files} indexed files &middot; {lines:,} lines &middot; 4 projects</li>
    <li>{n} questions &middot; one labeller</li>
    <li>Results of {esc(date)}</li>
  </ul>
</header>
<main id="main">
{_stats(r)}
"""
    )

    parts.append(
        f"""
<h2 id="what-is-measured">What is measured</h2>
<p>The corpus is four repositories of mine, checked out at fixed commits and listed file by file, with line counts and hashes, in the results directory. The evaluator refuses to run on a corpus that differs from that list, so the numbers here can be regenerated by anyone.</p>
{_corpus_table(r)}
<p>The question set is {n} questions I wrote against these projects, phrased the way someone new to a repository would ask rather than by pasting identifiers, each with the files that answer it. Relevance is judged at file level: a retrieved chunk counts when its source file is one of the labelled answers. I wrote both the questions and the code, and there has been one labeller so far; the labelling protocol for a second round with two independent labellers is in the repository.</p>
<p>The questions are split 50/50 into a dev half and a test half. The only setting chosen on the data, the per-file cap value, is chosen on the dev half; every table reports all {n} questions with a bootstrap interval, because eighteen questions per half say very little on their own.</p>
<h3>Metrics</h3>
<ul>
  <li><strong>hit rate@k</strong>: 1 when any labelled file appears in the top k chunks. The first version of this project called this recall@k.</li>
  <li><strong>recall@k</strong>: the share of the labelled files that appear in the top k chunks. For a question with four answer files, one of them in the top 5 is a hit rate of 1 and a recall of 0.25.</li>
  <li><strong>MRR</strong>: one over the rank of the first relevant chunk, averaged over questions.</li>
  <li><strong>nDCG@10</strong>: discounted gain with one binary gain per relevant file at its first chunk, normalised by the ideal ordering of the labelled set.</li>
</ul>
<p>Every cell carries a 95% percentile bootstrap interval over questions (10,000 resamples). Every claim that one configuration beats another rests on a paired bootstrap of the per-question differences and an exact sign test on the questions where the two differ. When the interval includes zero the page says so.</p>
"""
    )

    parts.append(
        f"""
<h2 id="results">Results</h2>
<p>The shipped configuration is merged structural chunking, hybrid retrieval, and a per-file cap of {cap_value}.</p>
{_headline_table(r)}
<h3>Every configuration</h3>
{_chart(r)}
{_all_table(r)}
"""
    )

    parts.append(
        f"""
<h2 id="what-beat-what">What beat what, and what was within noise</h2>
<p><strong>Structural boundaries against windows.</strong> {_comparison_sentence(r, "structural_merged/hybrid", "window/hybrid", "hit_rate@1", "Merged structural chunking with hybrid retrieval", "windows")} {_comparison_sentence(r, "structural_merged/hybrid", "window/hybrid", "mrr", "The same chunking", "windows")} Both chunkers are packed to the same token budget, so the comparison is about where boundaries fall, not about chunk size.</p>
<p><strong>Hybrid against dense.</strong> {_comparison_sentence(r, "structural_merged/hybrid", "structural_merged/dense", "mrr", "Hybrid retrieval", "dense retrieval alone")} {_comparison_sentence(r, "structural_merged/hybrid", "structural_merged/dense", "hit_rate@5", "Hybrid retrieval", "dense retrieval")} Fusing in the BM25 ranking pushes some files that dense retrieval had in the top 5 below the cut.</p>
<p><strong>Hybrid against BM25.</strong> {_comparison_sentence(r, "structural_merged/hybrid", "structural_merged/bm25", "mrr", "Hybrid retrieval", "BM25 alone")}</p>
<p><strong>The per-file cap.</strong> {_comparison_sentence(r, "structural_merged/hybrid+cap", "structural_merged/hybrid", "recall@5", "With the cap, hybrid retrieval", "the same ranking without it")} {_comparison_sentence(r, "structural_merged/hybrid+cap", "structural_merged/hybrid", "mrr", "The cap", "no cap")} Under file-level relevance the cap cannot lower hit rate or MRR, because the first chunk of every file is always kept and only moves up; the metric it can lower is recall, and it did not.</p>
{_cap_table(r)}
<p><strong>The shipped configuration against the next best cell.</strong> {(_comparison_sentence(r, best_other["a"], best_other["b"], "mrr", "The shipped configuration", config_label(best_other["b"])) if best_other else "")}</p>
<h3>All comparisons</h3>
{_comparisons_table(r)}
"""
    )

    parts.append(
        f"""
<h2 id="reranker">The cross-encoder reranker</h2>
<p>Reranking the top 30 with <code>{esc(reranker.get("name", "the cross-encoder"))}</code> was run two ways: on the chunk text alone, and on the same <code>path :: name</code> payload the embedder sees, so that the reranker is not judged on less information than the first stage had. Every chunk fits the reranker's window, so nothing is truncated in either variant.</p>
{_reranker_paragraphs(r)}
<p>What the data supports: in this setup the reranker does not improve ranking, and on merged structural chunks the text-only variant is worse by more than the noise. It does not support a cause. Whether the model's training domain, the small candidate set, or the file-level judgement explains it is not something this experiment tested.</p>
"""
    )

    parts.append(
        f"""
<h2 id="failures">Where it fails</h2>
<p>The shipped configuration misses {f["n_misses"]} of {n} questions at rank 5: {misses}. Two things were measured about what fills the top 5 on those questions.</p>
{_failures_table(r)}
<p>The documentation share differs by {doc["misses"] - doc["hits"]:+.2f} between misses and hits, and its interval {doc_verdict}. The wrong-project share differs by {wrong["misses"] - wrong["hits"]:+.2f}, and its interval {wrong_verdict}. The corpus holds four separate projects and nothing in the retriever scopes a query to one of them; a project filter is the obvious next fix, and it would be tested the same way.</p>
"""
    )

    parts.append(
        f"""
<h2 id="how-it-works">How it works</h2>
<pre>files ──► chunker ──┬─► BM25 index          ┐
                    └─► MiniLM embeddings   ├─► RRF fusion ─► per-file cap ─► top k
                                            ┘</pre>
<p>Tree-sitter parses Python and Java. A class becomes a header chunk (its signature, fields and docstring) plus one chunk per method, so no line is indexed twice; imports and module-level code become their own chunks; Markdown splits on headings and carries the heading path into the text. Files without a grammar (JavaScript, YAML) use windows in every chunking. Every chunk is sized to fit the embedder's window: <code>{esc(embedder["name"])}</code> reads at most {embedder["max_tokens"]} word pieces, so the <code>path :: name</code> prefix plus the text is kept within {budget} tokens, and a declaration longer than that is split into consecutive parts. The window baseline packs lines to the same budget with a quarter of each window repeated in the next.</p>
{_chunking_table(r)}
<p>BM25 runs on a tokeniser that splits identifiers (<code>RateLimitFilter</code> is reachable from "rate limit"), drops English stopwords and stems with Snowball. Fusion is Reciprocal Rank Fusion with k = 60 over the top 60 of each list. The per-file cap keeps the first {"chunk" if cap_value == 1 else f"{cap_value} chunks"} of every file at the head of the list and pushes the rest below every survivor, so nothing is discarded. Both models run on CPU at pinned revisions.</p>
"""
    )

    parts.append(
        f"""
<h2 id="limitations">Limitations</h2>
<ul>
  <li><strong>{n} questions.</strong> The intervals are the honest width: a hit rate of {fmt(s["hit_rate@5"]["value"], "hit_rate@5")} at rank 5 sits in {_ci_text(s["hit_rate@5"], "hit_rate@5")}. Most differences between configurations are inside that width.</li>
  <li><strong>One labeller, who also wrote the code.</strong> A second round with two independent labellers, an agreement statistic and an adjudication log is specified but not done.</li>
  <li><strong>File-level relevance is generous.</strong> A chunk from the right file counts even when it is the wrong function in that file.</li>
  <li><strong>The dev/test halves are small.</strong> The cap value was chosen on eighteen questions; the test half agrees on direction, and that is all it can say.</li>
  <li><strong>Retrieval only.</strong> No generation, so no measurement of whether an answer follows from the retrieved context.</li>
</ul>
</main>
<footer>
  <p>{_generated_line(r)}</p>
  <p>Roy Carlous Christudass &middot; <a href="/">roycarlous.com</a> &middot; <a href="https://github.com/carlous-roy">GitHub</a> &middot; <a href="https://www.linkedin.com/in/roy-carlous-c/">LinkedIn</a></p>
  <noscript><p>This page needs no JavaScript; every number is in the tables above.</p></noscript>
</footer>
</div>
</body>
</html>
"""
    )
    return "".join(parts)
