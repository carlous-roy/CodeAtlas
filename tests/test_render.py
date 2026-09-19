from __future__ import annotations

from pathlib import Path

import pytest

from codeatlas.render import BLOCKS, Results, fmt_ci, render_case_study, render_readme

RESULTS = Path("results")


@pytest.fixture(scope="module")
def results() -> Results:
    return Results(RESULTS)


def test_results_loads_committed_files(results):
    assert results.n == 36
    assert results.shipped_key in results.by_key
    assert results.shipped["strategy"] in ("hybrid+cap", "hybrid")


def test_every_block_renders_a_table(results):
    for name, block in BLOCKS.items():
        text = block(results)
        assert text.strip(), name
        if name != "metadata":
            assert text.startswith("| "), name


def test_readme_markers_are_replaced_and_kept():
    r = Results(RESULTS)
    template = "intro\n<!-- codeatlas:table:headline -->\nstale\n<!-- /codeatlas:table:headline -->\noutro\n"
    out = render_readme(template, r)
    assert out.startswith("intro\n<!-- codeatlas:table:headline -->\n| Configuration |")
    assert out.endswith("<!-- /codeatlas:table:headline -->\noutro\n")
    assert "stale" not in out
    assert render_readme(out, r) == out


def test_readme_rejects_unknown_block():
    r = Results(RESULTS)
    with pytest.raises(KeyError):
        render_readme("<!-- codeatlas:table:nope -->\n<!-- /codeatlas:table:nope -->", r)
    with pytest.raises(ValueError):
        render_readme("no markers here", r)


def test_committed_readme_and_case_study_are_current():
    r = Results(RESULTS)
    readme = Path("README.md").read_text(encoding="utf-8")
    assert render_readme(readme, r) == readme
    assert Path("docs/case-study.html").read_text(encoding="utf-8") == render_case_study(r)


def test_case_study_carries_the_headline_numbers_and_landmarks(results):
    page = render_case_study(results)
    mrr = results.shipped["metrics"]["mrr"]
    assert f"{mrr['value']:.3f}" in page
    assert fmt_ci(results.shipped["metrics"]["hit_rate@5"], "hit_rate@5") in page
    for tag in (
        "<header>",
        '<main id="main">',
        "<footer>",
        '<nav class="crumbs" aria-label="Breadcrumb">',
        "<noscript>",
    ):
        assert tag in page
    assert "<script" not in page
    assert page.count("<caption>") == page.count("<table>")
    assert 'property="og:title"' in page
    assert results.metadata["generated_at"][:10] in page
