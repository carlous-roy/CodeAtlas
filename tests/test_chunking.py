from __future__ import annotations

import re
from itertools import pairwise

import pytest

from codeatlas.chunking import (
    LANGUAGES,
    Unit,
    _Sizer,
    build_chunks,
    chunk_file,
    code_units,
    markdown_units,
    merge_units,
    window_units,
)

ALNUM = re.compile(r"[A-Za-z0-9]")
PY = "corpus/demo/app/service.py"
JAVA = "corpus/demo/src/main/java/demo/ReportService.java"
MD = "corpus/demo/README.md"


def _by_name(chunks):
    return {c.name: c for c in chunks}


def assert_partition(units: list[Unit], lines: list[str]) -> None:
    """Every line with an alphanumeric character is in exactly one unit."""
    cover = [0] * len(lines)
    for u in units:
        for row in range(u.start, u.end + 1):
            cover[row] += 1
    for i, line in enumerate(lines):
        if ALNUM.search(line):
            assert cover[i] == 1, f"line {i + 1} covered {cover[i]} times: {line!r}"
    for a, b in pairwise(units):
        assert b.start > a.end


# ---------------------------------------------------------------- python
def test_python_units_partition_the_file(read_fixture):
    src = read_fixture("app/service.py")
    units = code_units(src, LANGUAGES[".py"])
    assert_partition(units, src.splitlines())


def test_python_declarations_named_and_scoped(read_fixture, count):
    chunks = _by_name(chunk_file(PY, read_fixture("app/service.py"), "structural", count))
    assert chunks["Job"].kind == "class"
    assert chunks["Job.next_delay"].kind == "method"
    assert chunks["RateLimiter.allow"].kind == "method"
    assert chunks["submit"].kind == "function"
    assert chunks["module level"].kind == "preamble"


def test_python_class_header_holds_docstring_and_fields_only(read_fixture, count):
    chunks = _by_name(chunk_file(PY, read_fixture("app/service.py"), "structural", count))
    header = chunks["Job"]
    assert "One report job" in header.text
    assert "attempts: int = 0" in header.text
    assert "def next_delay" not in header.text


def test_python_decorator_and_leading_comment_belong_to_the_declaration(read_fixture, count):
    chunks = _by_name(chunk_file(PY, read_fixture("app/service.py"), "structural", count))
    assert chunks["submit"].text.startswith("@log_calls\ndef submit")
    assert chunks["Job.give_up"].text.lstrip().startswith("# Mark the job as failed for good.\n    def give_up")


def test_python_module_level_code_is_kept(read_fixture, count):
    chunks = chunk_file(PY, read_fixture("app/service.py"), "structural", count)
    preambles = [c for c in chunks if c.kind == "preamble"]
    texts = "\n".join(c.text for c in preambles)
    assert "MAX_ATTEMPTS = 3" in texts
    assert 'if __name__ == "__main__":' in texts
    assert all(c.start_line <= c.end_line for c in chunks)


def test_chunk_lines_match_text(read_fixture, count):
    src = read_fixture("app/service.py")
    lines = src.splitlines()
    for c in chunk_file(PY, src, "structural", count):
        assert c.text == "\n".join(lines[c.start_line - 1 : c.end_line])


# ---------------------------------------------------------------- java
def test_java_method_names_are_identifiers_not_return_types(read_fixture, count):
    chunks = _by_name(chunk_file(JAVA, read_fixture("src/main/java/demo/ReportService.java"), "structural", count))
    assert "ReportService.getDownloadUrl" in chunks
    assert "ReportService.submit" in chunks
    assert "ReportService.listReports" in chunks
    assert not any(name.endswith(".String") or name.endswith(".ReportJob") for name in chunks)
    assert chunks["ReportService.ReportService"].kind == "constructor"
    assert chunks["ReportService.ReportType"].kind == "enum"


def test_java_javadoc_attaches_to_its_method(read_fixture, count):
    chunks = _by_name(chunk_file(JAVA, read_fixture("src/main/java/demo/ReportService.java"), "structural", count))
    assert chunks["ReportService.getDownloadUrl"].text.lstrip().startswith("/** Where a finished report")
    assert "/** Creates report jobs" in chunks["ReportService"].text
    assert "private final StorageClient storage;" in chunks["ReportService"].text


def test_java_units_partition_the_file(read_fixture):
    src = read_fixture("src/main/java/demo/ReportService.java")
    assert_partition(code_units(src, LANGUAGES[".java"]), src.splitlines())


# ---------------------------------------------------------------- markdown
def test_markdown_sections_carry_heading_path(read_fixture, count):
    chunks = _by_name(chunk_file(MD, read_fixture("README.md"), "structural", count))
    assert "Demo service > Fault tolerance > Retries" in chunks
    retries = chunks["Demo service > Fault tolerance > Retries"]
    assert retries.text.startswith("Demo service > Fault tolerance > Retries\n\n### Retries")
    assert retries.kind == "markdown_section"


def test_markdown_without_headings_falls_back_to_windows(count):
    chunks = chunk_file("corpus/demo/notes.md", "just a paragraph\nand another\n", "structural", count)
    assert [c.kind for c in chunks] == ["window"]


def test_markdown_preamble_before_first_heading():
    units = markdown_units("intro line\n\n# Title\nbody\n")
    assert units[0].name == "(preamble)"
    assert units[1].name == "Title"


# ---------------------------------------------------------------- fallback and windows
def test_files_without_grammar_use_windows_in_every_chunking(read_fixture, count):
    src = read_fixture("web/app.jsx")
    for chunking in ("structural", "structural_merged", "window"):
        chunks = chunk_file("corpus/demo/web/app.jsx", src, chunking, count)
        assert chunks and all(c.kind == "window" for c in chunks)


def test_windows_respect_budget_and_overlap():
    lines = [f"word{i} extra" for i in range(40)]  # two tokens per line
    sizer = _Sizer("corpus/p/f.txt", lines, lambda s: len(s.split()), budget=30)
    units = window_units(sizer)
    prefix = sizer.prefix_tokens("lines 40-40")
    for u in units:
        assert sizer.span_tokens(u.start, u.end) + prefix <= 30
    # each window starts inside the previous one: the overlap
    for a, b in pairwise(units):
        assert a.start < b.start <= a.end
    assert units[0].start == 0
    assert units[-1].end == len(lines) - 1


def test_long_line_is_split_on_whitespace():
    line = " ".join(f"w{i}" for i in range(50))
    sizer = _Sizer("corpus/p/f.txt", [line], lambda s: len(s.split()), budget=30)
    units = window_units(sizer)
    assert len(units) > 1
    assert all(u.atomic for u in units)
    assert " ".join(u.text for u in units) == line


# ---------------------------------------------------------------- merge
def test_merge_packs_neighbours_without_exceeding_budget():
    lines = [f"line{i} a b" for i in range(12)]  # three tokens per line
    units = [Unit(i, i, f"u{i}", "function") for i in range(12)]
    sizer = _Sizer("corpus/p/f.py", lines, lambda s: len(s.split()), budget=40)
    merged = merge_units(units, sizer)
    assert len(merged) < len(units)
    for m in merged:
        assert sizer.fits(m.name, "", m.start, m.end)
    assert merged[0].name.startswith("u0 | u1")
    assert [m.start for m in merged] == [0] + [merged[i - 1].end + 1 for i in range(1, len(merged))]
    assert merged[-1].end == 11


def test_merged_text_is_the_exact_source_span_without_duplicates(read_fixture, count):
    src = read_fixture("src/main/java/demo/ReportService.java")
    lines = src.splitlines()
    chunks = chunk_file(JAVA, src, "structural_merged", count)
    merged = [c for c in chunks if c.kind == "merged"]
    assert merged, "the fixture should produce at least one merged chunk"
    for c in chunks:
        assert c.text == "\n".join(lines[c.start_line - 1 : c.end_line])
    # no source line appears in two chunks
    seen: set[int] = set()
    for c in chunks:
        rows = set(range(c.start_line, c.end_line + 1))
        assert not rows & seen
        seen |= rows


def test_merge_never_joins_atomic_pieces():
    lines = ["a b", "c d", "e f"]
    units = [
        Unit(0, 0, "x", "window", text="a b", atomic=True),
        Unit(1, 1, "y", "function"),
        Unit(2, 2, "z", "function"),
    ]
    sizer = _Sizer("corpus/p/f.py", lines, lambda s: len(s.split()), budget=100)
    merged = merge_units(units, sizer)
    assert merged[0].atomic and merged[0].text == "a b"
    assert merged[1].start == 1 and merged[1].end == 2


# ---------------------------------------------------------------- corpus level
@pytest.mark.parametrize("chunking", ["structural", "structural_merged", "window"])
def test_build_chunks_over_fixture_corpus(fixture_corpus, count, chunking):
    chunks = build_chunks(fixture_corpus, chunking, count)
    assert {c.path for c in chunks} == {
        "corpus/demo/README.md",
        "corpus/demo/app/service.py",
        "corpus/demo/config.yml",
        "corpus/demo/src/main/java/demo/ReportService.java",
        "corpus/demo/web/app.jsx",
    }
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c.project == "demo" for c in chunks)
