from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from codeatlas.cli import build_parser, main
from codeatlas.corpus import ProjectPin, build_manifest, load_manifest, verify_manifest, write_manifest
from codeatlas.evaluation import load_golden, load_split, make_split

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def corpus_copy(tmp_path) -> Path:
    dest = tmp_path / "corpus"
    shutil.copytree(FIXTURES / "corpus", dest)
    return dest


def test_manifest_lists_every_indexed_file(corpus_copy):
    manifest = build_manifest(corpus_copy)
    assert manifest["totals"]["files"] == 5
    assert {e["path"] for e in manifest["files"]} == {
        "README.md",
        "app/service.py",
        "config.yml",
        "src/main/java/demo/ReportService.java",
        "web/app.jsx",
    }
    assert all(e["lines"] > 0 and len(e["sha256"]) == 64 for e in manifest["files"])
    assert manifest["projects"]["demo"]["commit"] is None


def test_committed_fixture_manifest_matches_fixture_corpus():
    manifest = load_manifest(FIXTURES / "corpus_manifest.json")
    assert verify_manifest(FIXTURES / "corpus", manifest) == []


def test_verify_reports_changed_added_and_missing_files(corpus_copy):
    manifest = build_manifest(corpus_copy)
    assert verify_manifest(corpus_copy, manifest) == []
    (corpus_copy / "demo" / "config.yml").write_text("queue:\n  name: other\n", encoding="utf-8")
    (corpus_copy / "demo" / "extra.py").write_text("x = 1\n", encoding="utf-8")
    (corpus_copy / "demo" / "README.md").unlink()
    problems = verify_manifest(corpus_copy, manifest)
    assert "content differs: demo/config.yml" in problems
    assert "not in manifest: demo/extra.py" in problems
    assert "missing: demo/README.md" in problems
    assert len(problems) == 3


def test_verify_ignores_files_the_indexer_ignores(corpus_copy):
    manifest = build_manifest(corpus_copy)
    (corpus_copy / "demo" / "logo.png").write_bytes(b"\x89PNG")
    (corpus_copy / "demo" / "package-lock.json").write_text("{}", encoding="utf-8")
    assert verify_manifest(corpus_copy, manifest) == []


def test_manifest_records_explicit_pins(corpus_copy):
    pin = ProjectPin.parse("demo=https://example.org/demo.git@0123456789abcdef")
    manifest = build_manifest(corpus_copy, [pin])
    assert manifest["projects"]["demo"]["repository"] == "https://example.org/demo.git"
    assert manifest["projects"]["demo"]["commit"] == "0123456789abcdef"


def test_pin_parsing_rejects_malformed_specs():
    with pytest.raises(ValueError):
        ProjectPin.parse("demo=https://example.org/demo.git")
    with pytest.raises(ValueError):
        ProjectPin.parse("nonsense")


def test_manifest_round_trip(tmp_path, corpus_copy):
    manifest = build_manifest(corpus_copy)
    out = tmp_path / "m.json"
    write_manifest(manifest, out)
    assert load_manifest(out) == manifest


# ---------------------------------------------------------------- split
def test_make_split_is_stratified_and_deterministic():
    _, questions = load_golden(Path("eval/golden.json"))
    split = make_split(questions, seed=20260917)
    assert split == make_split(questions, seed=20260917)
    ids = {q.id for q in questions}
    assert set(split["dev"]) | set(split["test"]) == ids
    assert not set(split["dev"]) & set(split["test"])
    assert abs(len(split["dev"]) - len(split["test"])) <= 1
    for prefix in ("tf", "dl", "gc", "pf"):
        dev = sum(1 for i in split["dev"] if i.startswith(prefix))
        test = sum(1 for i in split["test"] if i.startswith(prefix))
        assert abs(dev - test) <= 1


def test_committed_split_covers_golden_set():
    _, questions = load_golden(Path("eval/golden.json"))
    split = load_split(Path("eval/split.json"), questions)
    assert len(split["dev"]) == 18 and len(split["test"]) == 18


def test_load_split_rejects_overlap(tmp_path):
    _, questions = load_golden(FIXTURES / "golden.json")
    bad = tmp_path / "split.json"
    bad.write_text(json.dumps({"dev": ["fx01", "fx02"], "test": ["fx02", "fx03", "fx04", "fx05", "fx06"]}))
    with pytest.raises(ValueError):
        load_split(bad, questions)


# ---------------------------------------------------------------- cli
def test_cli_parser_lists_commands():
    parser = build_parser()
    args = parser.parse_args(["evaluate", "--no-rerank", "--chunkings", "window"])
    assert args.no_rerank and args.chunkings == "window"
    args = parser.parse_args(["corpus", "fetch", "--project", "a=https://x/y.git@abcdef0"])
    assert args.project == ["a=https://x/y.git@abcdef0"]


def test_cli_corpus_verify_and_make_split(tmp_path, capsys):
    rc = main(
        ["corpus", "verify", "--corpus", str(FIXTURES / "corpus"), "--manifest", str(FIXTURES / "corpus_manifest.json")]
    )
    assert rc == 0
    out = tmp_path / "split.json"
    rc = main(["make-split", "--golden", str(FIXTURES / "golden.json"), "--out", str(out)])
    assert rc == 0
    split = json.loads(out.read_text())
    assert sorted(split["dev"] + split["test"]) == [f"fx0{i}" for i in range(1, 7)]


def test_cli_evaluate_refuses_without_manifest(tmp_path, capsys):
    rc = main(["evaluate", "--corpus", str(FIXTURES / "corpus"), "--manifest", str(tmp_path / "none.json")])
    assert rc == 2
    assert "not found" in capsys.readouterr().err
