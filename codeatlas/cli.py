"""Command line entry points: ``codeatlas <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from codeatlas import __version__
from codeatlas.chunking import CHUNKINGS, build_chunks, read_chunks, size_summary, write_chunks
from codeatlas.metrics import BOOTSTRAP_RESAMPLES, SEED
from codeatlas.retrieval import DEFAULT_PER_FILE, STRATEGIES

DEFAULT_CORPUS = Path("corpus")
DEFAULT_INDEX = Path("index")
DEFAULT_RESULTS = Path("results")
DEFAULT_GOLDEN = Path("eval/golden.json")
DEFAULT_SPLIT = Path("eval/split.json")
DEFAULT_MANIFEST = DEFAULT_RESULTS / "corpus_manifest.json"


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )
    for noisy in (
        "sentence_transformers",
        "transformers",
        "huggingface_hub",
        "urllib3",
        "filelock",
        "httpx",
        "httpcore",
    ):
        logging.getLogger(noisy).setLevel(logging.ERROR)


# ------------------------------------------------------------------ corpus
def cmd_corpus_fetch(args: argparse.Namespace) -> int:
    from codeatlas.corpus import ProjectPin, fetch, load_manifest, pins_from_manifest

    if args.project:
        pins = [ProjectPin.parse(spec) for spec in args.project]
    else:
        pins = pins_from_manifest(load_manifest(args.manifest))
    fetch(pins, args.corpus)
    for pin in pins:
        print(f"{pin.name:16s} {pin.commit}  {pin.repository}")
    return 0


def cmd_corpus_manifest(args: argparse.Namespace) -> int:
    from codeatlas.corpus import ProjectPin, build_manifest, write_manifest

    pins = [ProjectPin.parse(spec) for spec in args.project] if args.project else None
    manifest = build_manifest(args.corpus, pins)
    write_manifest(manifest, args.out)
    t = manifest["totals"]
    print(f"wrote {args.out}: {len(manifest['projects'])} projects, {t['files']} files, {t['lines']} lines")
    return 0


def cmd_corpus_verify(args: argparse.Namespace) -> int:
    from codeatlas.corpus import load_manifest, verify_manifest

    problems = verify_manifest(args.corpus, load_manifest(args.manifest))
    if problems:
        print(f"{len(problems)} problems:")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"{args.corpus} matches {args.manifest}")
    return 0


# ------------------------------------------------------------------ index
def cmd_build_index(args: argparse.Namespace) -> int:
    from codeatlas.models import load_token_counter

    count = load_token_counter()
    chunkings = CHUNKINGS if args.chunking == "all" else (args.chunking,)
    for chunking in chunkings:
        chunks = build_chunks(args.corpus, chunking, count)
        write_chunks(chunks, args.index / f"chunks_{chunking}.jsonl")
        print(f"{chunking:18s} {json.dumps(size_summary(chunks))}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from codeatlas.models import load_embedder, load_reranker
    from codeatlas.retrieval import Index

    chunk_file = args.index / f"chunks_{args.chunking}.jsonl"
    if not chunk_file.exists():
        print(f"{chunk_file} not found; run `codeatlas build-index` first", file=sys.stderr)
        return 1
    chunks = read_chunks(chunk_file)
    reranker = load_reranker() if args.strategy.startswith("hybrid+rerank") else None
    index = Index.build(chunks, load_embedder(), cache_dir=args.index, label=args.chunking, reranker=reranker)
    for rank, i in enumerate(index.search(args.query, args.strategy, k=args.k, per_file=args.per_file), start=1):
        c = index.chunks[i]
        print(f"{rank:2d}. {c.path}:{c.start_line}-{c.end_line}  {c.kind}  {c.name}")
    return 0


# ------------------------------------------------------------------ evaluate
def cmd_evaluate(args: argparse.Namespace) -> int:
    from codeatlas.evaluation import EvalSettings, Evaluation

    if args.allow_unverified_corpus:
        manifest = None
    else:
        manifest = args.manifest
        if not manifest.exists():
            print(
                f"{manifest} not found. Pass --manifest, or --allow-unverified-corpus for a corpus of your own.",
                file=sys.stderr,
            )
            return 2
    per_file: int | str = "auto" if args.per_file == "auto" else int(args.per_file)
    settings = EvalSettings(
        corpus_dir=args.corpus,
        golden_path=args.golden,
        split_path=args.split,
        out_dir=args.out,
        index_dir=args.index,
        manifest_path=manifest,
        chunkings=CHUNKINGS if args.chunkings == "all" else tuple(args.chunkings.split(",")),
        with_rerank=not args.no_rerank,
        per_file=per_file,
        n_resamples=args.bootstrap,
        seed=args.seed,
        repo_dir=Path.cwd(),
    )
    for c in settings.chunkings:
        if c not in CHUNKINGS:
            print(f"unknown chunking {c!r}; expected one of {CHUNKINGS}", file=sys.stderr)
            return 2
    Evaluation(settings).run()
    print(f"wrote {args.out}/metrics.json and companions")
    return 0


def cmd_make_split(args: argparse.Namespace) -> int:
    from codeatlas.evaluation import load_golden, make_split

    _, questions = load_golden(args.golden)
    split = make_split(questions, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: dev {len(split['dev'])}, test {len(split['test'])}")
    return 0


def cmd_render_docs(args: argparse.Namespace) -> int:
    from codeatlas.render import render_all

    return render_all(args.results, args.readme, args.case_study, check=args.check)


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codeatlas", description="Semantic code search and its evaluation harness.")
    parser.add_argument("--version", action="version", version=f"codeatlas {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    corpus = sub.add_parser("corpus", help="fetch, describe or verify the pinned corpus")
    corpus_sub = corpus.add_subparsers(dest="corpus_command", required=True)

    p = corpus_sub.add_parser("fetch", help="clone the projects and check out the pinned commits")
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument(
        "--project", action="append", metavar="NAME=URL@COMMIT", help="pin to use instead of the manifest (repeatable)"
    )
    p.set_defaults(func=cmd_corpus_fetch)

    p = corpus_sub.add_parser("manifest", help="write the manifest of the corpus directory")
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument(
        "--project",
        action="append",
        metavar="NAME=URL@COMMIT",
        help="record this pin instead of reading the checkout (repeatable)",
    )
    p.set_defaults(func=cmd_corpus_manifest)

    p = corpus_sub.add_parser("verify", help="check the corpus directory against the manifest")
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.set_defaults(func=cmd_corpus_verify)

    p = sub.add_parser("build-index", help="chunk the corpus and write index/chunks_<chunking>.jsonl")
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--chunking", choices=("all", *CHUNKINGS), default="all")
    p.set_defaults(func=cmd_build_index)

    p = sub.add_parser("search", help="query a built index")
    p.add_argument("query")
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--chunking", choices=CHUNKINGS, default="structural_merged")
    p.add_argument("--strategy", choices=STRATEGIES, default="hybrid+cap")
    p.add_argument("--per-file", type=int, default=DEFAULT_PER_FILE)
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("evaluate", help="score every configuration and write results/")
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    p.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    p.add_argument("--out", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument(
        "--allow-unverified-corpus", action="store_true", help="skip the manifest check (recorded in metadata.json)"
    )
    p.add_argument("--chunkings", default="all", help="comma-separated subset of " + ",".join(CHUNKINGS))
    p.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder strategies")
    p.add_argument("--per-file", default="auto", help="per-file cap, or 'auto' to choose on the dev half")
    p.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES)
    p.add_argument("--seed", type=int, default=SEED)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("make-split", help="write the provisional dev/test split of the golden set")
    p.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    p.add_argument("--out", type=Path, default=DEFAULT_SPLIT)
    p.add_argument("--seed", type=int, default=SEED)
    p.set_defaults(func=cmd_make_split)

    p = sub.add_parser("render-docs", help="render README tables and docs/case-study.html from results/")
    p.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--readme", type=Path, default=Path("README.md"))
    p.add_argument("--case-study", type=Path, default=Path("docs/case-study.html"))
    p.add_argument("--check", action="store_true", help="fail if the rendered files differ from what is on disk")
    p.set_defaults(func=cmd_render_docs)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
