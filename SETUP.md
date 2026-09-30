# Running CodeAtlas

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock      # exact versions, CPU-only torch
pip install -e .
```

Python 3.11 or newer. The two models (`all-MiniLM-L6-v2` for embeddings,
`ms-marco-MiniLM-L-6-v2` for the reranker) download from the Hugging Face hub
at pinned revisions on first use, about 180 MB together, and run on CPU.

## Reproduce the published numbers

```bash
codeatlas corpus fetch     # clones the four projects at the pinned commits into corpus/
codeatlas corpus verify    # optional: compares corpus/ with results/corpus_manifest.json
codeatlas evaluate         # writes results/; about 6 minutes on two CPU cores
codeatlas render-docs      # rerenders the README tables and docs/case-study.html
```

`codeatlas evaluate` verifies the corpus against the manifest before it starts
and stops with a list of differences if anything is missing, extra or changed.
Results are written after each chunking, so an interrupted run leaves the
configurations it finished. `--no-rerank` skips the cross-encoder strategies;
`--chunkings structural_merged` limits the run to one chunking; `--per-file N`
fixes the cap instead of choosing it on the dev half.

Two runs on the same machine produce identical `results/` files apart from
`generated_at`, `duration_seconds` and `codeatlas_commit` in `metadata.json`;
`codeatlas digest` prints a sha256 per file with those fields left out, so two
runs can be compared with `diff`.

## Point it at your own codebase

Put the code under `corpus/`, one directory per project, and write a manifest
for it so that later runs can prove they indexed the same files:

```
corpus/
  my-service/
  my-frontend/
```

```bash
codeatlas corpus manifest --corpus corpus --out results/corpus_manifest.json
```

If a project directory is a git checkout, its remote URL and commit are
recorded; otherwise those fields are `null`. `corpus/` and `index/` are
gitignored.

Chunk paths are logical: `corpus/<project>/<path inside the project>`,
regardless of where the directory lives on disk. Tree-sitter parses `.py` and
`.java`; Markdown splits on headings; every other indexed extension (`.js`,
`.jsx`, `.ts`, `.tsx`, `.yml`, `.yaml`, `.txt`, `.sql`, `.dax`) is chunked by
windows. Adding a language means adding its extension to `CODE_EXT` and its
grammar and declaration node types to `LANGUAGES` in `codeatlas/chunking.py`.

## Write a golden set

`eval/golden.json` is the part that takes real time and the part worth doing.
Each entry is a question plus the files that answer it:

```json
{"id": "svc01",
 "q": "What happens to a job that keeps failing and never succeeds?",
 "relevant": ["corpus/my-service/worker/JobProcessor.java"],
 "difficulty": "semantic"}
```

Two rules decide whether the evaluation measures anything:

1. Phrase questions the way someone new to the repository would ask. "Find the
   JobProcessor retry limit" tests string matching against a filename.
2. Label every file that genuinely answers the question, not just the first one
   you thought of. Under-labelling shows up as false failures.

`eval/LABELLING.md` has the full protocol, including the difficulty definitions
and the two-labeller procedure. Then split the set and evaluate:

```bash
codeatlas make-split --golden eval/golden.json --out eval/split.json
codeatlas evaluate --golden eval/golden.json --split eval/split.json
```

## Use the retriever

```bash
codeatlas build-index                                   # corpus/ -> index/chunks_*.jsonl
codeatlas search "how are retries backed off?" -k 5     # merged chunking, hybrid + per-file cap
codeatlas search "rate limiting" --strategy dense --chunking window
```

From Python:

```python
from pathlib import Path

from codeatlas.chunking import read_chunks
from codeatlas.models import load_embedder
from codeatlas.retrieval import Index

chunks = read_chunks(Path("index/chunks_structural_merged.jsonl"))
index = Index.build(chunks, load_embedder(), cache_dir=Path("index"), label="structural_merged")
for i in index.search("how are retries backed off?", "hybrid+cap", k=5):
    c = index.chunks[i]
    print(f"{c.path}:{c.start_line}-{c.end_line}  {c.name}")
```

`Index.search` accepts `bm25`, `dense`, `hybrid`, `hybrid+cap`,
`hybrid+rerank` and `hybrid+rerank+prefix` (the last two need
`reranker=load_reranker()` when building the index). Embeddings are cached
under `cache_dir` keyed on a hash of the chunk payloads and the model revision.

## Tests and lint

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest
```

The tests use a five-file fixture corpus under `tests/fixtures/` and need no
model download. CI additionally runs `codeatlas evaluate` on that fixture with
both models and checks that the rendered docs match `results/`.
