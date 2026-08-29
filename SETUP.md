# Running CodeAtlas on your own codebase

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

First run downloads two models from Hugging Face, about 120MB total:
`all-MiniLM-L6-v2` for embeddings and `ms-marco-MiniLM-L-6-v2` for the reranker
ablation. Both run on CPU.

## Point it at a codebase

Put the code under `corpus/`, one directory per project:

```
corpus/
  my-service/
  my-frontend/
```

`corpus/` is gitignored, so nothing you index gets committed by accident.

Then:

```bash
python build_index.py
```

This writes three chunkings to `index/`: `structural`, `structural_merged` and
`window`. Extending it to another language means adding the Tree-sitter grammar to
`LANGS` and its declaration node types to `DECL_NODES` in `build_index.py`.
Anything without a grammar falls back to windows automatically, so unsupported
files degrade rather than break.

## Write a golden set

`eval/golden.json` is the part that takes real time, and it is the part worth
doing. Each entry is a question plus the files that answer it:

```json
{"id": "svc01",
 "q": "What happens to a job that keeps failing and never succeeds?",
 "relevant": ["corpus/my-service/worker/JobProcessor.java"],
 "difficulty": "semantic"}
```

Two rules that decide whether the evaluation measures anything:

1. **Phrase questions the way someone new to the repo would ask.** If you write
   "find the JobProcessor retry limit" you are testing string matching against a
   filename, and BM25 will look spectacular for no reason.
2. **Label every file that genuinely answers it**, not just the one you thought of
   first. Under-labelling shows up as false failures, and you will waste an
   afternoon debugging a retriever that was right.

Mark each question `lexical`, `semantic` or `cross_file` so the breakdown by
difficulty means something.

## Evaluate

```bash
python evaluate.py           # all chunkings x all strategies
python analyze_failures.py   # failure diagnostics and the per-file cap ablation
```

Results land in `results/`. `retrievals.json` holds the full top-10 for every
question with hit/miss flags, which is what you read when a number moves and you
want to know why.

## Using the retriever

```python
from sentence_transformers import SentenceTransformer
from retrieve import Index, load_chunks

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
idx = Index(load_chunks("structural_merged"), embedder=embedder)

for i in idx.hybrid_rank("how are retries backed off?", k=5, embedder=embedder):
    c = idx.chunks[i]
    print(f"{c['path']}:{c['start_line']}  {c['name']}")
```

Add the per-file cap from `analyze_failures.py` if you are feeding a context
window rather than showing a list to a human. It was worth +0.11 R@3 here.
