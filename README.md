# CodeAtlas

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/sentence--transformers-FFCA28?style=flat-square" alt="sentence-transformers" />
  <img src="https://img.shields.io/badge/Tree--sitter-6E4C13?style=flat-square" alt="Tree-sitter" />
  <img src="https://img.shields.io/badge/BM25-4B5563?style=flat-square" alt="BM25" />
</p>

Semantic search over a codebase, and an evaluation harness that says how well it
actually works.

The retrieval part is a weekend. The reason this repo exists is the second part:
36 questions with labelled answers, four retrieval strategies and three chunking
strategies scored against them, and the failures written down instead of rounded
off. Every number below is read from `results/`, which is committed, and
regenerates from `python build_index.py && python evaluate.py` against the same
corpus.

**Corpus:** 152 files and roughly 11,000 lines across four of my own projects,
Python, Java, JavaScript and Markdown. Using my own code means I can label
ground truth honestly, and it means every claim here is checkable against the
repos it indexes.

---

## Results

Best configuration: structural chunking merged to a minimum size, hybrid
retrieval, one chunk per file.

| | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| **structural-merged + hybrid + per-file cap** | **0.42** | **0.78** | **0.86** | **0.92** | **0.591** |

Relevance is judged at file level: a retrieved chunk counts if its source file is
one of the labelled answers. That is coarser than passage-level judging and it is
reproducible, which matters more on a set this size.

### Every configuration

| Chunking | Strategy | Chunks | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|---|
| structural-merged | bm25 | 371 | 0.14 | 0.44 | 0.58 | 0.256 | 0.342 |
| structural-merged | dense | 371 | 0.39 | 0.78 | 0.78 | 0.554 | 0.577 |
| structural-merged | **hybrid** | 371 | **0.42** | **0.81** | 0.89 | **0.554** | **0.618** |
| structural-merged | hybrid + rerank | 371 | 0.36 | 0.78 | 0.89 | 0.520 | 0.577 |
| window | bm25 | 403 | 0.11 | 0.50 | 0.67 | 0.246 | 0.362 |
| window | dense | 403 | 0.36 | 0.81 | 0.83 | 0.548 | 0.594 |
| window | hybrid | 403 | 0.31 | 0.81 | 0.89 | 0.515 | 0.596 |
| window | hybrid + rerank | 403 | 0.31 | 0.72 | 0.81 | 0.458 | 0.537 |
| structural (unmerged) | bm25 | 819 | 0.08 | 0.53 | 0.64 | 0.251 | 0.349 |
| structural (unmerged) | dense | 819 | 0.36 | 0.75 | 0.81 | 0.503 | 0.561 |
| structural (unmerged) | hybrid | 819 | 0.25 | 0.72 | 0.89 | 0.472 | 0.563 |

---

## Three things the numbers said that I did not expect

### 1. My chunking hypothesis was wrong the first time, because the experiment was wrong

The idea was that splitting code on declaration boundaries with Tree-sitter should
beat sliding a fixed window, because a window cuts functions in half and half a
function retrieves badly.

The first run said otherwise. Structural chunking scored R@5 0.72 against the
window baseline's 0.81. The clean conclusion would have been that the idea does
not work.

It was not a clean experiment. Structural chunking produced 819 chunks at a median
of 9 lines; the window baseline produced 403 at a median of 40. Two variables moved
at once, so the result could not separate *boundaries in the wrong place* from
*chunks too small to carry meaning*. A 9-line method embedded alone is mostly a
signature.

Merging adjacent declarations up to a 25-line minimum, without ever splitting one,
gives 371 chunks at a median of 36 lines. That is the same shape as the baseline
and only the boundary policy differs:

| At matched chunk size | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| structural-merged, hybrid | **0.42** | 0.81 | 0.89 | **0.554** | **0.618** |
| window, hybrid | 0.31 | 0.81 | 0.89 | 0.515 | 0.596 |

**Boundaries help ranking, not recall.** The right file lands in the top 5 either
way. Structural boundaries make it more likely to be *first*, which is what matters
when the context budget is small. And chunk size mattered more than boundary
policy: getting size wrong wiped out the boundary gain and then some.

### 2. The cross-encoder reranker made things worse

Reranking the top 30 with `ms-marco-MiniLM-L-6-v2` is close to standard advice.
Here it cost MRR on both chunkings, 0.554 to 0.520 on structural-merged and 0.515
to 0.458 on windows.

That model is trained on MS MARCO, which is web passages in natural language. Code
is out of distribution for it, and a reranker that is confidently wrong is worse
than no reranker, because it reorders results that the first stage already had
roughly right. A code-trained reranker would probably help; this one does not, and
shipping it because the architecture diagram looks better would be the wrong call.

### 3. BM25 is much weaker here than the literature suggests, and that is a property of the questions

BM25 tops out at R@5 0.44 to 0.53. Hybrid retrieval still beats dense alone on
ranking metrics, so the lexical signal is worth keeping, but it is not carrying the
system.

This is a direct consequence of how the questions were written. They are phrased
the way someone new to the repo would ask, "what stops the relays flickering when
my hand shakes slightly", not "find the jitter threshold". If the questions had
reused the codebase's identifiers, BM25 would have looked excellent and the
evaluation would have measured nothing.

---

## Where it fails

Seven of 36 questions still miss at rank 5. Reading them produced two failure
modes, and both turned out to be measurable rather than impressionistic:

| | on misses | on hits |
|---|---|---|
| share of top-5 that is documentation | **0.46** | 0.29 |
| share of top-5 from the wrong project | **0.29** | 0.12 |

**Documentation crowds out code.** READMEs are written in the same register as the
questions, so they sit closer in embedding space than the code that implements the
answer. Ask "where is job state stored so the API and the worker both see it" and
all five results are README sections, while `ReportJobRepository.java` sits below
the cut.

**Nothing scopes the query to a project.** The corpus holds four separate projects.
"Is anything stopping one client from hammering the API" is a reasonable question
about two of them, and the retriever has no way to know which.

There is also a third mode that no amount of ranking fixes: **enumeration questions.**
"What kinds of report can this produce" needs the complete set across four files.
Retrieval is a similarity function, not an aggregation, and asking it to return an
exhaustive list is asking the wrong thing of it.

### The fix, and what it bought

The first failure mode has a small fix: cap how many chunks any single file may
contribute, so one long README cannot occupy the whole context window. Overflow is
pushed below the survivors rather than discarded.

| | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| hybrid | 0.42 | 0.67 | 0.81 | 0.89 | 0.554 |
| hybrid + per-file cap 2 | 0.42 | 0.72 | 0.81 | 0.89 | 0.560 |
| **hybrid + per-file cap 1** | 0.42 | **0.78** | **0.86** | **0.92** | **0.591** |

About ten lines, +0.11 R@3 and +0.037 MRR, and it came out of reading the failures
rather than from tuning. That is the argument for building the harness first.

---

## How it works

```
files ──► chunker ──┬─► BM25 index          ┐
                    └─► MiniLM embeddings   ├─► RRF fusion ─► per-file cap ─► top k
                                            ┘
```

**Chunking.** Tree-sitter parses Python and Java and splits on declaration
boundaries: one chunk per function, method or class, with the enclosing scope
recorded in the chunk's name. Markdown splits on headings, and the heading path is
prefixed onto the chunk text, so a section called "Retries" nested under "Fault
Tolerance" is retrievable by either term. Adjacent declarations are then merged up
to a 25-line minimum. Anything outside a declaration, imports and module-level
constants, becomes a preamble chunk rather than being dropped, because "what is the
retry limit" often needs exactly that.

**Tokenization for BM25** splits identifiers as well as words, so `RateLimitFilter`
is findable by "rate limit".

**Fusion** is Reciprocal Rank Fusion rather than a weighted score blend. BM25 scores
and cosine similarities are not on a comparable scale, and normalising them adds a
weight nobody can tune honestly against 36 questions.

**Embeddings** prefix the file path and declaration name onto the chunk text before
encoding. The name is signal: `RelayController.set_relay` carries meaning the body
alone may not.

---

## Running it

```bash
pip install -r requirements.txt
python build_index.py        # corpus/ -> index/
python evaluate.py           # -> results/metrics.json, per_question.json, retrievals.json
python analyze_failures.py   # -> results/failure_analysis.json
```

`SETUP.md` covers pointing it at a different codebase.

Everything in `results/` is committed, so every claim here is checkable without
running anything. The `corpus/` itself is not committed, since it is four other
repositories vendored in; `SETUP.md` explains how to reassemble it or point the
harness at your own code. If a number in this README and a number in `results/`
disagree, the files are right and this document is stale.

---

## What I would do differently

- **36 questions is a small set.** A five-point difference between two strategies
  is within noise at this size, which is why the conclusions above lean on the
  differences that are large and consistent across chunkings rather than on any
  single cell. Two hundred questions would let me put confidence intervals on this
  and I would trust the reranker result more.
- **I wrote the questions and I wrote the code being searched.** That is the
  cleanest way to get honest labels at this scale and it is also the obvious bias:
  I know what the answer files are called. Questions written by someone who has not
  seen the codebase would be harder and more realistic.
- **File-level relevance is generous.** A chunk from the right file counts even if
  it is the wrong function in that file. Passage-level judgements would be stricter
  and would probably lower every number here.
- **No generation, so no faithfulness measurement.** This evaluates retrieval only.
  Retrieval quality bounds answer quality, so it is the right thing to fix first,
  but "did the model's answer follow from the retrieved context" is a separate
  measurement I have not made.
- **Project routing is the obvious next fix.** The wrong-project rate more than
  doubles on failures. A cheap classifier over the query, or simply a project
  filter in the UI, addresses more of the remaining gap than more ranking work
  would.
