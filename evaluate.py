"""
Score every strategy on the same 36 questions and write the results out.

Metrics, and why these three:
  recall@k  did any relevant file make the top k. This is the one that matters
            for a RAG system, because the generator only sees what was retrieved.
            recall@5 is the headline: five chunks is a realistic context budget.
  MRR       how far down the first relevant hit was. Two systems can have the
            same recall@10 and be very different to use.
  nDCG@10   rewards putting relevant chunks higher and credits more than one.

Everything is written to results/ so the README's numbers can be regenerated
rather than trusted.
"""
import json
import math
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from retrieve import Index, load_chunks

K_LIST = (1, 3, 5, 10)
DEPTH = 10
OUT = Path("results"); OUT.mkdir(exist_ok=True)
Path("index").mkdir(exist_ok=True)

golden = json.load(open("eval/golden.json", encoding="utf-8"))
QS = golden["questions"]

print("loading models...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def ndcg_at_k(rels, k):
    dcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    ideal = sorted(rels, reverse=True)
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal[:k]))
    return dcg / idcg if idcg else 0.0


def score(ranked_paths, relevant):
    rels = [1 if p in relevant else 0 for p in ranked_paths]
    out = {f"recall@{k}": float(any(rels[:k])) for k in K_LIST}
    first = next((i for i, r in enumerate(rels) if r), None)
    out["mrr"] = 1.0 / (first + 1) if first is not None else 0.0
    out["ndcg@10"] = ndcg_at_k(rels, 10)
    return out


def run(strategy_name, index, fn):
    per_q, rows = [], []
    for q in QS:
        idxs = fn(q["q"])
        paths = [index.chunks[i]["path"] for i in idxs]
        s = score(paths, set(q["relevant"]))
        s.update(id=q["id"], difficulty=q["difficulty"])
        per_q.append(s)
        rows.append({"id": q["id"], "q": q["q"], "difficulty": q["difficulty"],
                     "relevant": q["relevant"],
                     "retrieved": [{"path": index.chunks[i]["path"],
                                    "name": index.chunks[i]["name"],
                                    "kind": index.chunks[i]["kind"],
                                    "hit": index.chunks[i]["path"] in set(q["relevant"])}
                                   for i in idxs]})
    agg = {m: float(np.mean([p[m] for p in per_q]))
           for m in list(per_q[0]) if m not in ("id", "difficulty")}
    agg["strategy"] = strategy_name
    return agg, per_q, rows


results, all_per_q, all_rows = [], {}, {}

for chunking in ("structural_merged", "window", "structural"):
    chunks = load_chunks(chunking)
    print(f"\nindexing {chunking}: {len(chunks)} chunks")
    idx = Index(chunks, embedder=embedder, cache=Path(f"index/emb_{chunking}.npy"))

    strategies = {
        "bm25":   lambda q: idx.bm25_rank(q, DEPTH),
        "dense":  lambda q: idx.dense_rank(q, DEPTH, embedder),
        "hybrid": lambda q: idx.hybrid_rank(q, DEPTH, embedder),
    }
    if chunking in ("structural_merged", "window"):
        strategies["hybrid+rerank"] = lambda q: idx.rerank(q, DEPTH, embedder, cross)

    for name, fn in strategies.items():
        agg, per_q, rows = run(name, idx, fn)
        agg["chunking"] = chunking
        agg["n_chunks"] = len(chunks)
        results.append(agg)
        all_per_q[f"{chunking}/{name}"] = per_q
        all_rows[f"{chunking}/{name}"] = rows
        print(f"  {name:14s} R@1 {agg['recall@1']:.2f}  R@5 {agg['recall@5']:.2f}  "
              f"R@10 {agg['recall@10']:.2f}  MRR {agg['mrr']:.3f}  nDCG@10 {agg['ndcg@10']:.3f}")

json.dump(results, open(OUT / "metrics.json", "w"), indent=2)
json.dump(all_per_q, open(OUT / "per_question.json", "w"), indent=2)
json.dump(all_rows, open(OUT / "retrievals.json", "w"), indent=2)

# ---- breakdown by question difficulty, on the best structural strategies
print("\nby difficulty (structural_merged chunks):")
hdr = f"  {'difficulty':<12s} {'n':>3s}"
for s in ("bm25", "dense", "hybrid", "hybrid+rerank"):
    hdr += f" {s:>14s}"
print(hdr)
diffs = sorted({q["difficulty"] for q in QS})
by_diff = {}
for d in diffs:
    line = f"  {d:<12s} {sum(1 for q in QS if q['difficulty']==d):>3d}"
    by_diff[d] = {}
    for s in ("bm25", "dense", "hybrid", "hybrid+rerank"):
        vals = [p["recall@5"] for p in all_per_q[f"structural_merged/{s}"] if p["difficulty"] == d]
        by_diff[d][s] = float(np.mean(vals))
        line += f" {np.mean(vals):>14.2f}"
    print(line)
json.dump(by_diff, open(OUT / "by_difficulty.json", "w"), indent=2)

# ---- questions the best system still misses
best = all_per_q["structural_merged/hybrid"]
misses = [p["id"] for p in best if p["recall@5"] == 0.0]
print(f"\nmissed at recall@5 by structural_merged/hybrid: {misses if misses else 'none'}")
json.dump(misses, open(OUT / "failures.json", "w"), indent=2)
