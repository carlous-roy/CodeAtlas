"""
Turn the misses into two measured failure modes, then test a fix against them.

Reading the seven failures by hand, two patterns account for all of them:

  1. Documentation crowds out code. READMEs are written in the same register as
     the questions, so they sit closer in embedding space than the code that
     actually implements the answer. Five of the seven misses carry at least one
     documentation chunk in their top 5, and four of them rank one first, while
     the implementing file sits below the cut. The doc_share figures below
     measure this rather than asserting it.

  2. No project scoping. The corpus holds four separate projects. "Is anything
     stopping one client from hammering the API" is a reasonable question about
     TaskForge and an equally reasonable question about DiffLens, and nothing in
     the retriever knows which one is being asked about.

Both are quantified below rather than asserted, and then one fix is applied and
re-measured. The fix is deliberately the simplest thing that addresses mode 1:
cap how many chunks any single file may contribute to the result set, so a long
README cannot occupy the whole context window.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from retrieve import Index, load_chunks

OUT = Path("results")
golden = json.load(open("eval/golden.json", encoding="utf-8"))
QS = golden["questions"]
K_LIST = (1, 3, 5, 10)
DEPTH = 10

DOC_SUFFIXES = (".md", ".txt")


def project_of(path: str) -> str:
    return path.split("/")[1] if path.startswith("corpus/") else "?"


# ---------------------------------------------------------------- quantify
rows = json.load(open(OUT / "retrievals.json"))["structural_merged/hybrid"]
per = {p["id"]: p for p in json.load(open(OUT / "per_question.json"))["structural_merged/hybrid"]}

doc_share, wrong_project, dup_file = [], [], []
for r in rows:
    top5 = r["retrieved"][:5]
    doc_share.append(sum(1 for x in top5 if x["path"].endswith(DOC_SUFFIXES)) / max(len(top5), 1))
    want_projects = {project_of(p) for p in r["relevant"]}
    wrong_project.append(sum(1 for x in top5 if project_of(x["path"]) not in want_projects) / max(len(top5), 1))
    c = Counter(x["path"] for x in top5)
    dup_file.append(max(c.values()) if c else 0)

missed = [r for r in rows if per[r["id"]]["recall@5"] == 0.0]
hit = [r for r in rows if per[r["id"]]["recall@5"] == 1.0]


def mean_doc(rs):
    return float(np.mean([sum(1 for x in r["retrieved"][:5] if x["path"].endswith(DOC_SUFFIXES)) / 5
                          for r in rs])) if rs else 0.0


def mean_wrongproj(rs):
    out = []
    for r in rs:
        want = {project_of(p) for p in r["relevant"]}
        out.append(sum(1 for x in r["retrieved"][:5] if project_of(x["path"]) not in want) / 5)
    return float(np.mean(out)) if out else 0.0


diag = {
    "questions": len(rows),
    "missed_at_5": len(missed),
    "doc_share_top5_all": round(float(np.mean(doc_share)), 3),
    "doc_share_top5_on_misses": round(mean_doc(missed), 3),
    "doc_share_top5_on_hits": round(mean_doc(hit), 3),
    "wrong_project_share_top5_all": round(float(np.mean(wrong_project)), 3),
    "wrong_project_share_top5_on_misses": round(mean_wrongproj(missed), 3),
    "wrong_project_share_top5_on_hits": round(mean_wrongproj(hit), 3),
    "max_chunks_from_one_file_top5_mean": round(float(np.mean(dup_file)), 2),
}
print("failure diagnostics (structural_merged / hybrid, top 5):")
for k, v in diag.items():
    print(f"  {k:42s} {v}")

# ---------------------------------------------------------------- the fix
print("\napplying per-file cap and re-measuring...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
chunks = load_chunks("structural_merged")
idx = Index(chunks, embedder=embedder, cache=Path("index/emb_structural_merged.npy"))


def capped(q, k=DEPTH, per_file=2, depth=60):
    """Same hybrid ranking, but no single file may contribute more than
    `per_file` chunks to the result set. Overflow is not discarded, it is pushed
    below everything that survives the cap, so nothing is lost from the tail."""
    ranked = idx.hybrid_rank(q, depth, embedder)
    seen, keep, overflow = Counter(), [], []
    for i in ranked:
        p = idx.chunks[i]["path"]
        if seen[p] < per_file:
            seen[p] += 1
            keep.append(i)
        else:
            overflow.append(i)
        if len(keep) >= k:
            break
    return (keep + overflow)[:k]


def evaluate(fn, label):
    recs = {f"recall@{k}": [] for k in K_LIST}
    mrrs = []
    for q in QS:
        paths = [idx.chunks[i]["path"] for i in fn(q["q"])]
        rel = set(q["relevant"])
        flags = [p in rel for p in paths]
        for k in K_LIST:
            recs[f"recall@{k}"].append(float(any(flags[:k])))
        first = next((i for i, f in enumerate(flags) if f), None)
        mrrs.append(1.0 / (first + 1) if first is not None else 0.0)
    out = {m: round(float(np.mean(v)), 3) for m, v in recs.items()}
    out["mrr"] = round(float(np.mean(mrrs)), 3)
    out["strategy"] = label
    return out


base = evaluate(lambda q: idx.hybrid_rank(q, DEPTH, embedder), "hybrid")
fix2 = evaluate(lambda q: capped(q, per_file=2), "hybrid + per-file cap 2")
fix1 = evaluate(lambda q: capped(q, per_file=1), "hybrid + per-file cap 1")

print(f"\n  {'strategy':<26s} {'R@1':>6s} {'R@3':>6s} {'R@5':>6s} {'R@10':>6s} {'MRR':>7s}")
for r in (base, fix2, fix1):
    print(f"  {r['strategy']:<26s} {r['recall@1']:>6.2f} {r['recall@3']:>6.2f} "
          f"{r['recall@5']:>6.2f} {r['recall@10']:>6.2f} {r['mrr']:>7.3f}")

json.dump({"diagnostics": diag, "ablation": [base, fix2, fix1]},
          open(OUT / "failure_analysis.json", "w"), indent=2)
print("\nwrote results/failure_analysis.json")
