# Labelling protocol

This file fixes how questions and answer files are labelled, so that the set
can grow without the rules changing under it. `golden.json` holds the 36
questions of the first round, written and labelled by one person (the author
of the indexed code). `split.json` divides them into a dev half and a test half.

## Files

`golden.json`

```json
{
  "protocol": {"relevance": "...", "authoring": "...", "difficulty": "..."},
  "questions": [
    {
      "id": "tf03",
      "q": "Is there anything stopping one client from hammering the API?",
      "relevant": ["corpus/taskforge/taskforge-api/src/main/java/com/taskforge/api/config/RateLimitFilter.java"],
      "difficulty": "semantic"
    }
  ]
}
```

- `id`: a project prefix (`tf`, `dl`, `gc`, `pf`) and a two-digit number. Ids
  are never reused or renumbered once a question is committed.
- `q`: the question, phrased the way someone new to the repository would ask
  it, without pasting identifiers from the code.
- `relevant`: every file that answers the question, as a logical path
  `corpus/<project>/<path inside the project>`. Paths refer to the commits
  pinned in `results/corpus_manifest.json`.
- `difficulty`: one of `lexical`, `semantic`, `cross_file` (definitions below).

`split.json`

```json
{"method": "...", "seed": 20260917, "dev": ["dl01", "..."], "test": ["dl03", "..."]}
```

The dev half is the only half on which a setting may be chosen (the per-file
cap value is chosen there by `codeatlas evaluate`). Every table reports all
questions with an interval, and the dev/test breakdown of the shipped
configuration is kept in `results/cap_ablation.json`.

## Relevance

Relevance is judged at file level: a retrieved chunk counts if its source file
is in `relevant`. A file is relevant when a reader who opened it, and nothing
else, could answer the question from what it contains. A file that only
mentions the topic, or only calls the code that answers it, is not relevant.
Documentation that answers the question is relevant, and so is the code it
describes; both are listed.

## Difficulty

- `lexical`: at least one relevant file contains the question's key terms
  verbatim (after the tokeniser's stemming, `retries` and `retry` count as the
  same term).
- `semantic`: no relevant file contains the key terms; the match has to come
  from meaning.
- `cross_file`: the answer needs more than one file. A question with two or
  more relevant files is `cross_file` regardless of vocabulary.

The first-round labels do not all follow these definitions (several
`semantic` questions list more than one file). They are left as they are; the
difficulty field is not used in any published table, and it will be relabelled
under these definitions in the second round.

## Second round: two labellers

The second round grows the set to about 100 questions. The steps:

1. Each labeller receives the question text only and lists the relevant files
   independently, browsing the pinned corpus. Labellers do not see retrieval
   results.
2. Agreement is measured per (question, file) pair over the union of the files
   either labeller listed, with Cohen's kappa. The kappa and the raw agreement
   rate are reported in `results/labelling.json` and in the README.
3. Every disagreement is adjudicated by discussion and written to
   `eval/adjudication.jsonl`, one line per pair:

   ```json
   {"id": "tf14", "path": "corpus/taskforge/README.md", "labeller_a": true, "labeller_b": false, "decision": false, "reason": "the README names the queue but does not say how a job reaches it"}
   ```

4. The adjudicated set is frozen in `golden.json`; the split is regenerated
   with `codeatlas make-split` (same seed, stratified by project) and frozen
   in `split.json`.
5. Questions that are answered by the retriever's own documentation, or that
   repeat an identifier from the answer file, are rejected at review.

Until the second round is done, every number in the README rests on 36
questions labelled by one person, and the intervals show how wide that is.
