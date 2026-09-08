# Baseline experiment protocol

MultiHop-RAG is the experimental dataset. The current system is an offline
LightRAG-inspired baseline for subsequent agentic retrieval and online graph
memory. Evidence retrieval metrics do not measure final answer correctness.

## Evidence and dataset validity

Facts belong to specific documents and may have several source occurrences.
Intervals are zero-based and half-open: `[start, end)`. Document IDs, source
slices, metadata, hashes, and offsets are checked independently of retrieval.

A fact absent from all individual chunks is an index-coverage limitation, not
corrupt benchmark data. It remains in the gold set and in recall denominators.
Questions are not dropped merely because chunking makes their evidence
unreachable. Null queries have no evidence target: report context statistics
separately and exclude them from relevance and official means.

## Strict extended metrics

A chunk covers a fact if it completely contains one occurrence in the correct
document. Duplicate chunk IDs do not create additional ranked items. Distinct
overlapping chunks remain separate items and consume context.

For a question with gold facts `G`, requested cutoff `K`, and returned prefix `R`:

- `evidence_recall = covered facts / |G|`.
- `joint_evidence_success`: every fact in `G` is covered.
- `chunk_precision = relevant returned chunks / K`, including when fewer than
  K chunks are returned.
- `precision_among_returned = relevant returned chunks / |R|`, or zero for an
  empty list. This is deliberately separate from P@K.
- `coverage_f1` is the harmonic mean of chunk P@K and evidence recall. It mixes
  chunk and fact units, so treat it as an auxiliary composite, not a conventional
  same-unit F1 score or the primary multi-hop result.
- MRR uses the first relevant chunk. Extended AP@K and nDCG@K use binary chunk
  relevance and are distinct from the official AP formula.
- Document recall and joint document success use returned document IDs. An
  irrelevant chunk from a gold document can satisfy document recall without
  satisfying evidence recall.

Macro metrics average question-level values. Micro P@K divides total relevant
chunks by the sum of requested K. Micro evidence recall divides total covered
facts by total gold facts. Micro precision-among-returned instead divides by the
actual number of returned chunks.

Groups use question type, evidence count, and unique evidence-document count.
Evidence count is not a graph traversal depth or an annotated reasoning path.

## Index coverage ceilings

For each answerable question, record reachable and unreachable evidence IDs.
The strict evidence-recall ceiling is the reachable fraction; the strict joint
ceiling is one only when all facts are reachable. Aggregate these independently
of retrieval mode and K. These ceilings assume access to all indexed chunks;
they do not promise achievable scores at a finite K.

Keep the main results on all selected questions. Coverage ceilings explain
failure sources; they do not justify replacing results with scores on an easier
fully reachable subset.

## Cross-chunk union diagnostic

`cross_chunk_union_evidence_recall` merges returned intervals separately within
each document. A fact is covered only when that union completely covers one
specific occurrence. Contiguous intervals can join; gaps, different occurrences,
and different documents cannot be stitched together. The corresponding joint
score requires every fact.

This diagnostic explains boundary-split evidence. It changes neither the strict
metric nor the official metric and does not establish successful QA reasoning.

## Information passed to indexing models

Chunk embedding and graph extraction receive the verbatim source chunk together
with dataset-provided title, author, date, and source fields. Agentic-generated
chunk titles and summaries are retained as metadata for later context design,
but are excluded from both indexing inputs. Other dataset metadata, such as
category, remains persisted without being sent to either model. Document ID, URL,
and document-relative source offsets remain persisted on every chunk for identity
and provenance. Chunk IDs are derived only from document identity, chunk position,
source offsets, and source text. Answer-context serialization is specified
separately and is not changed by this policy. The embedding adapter adds
`search_document: ` to indexed documents, `search_query: ` to retrieval
queries, and `clustering: ` to semantic boundary inputs. Prefixes are applied
only at the embedding request boundary, never to persisted model text.
Entity vectors contain the entity name, type, and description. Relation vectors
contain both endpoint names, keywords, and description, so relation identity is
preserved even when descriptive fields are empty.
Ollama requests disable truncation. Context overflow therefore fails the
embedding operation and reports the affected purpose and record IDs; no vector
is created from a silently shortened input.

## Density and context size

All strategies choose non-overlapping core source spans first. The common
overlap postprocess then extends each span after the first backwards by up to
the configured number of source characters; the first chunk stays anchored at
zero and starts remain strictly increasing. Report the requested overlap from
the build provenance and the effective per-boundary overlaps from the chunk
audit trace. This is a maximum character overlap, so very short spans may
receive a shorter effective prefix.

For each document, intersect the union of its gold occurrence intervals with
the union of its retrieved intervals. Sum these intersection lengths across
documents, then divide by the sum of all returned chunk-text lengths.

The numerator counts each covered source character once, including partial
facts; nested facts do not double-count characters. The denominator counts
overlap repeated in distinct returned chunks repeatedly. Equal text at different
source positions is not one source interval. Density describes character
concentration, not the proportion of complete facts.

- `chunk_source_tokens`: original returned chunk texts joined by two newlines,
  tokenized without special tokens.
- `generation_context_tokens`: tokenization of `LightRAG.build_context` for the
  returned entities, relations, and the same chunk prefix.

The latter includes graph descriptions and serialization but excludes the user
question, surrounding answer instructions, provider chat framing, and output.
It is a context diagnostic, not billed usage or latency. Evaluation does not
make a generation call. Keep the tokenizer fixed across comparisons; it may
differ from the provider's tokenizer.

The baseline reports K curves and context sizes without adding a cross-product
of token-budget experiments. These are not equal-token-budget comparisons.

## Official metrics

Preserve the [authors' evaluator](https://github.com/yixuantt/MultiHop-RAG/blob/main/retrieval_evaluate.py):
remove literal spaces/newlines, determine relevance by normalized gold-fact
substring containment, score top 10, exclude null queries, and aggregate
`Hits@4`, `Hits@10`, `MAP@10`, and `MRR@10` using upstream formulas.

`official/<mode>.json` retains the `query` / `retrieval_list` / `gold_list`
structure for independent rescoring. The document-relative extension and the
literal-text official protocol can disagree; retain both definitions.

## Extraction observability

The per-response contract is 20 entities and 50 total records, checked before
duplicate merging. These are limits, not extraction targets. Reaching a limit
does not prove that additional facts were omitted.

Each attempt retains identity, run identity, raw counts when observable,
completion reason, truncation status, reported usage, outcome/error, and the
previous attempt identity for an actual retry. A started request whose result
was not received stays unresolved, not falsely successful.

Raw counts are observed from strict JSON before record normalization and
deduplication, with `raw_count_source = strict_json`. If JSON repair is needed,
raw counts remain unknown even when the repaired response passes validation.
Accepted counts describe the validated, deduplicated records from successful
attempts, and are reported separately.

Summaries separate the current invocation, previous invocations, and cumulative
history. Cached success retains the attempts used to produce it; failed attempts
also have persistent ledgers. Reusing an index or chunk is not a new request.

Reports expose counts and denominators for:

- Responses reaching entity/total limits, alongside chunk-level indicators and
  accepted-output observations.
- Over-limit attempts, affected chunks, and actual retries following an
  over-limit response. An exhausted final attempt is not a nonexistent retry.
- Explicitly truncated responses over responses with known truncation status,
  plus unknown-status counts and observation coverage. Malformed JSON is not a
  truncation detector; plain-string callables have unknown status.

Rates with no eligible observations are `null`. Provider token usage is distinct
from token estimates over source text. Explicit output-limit completion cannot
be repaired into a successful extraction or agentic decision merely because
its text is parseable JSON.

Within each of `run_summary`, `historical_summary`, and `cumulative_summary`:

| Statistic | Field and denominator |
| --- | --- |
| Entity / total cap rate | `entity_limit` / `total_limit`: `at_or_above_limit_rate_per_known_attempt`, over attempts with observable raw counts |
| Chunk-level cap rate | Those same groups: `at_or_above_limit_rate_per_known_chunk`, over chunks with at least one observable count |
| Accepted-output cap rate | `accepted_entity_limit` / `accepted_total_limit`, using successful validated outputs |
| Over-limit incidence | `over_limit_rate_per_known_attempt`, over attempts with a decidable raw-limit result |
| Actual over-limit retry rate | `over_limit_retry_rate_per_violation`, over observed over-limit attempts |
| Output truncation rate | `output_truncation_rate_per_known_response`, over responses with explicit completion status |
| Missing completion status | `truncation_unknown_rate_per_response`, over received responses |
| Started calls without a resolved outcome | `unresolved_attempt_count`; retained from the attempt ledger |

Each cap group also exposes exact-at-limit counts and observation counts.
At-or-above includes violations; exact-at-limit does not. These observations
measure saturation of the configured contract, not semantic extraction recall.

`extraction_report.json` links full diagnostics to build/extraction fingerprints.
The build manifest holds compact summaries; cache run reports retain invocation
history. Reports cannot establish the outcome of a server call interrupted
before the client receives completion metadata. Missing historical metadata
cannot be reconstructed from an existing graph.

## Versions and resumption

Evaluation schema 3 changes definitions and field names. Schema-2 checkpoints
must not be silently combined with schema-3 rows. Runs identify dataset, selected
questions, code hashes, retrieval settings, and index; per-question/mode records
are the source of final summaries.

Extraction-contract changes invalidate extraction and the final index while
compatible chunk caches remain reusable. Semantic and agentic have explicit
strategy versions for their revised boundary behavior. Existing indexes are not
overwritten. Passing offline tests verifies implementation, not benchmark
performance; real scores require model-backed evaluation of a frozen baseline.
