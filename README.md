# rag-research

A LightRAG-inspired MultiHop-RAG baseline for a master's thesis on agentic
search with persistent graph memory. The current system builds a document
index offline; iterative retrieval and interaction-driven graph updates are
subsequent stages of the thesis.

MultiHop-RAG is the experimental dataset. A Christmas Carol remains a small
build/query demonstration; its custom evaluation pipeline has been removed.
See [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) for the measurement contract.

## Pipeline and layout

Documents → fixed/semantic/agentic chunks → entity/relation extraction → record
merging → KV/vector/graph indexes → retrieval → evaluation or answer generation.

```text
src/rag_research/
├── backends.py             model service adapters
├── llm.py                  explicit completion text, status, and usage
├── chunking/               strategies, source spans, boundary constraints
├── core.py                 construction, retrieval, context serialization
├── extraction.py           extraction validation, retries, and diagnostics
├── evaluation.py           MultiHop-RAG evidence and ranking metrics
├── datasets/               validated dataset loading
├── embedding.py            bounded batching and vector validation
├── models.py               document, evidence, chunk, and build records
├── prompts.py              chunking, extraction, and answer prompts
└── storage.py              atomic JSON/NumPy persistence and graph storage
scripts/
├── build_multihop_index.py
├── evaluate_multihop_retrieval.py
├── run_carol_demo.py
└── visualize_graph.py
tests/                     offline unit and integration checks
data/raw/                  input corpora
artifacts/stores/           version-specific completed indexes
artifacts/cache/            stage caches and extraction attempt history
artifacts/evaluations/      fingerprinted evaluation runs
```

Chunks retain exact source text and document-relative offsets. Chunking and
extraction have separate stage caches. Completed indexes are reused only when
fingerprints match and KV, vector, graph, and source references agree.

## Setup and execution

Requires Python 3.12+ and uv.

```bash
uv sync
cp .env.example .env
```

Place `corpus.json` and `MultiHopRAG.json` from the
[authors' repository](https://github.com/yixuantt/MultiHop-RAG) in
`data/raw/MultiHopRAG/`, or configure `MULTIHOP_DATASET_DIR`.

Select `LLM_BACKEND=ollama` or `LLM_BACKEND=api` and configure the corresponding
model/connection values in `.env`. Embeddings currently use Ollama through
`EMBED_MODEL`. Chunk embedding input follows the indexing information policy
described below; `EMBEDDING_BATCH_SIZE` and `EMBEDDING_CONCURRENCY` bound
embedding work.

```env
MULTIHOP_DATASET_DIR=./data/raw/MultiHopRAG
WORKING_DIR=./artifacts/stores/multihop_fixed_v2
CACHE_DIR=./artifacts/cache/rag_research
EVAL_OUTPUT_DIR=./artifacts/evaluations/multihop_rag
EVAL_MODES=naive,local,global,hybrid
EVAL_K_VALUES=1,3,5,10,20
EVAL_CONCURRENCY=4
EVAL_INCLUDE_NULL=true
EVAL_MAX_QUESTIONS=0
EVAL_TOKENIZER_MODEL=mixedbread-ai/mxbai-rerank-base-v1
```

```bash
uv run python scripts/build_multihop_index.py
uv run python scripts/evaluate_multihop_retrieval.py
```

Use a new `WORKING_DIR` after a build-protocol change. Compatible stage results
can be reused through the shared `CACHE_DIR`. Conflicting existing indexes are
rejected without being overwritten. The evaluator requires a completed index
whose build fingerprint matches the active configuration; build it first.

`EVAL_MAX_QUESTIONS=0` selects all eligible questions; a positive value selects a
source-order prefix for smoke testing, not a random evaluation split.

For a 20-document construction smoke test, use the saved random sample:

```bash
CHUNKING_STRATEGY=agentic uv run python scripts/build_multihop_index.py \
  --corpus data/samples/multihop_20_seed42/corpus.json \
  --working-dir artifacts/stores/multihop_agentic_20_seed42
```

The sample uses seed 42, sampling without replacement and preserving source
order and complete records. Its adjacent `manifest.json` records source/sample
hashes, source indexes, and document URLs. `--corpus` loads only documents and
requires an explicit output directory; the shared `CACHE_DIR` still applies.
The build report marks question metadata as unavailable (`null`). This sample
is for construction, chunk inspection, and runtime checks. For a retrieval
smoke test, sample questions and include all their evidence documents plus
distractors; a smaller retrieval corpus is not comparable to the full benchmark.
The existing evaluator expects a matching full dataset and index.

## Chunking

- All strategies first produce non-overlapping core spans. A shared
  `CHUNK_OVERLAP` postprocess then adds up to that many source characters to the
  beginning of every chunk after the first. The effective overlap can be shorter
  at the document start or for very short core spans; source boundaries remain
  ordered and lossless.
- `fixed`: character core windows controlled by `FIXED_WINDOW_SIZE` (default
  2400), followed by the shared overlap postprocess. Final chunks can therefore
  be longer than the core window by the overlap amount.
- `semantic`: adjacent sentence-context distances must strictly exceed the
  configured percentile plus `1e-12`. Minimum/maximum sentence counts and final
  rebalancing remain enforced. Percentile 100 disables semantic soft cuts.
- `agentic`: sequential source sentence groups are appended to the open chunk
  or begin a new chunk, updating its title and summary. Proposition extraction
  and final metadata refresh can run concurrently; transitions are sequential.

Agentic build logs use the compact form `[agentic] i/N | state j/M | chunks K |
elapsed T | eta E`. `i/N` is corpus progress; `j/M` is transition progress
inside the current document, where `M` is that document's proposition count and
may differ between documents. Document start and completion lines use the same
format, with `start`, `done`, or `cache` in place of the state detail.

Semantic and agentic retain their non-overlapping core spans for boundary
decisions; the shared source-overlap postprocess is applied afterward. The final
stage-specific information policy remains an explicit design decision.

Agentic JSON boundaries require integers, rejecting floats, booleans and
numeric strings. A sentence whose stripped source text exceeds the character
batch limit fails
before the proposition call with its source range; text is never silently
shortened. This precheck excludes prompt overhead and is not a model token-limit
guarantee. Iterative boundary projection preserves the exact objective and tie
rules. Final metadata refresh records final bounds, content, decision source,
and fallback errors. Explicitly truncated completions cannot be accepted as
successful JSON, even if JSON repair could parse them.

## Extraction and diagnostics

Responses may contain at most 20 entities and at most 50 total records
(entities plus relationships). Offline regression examples are checked against
the actual parser. The production extraction request is schema-only: it contains the
contract and an output shape, never realistic few-shot entities,
relationships, or facts. Rich examples remain in a test-only fixture and are
parsed offline as a regression check. Every retry uses the same example-free
contract with an additional correction reminder, so failed attempts cannot
reinforce example copying. Limited Unicode/whitespace/dash normalization
supports surface variants without replacing validation with semantic similarity.
The extraction pipeline version and fingerprint include this policy; existing
extraction caches are therefore not reused silently after the change.

Attempts retain raw counts, errors, completion reasons, known/unknown truncation
status, reported token usage, and retry identities. Current, historical and
cumulative summaries distinguish fresh calls from cached work. An exhausted
final attempt is not counted as a retry that never occurred. Missing completion
metadata is unknown, not zero truncation.

Complete diagnostics are saved to `WORKING_DIR/extraction_report.json`, even
when extraction returns failed chunks. The manifest holds compact summaries;
cache ledgers and per-run reports preserve earlier attempts across resumption.
Fully reusing an index makes no new extraction calls and retains its original
build statistics. See [rate definitions](EXPERIMENT_PROTOCOL.md#extraction-observability).

## Retrieval and evaluation

- `naive`: dense chunk retrieval.
- `local`: dense entity retrieval, adjacent relations, and source chunks.
- `global`: dense relation retrieval, endpoint entities, and source chunks.
- `hybrid`: deduplicated local/global candidates followed by chunk ranking.

`ENABLE_RERANKER=true` enables CrossEncoder reranking for relation and chunk
shortlists. In graph modes, relation reranking can also change the downstream
chunk candidate set; this is a system comparison, not a chunk-only ablation.

Each question/mode retrieves once at `max(max(EVAL_K_VALUES), 10)`. Requested
prefixes are scored using strict evidence containment and an additional,
separately named cross-chunk union diagnostic. Unreachable facts remain in the
recall denominator; source/hash/offset corruption still raises an error.
Official Hits/MAP/MRR keep the authors' text-matching rules and top-10 protocol.
Null queries have context statistics only and are excluded from relevance means.

Runs contain `run_manifest.json`, per-question checkpoints, `results.json`,
`summaries.json`, and upstream-format `official/<mode>.json`. Fingerprints include
schema, code hashes, selected questions, index, and retrieval configuration.
Schema-2 checkpoints are incompatible with the revised schema-3 metrics.

Chunk embedding and graph extraction both receive the source chunk and the
dataset-provided title, author, date, and source. Agentic-generated title and
summary remain persisted chunk metadata for later use, but are excluded from
both inputs. Other dataset metadata, such as category, remains on the record
without being sent to either model. Document ID, URL, and source offsets remain
on the chunk record for identity, provenance, and citation. Chunk IDs are derived
from document identity, chunk position, source offsets, and source text only.
The embedding adapter adds `search_document: ` to indexed documents and
`search_query: ` to retrieval queries. Semantic boundary embeddings use
`clustering: `. These prefixes are added only at the embedding request boundary
and are not written into `model_text` or the answer context.
Entity vectors use name, type, and description. Relation vectors always use
both endpoint names, keywords, and description, including when the description
or keyword list is empty.
Ollama embedding requests set `truncate=false`; an input that exceeds the
backend context is rejected and reported with its purpose and record IDs rather
than silently losing source text.
Answer context is unchanged:
it contains the original chunk text and retrieved entity/relation descriptions,
without directly serializing chunk title/summary. The public `build_context`
serializer is shared by generation and evaluation token measurement. Explicitly
truncated answers raise an error.

## Development demo and checks

Use a separate store for the retained Carol demo:

```bash
WORKING_DIR=./artifacts/stores/carol_demo_v2 uv run python scripts/run_carol_demo.py
uv run python scripts/visualize_graph.py
PYTHONDONTWRITEBYTECODE=1 uv run python -B -m unittest discover -s tests -v
```

Tests use temporary stores and fake model responses to verify parser contracts,
boundaries, failure recovery, cache history, metrics, and checkpoint/export
behavior without making model requests.

## License

MIT; see [LICENSE](LICENSE).
