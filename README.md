# rag-research

Research code and experiment artifacts for a thesis project on retrieval-augmented generation with graph-structured memory. The project started as a LightRAG replication, but it is now used as a broader experimental workspace for comparing chunking strategies, graph-based retrieval behavior, and evaluation traces.

## Current Focus

- Build a lightweight RAG pipeline with KV storage, vector indexes, and a knowledge graph.
- Compare fixed-size, embedding-based semantic, and stateful agentic chunking.
- Evaluate single- and multi-document retrieval across naive, local, global, and hybrid modes.
- Use MultiHopRAG for cross-document evidence and joint multi-hop retrieval evaluation.

## Pipeline

1. **Chunking** ([chunking.py](src/rag_research/chunking.py))
   Splits documents with one of three strategies:
   - `fixed`: character-based sliding windows.
   - `semantic`: sentence-level semantic boundary detection using embeddings.
   - `agentic`: proposition-aware, stateful semantic chunk management.

2. **Extraction** ([extraction.py](src/rag_research/extraction.py))
   Runs concurrent LLM extraction over chunks, parses and validates JSON entities
   and relationships, enforces response limits and same-response relationship
   endpoints, merges duplicates, rejects prompt-example leakage, and retries
   failed chunk calls.

3. **Index Construction** ([core.py](src/rag_research/core.py))
   Merges duplicate entities and relations, stores chunks/entities/relations, embeds each retrieval unit, and builds a graph representation.
   Successful per-document chunking results and per-chunk extraction results are
   checkpointed under `CACHE_DIR` using independent stage fingerprints. A change
   limited to extraction invalidates extraction records while retaining compatible
   chunking records. The final persisted index remains protected by the complete
   build fingerprint.

4. **Storage** ([storage.py](src/rag_research/storage.py))
   Persists the experiment state to disk:
   - `entities.json`, `relations.json`, `chunks.json`
   - `entity_vectors.*`, `relation_vectors.*`, `chunk_vectors.*`
   - `graph.json`

5. **Retrieval** ([core.py](src/rag_research/core.py))
   Supports four modes:
   - `naive`: chunk-vector retrieval only.
   - `local`: entity-vector retrieval plus one-hop graph expansion.
   - `global`: relation-vector retrieval plus endpoint entity lookup.
   - `hybrid`: deduplicated merge of local and global retrieval traces.

6. **Evaluation** ([evaluate_retrieval.py](scripts/evaluate_retrieval.py), [evaluate_multihop_retrieval.py](scripts/evaluate_multihop_retrieval.py))
   Runs retrieval modes against chunk-independent canonical evidence. The
   MultiHopRAG evaluator isolates document-relative offsets, reports K curves
   and joint evidence/document success, and checkpoints every question/mode.

7. **Visualization** ([visualize_graph.py](scripts/visualize_graph.py))
   Renders the selected experiment graph under `artifacts/visualizations/` with `pyvis`.

## Project Layout

```text
src/rag_research/                  reusable RAG implementation
├── backends.py                    LLM and embedding backend adapters
├── chunking.py                    public fixed/semantic/agentic dispatcher
├── chunking_models.py             shared chunk configuration and span models
├── text_spans.py                  lossless sentence segmentation helpers
├── agentic_chunking.py            stateful Agentic Chunking workflow
├── agentic_llm.py                 Agentic LLM calls, validation, and recovery
├── agentic_boundaries.py          boundary projection and rebalancing rules
├── core.py                        LightRAG construction and retrieval pipeline
├── extraction.py                  concurrent entity/relation extraction
├── prompts.py                     chunking, extraction, and retrieval prompts
└── storage.py                     JSON/npy-backed KV, vector, and graph stores

scripts/                           runnable project entry points
├── run_demo.py                    build/query demo
├── evaluate_retrieval.py          retrieval evaluation
├── evaluate_multihop_retrieval.py MultiHopRAG retrieval evaluation
└── visualize_graph.py             graph visualization

data/
├── raw/                           source corpora
└── evaluation/                    canonical evidence annotations

artifacts/
├── stores/                        persisted stores for each chunking strategy
└── evaluations/                   generated evaluation results
```

## Setup

```bash
git clone https://github.com/xueyufeizhang/rag-research.git
cd rag-research
uv sync
cp .env.example .env
```

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

## Configuration

The project is configured through `.env`.

### LLM Backend

Set `LLM_BACKEND` to:

- `ollama`: local Ollama generation and embedding.
- `api`: any OpenAI-compatible chat completion endpoint for generation.

Embeddings currently use the Ollama `/api/embed` endpoint through `EMBED_MODEL`.

Index construction sends embeddings in bounded batches. `EMBEDDING_BATCH_SIZE`
controls the number of texts in each Ollama request and
`EMBEDDING_CONCURRENCY` limits simultaneous batch requests. The default values
are 32 and 2. Reduce the batch size first if the embedding model exceeds the
available memory. Semantic chunking uses the same batch backend and has its own
`SEMANTIC_EMBEDDING_BATCH_SIZE` setting.

### Chunking

Set `CHUNKING_STRATEGY` to one of:

- `fixed`
- `semantic`
- `agentic`

Use a separate `WORKING_DIR` for each strategy. A completed store is reused only
when its build fingerprint matches the corpus and complete pipeline provenance;
a conflicting store is rejected instead of being silently overwritten. Every
strategy is evaluated against the same canonical evidence file.

Use the same `CACHE_DIR` across related `WORKING_DIR` outputs when stage-level
reuse is desired. The chunking fingerprint covers the active chunking strategy,
its configuration, pipeline version, and any model that affects boundaries. The
extraction fingerprint separately covers the extraction backend, model, prompt,
and pipeline version. Cache record identities also include their exact document
or model input, so changed inputs coexist rather than being mistaken for hits.

Example fixed-size run:

```env
CHUNKING_STRATEGY=fixed
FIXED_WINDOW_SIZE=2400
FIXED_WINDOW_OVERLAP=200
WORKING_DIR=./artifacts/stores/dickens_fixed
```

Example semantic run:

```env
CHUNKING_STRATEGY=semantic
SEMANTIC_BREAKPOINT_PERCENTILE=92
SEMANTIC_MIN_SENTENCES=10
SEMANTIC_MAX_SENTENCES=32
SEMANTIC_BUFFER_SIZE=1
SEMANTIC_EMBEDDING_CONCURRENCY=4
WORKING_DIR=./artifacts/stores/dickens_semantic
```

Example final stateful agentic run:

```env
CHUNKING_STRATEGY=agentic
AGENTIC_BATCH_MAX_SENTENCES=60
AGENTIC_BATCH_MAX_CHARS=12000
AGENTIC_MIN_SENTENCES=4
AGENTIC_MAX_SENTENCES=20
AGENTIC_CONCURRENCY=4
AGENTIC_RETRIES=2
WORKING_DIR=./artifacts/stores/dickens_agentic
```

The final `agentic` strategy first extracts atomic propositions as source
sentence ranges. It then processes them sequentially through a state manager.
For each proposition, the manager reads the accumulated chunk catalog and the
open chunk's recent propositions, chooses `append` or `new_chunk`, and updates
the target chunk's title and summary. Hard size constraints restrict the actions
available to the model. Proposition text is never rewritten: final chunks remain
contiguous, lossless slices of the source, so canonical evidence offsets remain
valid. Titles and summaries are stored in chunk metadata and included in the
chunk embedding input, while graph extraction remains grounded only in source
text and document metadata. The per-document checkpoint also stores the
transition trace for audit and exact reuse.

`AGENTIC_CONCURRENCY` parallelizes proposition-extraction batches and any final
metadata refreshes. State transitions themselves remain sequential by design,
so this strategy makes substantially more LLM calls than non-agentic boundary-based
strategies.

### Optional reranking

Set `ENABLE_RERANKER=true` to apply CrossEncoder reranking after dense candidate
retrieval in naive, local, global, and hybrid modes. Set it to `false` for the dense-only
baseline. Both paths use the same candidate pools and final top-k limits so the
effect of CrossEncoder reranking can be compared directly.

```env
ENABLE_RERANKER=true
RERANK_MODEL=mixedbread-ai/mxbai-rerank-base-v1
```

## Usage

Run the demo query:

```bash
uv run python scripts/run_demo.py
```

This loads or builds the store configured by `WORKING_DIR`, then asks a sample hybrid retrieval question.

Run retrieval evaluation:

```bash
uv run python scripts/evaluate_retrieval.py
```

The default evaluation set is `data/evaluation/carol_canonical.json`. To run
only the naive retrieval mode while developing chunk reranking:

```bash
EVAL_MODES=naive uv run python scripts/evaluate_retrieval.py
```

With canonical evidence, the main retrieval metrics are:

- **Chunk Precision@K:** fraction of returned chunks that overlap gold evidence.
- **Evidence Recall@K:** fraction of canonical evidence spans covered by at least one returned chunk.
- **Answer-point Recall@K:** fraction of answer points supported by covered evidence.
- **MRR and nDCG@K:** rank-sensitive retrieval quality.
- **Chunk redundancy rate:** fraction of relevant returned chunks that add no new evidence coverage.
- **Average retrieved characters/tokens:** mean source-aligned context budget returned per question. Token counts use the fixed `EVAL_TOKENIZER_MODEL` with no special tokens.
- **Evidence density:** unique covered gold-evidence characters divided by all retrieved source characters; overlapping retrieved context is counted repeatedly only in the denominator.
- **Answer points per 1K tokens:** matched answer points per 1,000 retrieved tokens.

The evaluator writes both macro averages across questions and micro totals. It
rejects evaluation files that do not contain chunk-independent canonical
evidence.

The script evaluates `naive`, `local`, `global`, and `hybrid` modes with the question file configured by `EVAL_SET`, then writes:

- `artifacts/evaluations/retrieval_eval_<chunking>_<rerank-state>_results.json`
- `artifacts/evaluations/retrieval_eval_<chunking>_<rerank-state>_summaries.json`

For example: `retrieval_eval_semantic_rerank_results.json` and
`retrieval_eval_fixed_dense_only_summaries.json`.

### MultiHopRAG retrieval evaluation

Point `WORKING_DIR` at a completed MultiHopRAG index and run:

```bash
uv run python scripts/evaluate_multihop_retrieval.py
```

The evaluator verifies that the dataset and build fingerprint match the loaded
index, then produces two deliberately separate evaluation sections from the
same retrieval ranking:

- `thesis_extended` is the system-oriented evaluator. It reports Evidence
  Recall@K, Joint Evidence Success@K, Document Recall@K, Joint Document
  Success@K, Chunk Precision@K, MAP@K, MRR, nDCG@K, evidence density,
  retrieved tokens, and cross-document counts. `null_query` rows are kept
  separate and report context statistics only.
- `official` reproduces the [MultiHop-RAG official retrieval evaluator](https://github.com/yixuantt/MultiHop-RAG/blob/main/retrieval_evaluate.py):
  literal spaces and newlines are removed, a retrieved text is relevant when it
  contains a gold fact, and the aggregate contains exactly `Hits@4`, `Hits@10`,
  `MAP@10`, and `MRR@10`. It always scores top 10 and excludes `null_query`
  rows, matching the official baseline protocol.

Retrieval runs once at `max(max(EVAL_K_VALUES), 10)`. The extended evaluator
uses the configured K prefixes; the official evaluator uses the top-10 prefix.

Each run is stored under a configuration fingerprint in
`artifacts/evaluations/multihop_rag/`. Per-question/mode atomic checkpoints make
the 2,556-question run resumable. Final `results.json`, `summaries.json`, and
`run_manifest.json` preserve the dataset hashes, build fingerprint, retrieval
configuration, model names, and tokenizer. The `official/<mode>.json` files use
the upstream `query`/`retrieval_list`/`gold_list` JSON shape, so the official
script can independently rescore them. Useful controls are:

```env
EVAL_MODES=naive,local,global,hybrid
EVAL_K_VALUES=1,3,5,10,20
EVAL_CONCURRENCY=4
EVAL_INCLUDE_NULL=true
EVAL_MAX_QUESTIONS=0
```

Render the graph for the configured `WORKING_DIR`:

```bash
uv run python scripts/visualize_graph.py
```

## Experiment Artifacts

The checked-in Dickens stores are snapshots for comparing chunking behavior:

- `artifacts/stores/dickens_fixed`: fixed character windows.
- `artifacts/stores/dickens_semantic`: embedding-based semantic boundaries.

Each store has the same file schema, so retrieval and visualization can be pointed at any of them by changing `WORKING_DIR`. The evaluator maps the single canonical gold set to the selected store automatically.

## Chunk-independent gold evidence

`data/evaluation/carol_canonical.json` is the authoritative retrieval gold set for
*A Christmas Carol*. It identifies evidence with source sentence ranges and
zero-based character offsets, rather than with IDs from a particular chunking
run. Each evidence span also records which answer points it supports.

Rebuild the canonical file after changing its reviewed sentence ranges:

```bash
python scripts/build_canonical_gold.py
```

Map the same evidence spans to any persisted chunk store:

```bash
python scripts/map_canonical_gold_to_chunks.py \
  --chunks artifacts/stores/dickens_fixed/chunks.json \
  --output artifacts/evaluations/carol_fixed_from_canonical.json
```

The derived file contains the evidence spans covered by each chunk. The main
evaluator performs this mapping in memory, so a derived file is needed only for
inspection or annotation review.

## License

MIT - see [LICENSE](LICENSE).
