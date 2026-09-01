# rag-research

Research code and experiment artifacts for a thesis project on retrieval-augmented generation with graph-structured memory. The project started as a LightRAG replication, but it is now used as a broader experimental workspace for comparing chunking strategies, graph-based retrieval behavior, and evaluation traces.

## Current Focus

- Build a lightweight RAG pipeline with KV storage, vector indexes, and a knowledge graph.
- Compare fixed-size, sentence-window, and embedding-based semantic chunking.
- Evaluate retrieval quality across naive, local, global, and hybrid retrieval modes.
- Keep reusable experiment outputs for the *A Christmas Carol* corpus while the thesis method evolves.

## Pipeline

1. **Chunking** ([chunking.py](src/rag_research/chunking.py))
   Splits documents with one of three strategies:
   - `fixed`: character-based sliding windows.
   - `semantic`: sentence-level semantic boundary detection using embeddings.
   - `agentic_ibm`: constrained LLM topic-boundary selection over numbered sentences.

2. **Extraction** ([extraction.py](src/rag_research/extraction.py))
   Runs concurrent LLM extraction over chunks, parses and validates JSON entities
   and relationships, enforces response limits and same-response relationship
   endpoints, merges duplicates, rejects prompt-example leakage, and retries
   failed chunk calls.

3. **Index Construction** ([core.py](src/rag_research/core.py))
   Merges duplicate entities and relations, stores chunks/entities/relations, embeds each retrieval unit, and builds a graph representation.
   Successful per-document chunking results and per-chunk extraction results are
   checkpointed under the selected working directory. Re-running an interrupted
   build reuses checkpoints only when the complete build fingerprint matches.

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

6. **Evaluation** ([evaluate_retrieval.py](scripts/evaluate_retrieval.py))
   Runs retrieval modes against chunk-independent canonical evidence and writes detailed results plus summary metrics under `artifacts/evaluations/`.

7. **Visualization** ([visualize_graph.py](scripts/visualize_graph.py))
   Renders the selected experiment graph under `artifacts/visualizations/` with `pyvis`.

## Project Layout

```text
src/rag_research/                  reusable RAG implementation
├── backends.py                    LLM and embedding backend adapters
├── chunking.py                    fixed-size, sentence-window, and semantic chunkers
├── core.py                        LightRAG construction and retrieval pipeline
├── extraction.py                  concurrent entity/relation extraction
├── prompts.py                     extraction and retrieval prompt templates
└── storage.py                     JSON/npy-backed KV, vector, and graph stores

scripts/                           runnable project entry points
├── run_demo.py                    build/query demo
├── evaluate_retrieval.py          retrieval evaluation
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
- `agentic_ibm`

Use a separate `WORKING_DIR` for each strategy. A completed store is reused only
when its build fingerprint matches the corpus and complete pipeline provenance;
a conflicting store is rejected instead of being silently overwritten. Every
strategy is evaluated against the same canonical evidence file.

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

Example IBM-style agentic run:

```env
CHUNKING_STRATEGY=agentic_ibm
AGENTIC_BATCH_MAX_SENTENCES=60
AGENTIC_BATCH_MAX_CHARS=12000
AGENTIC_MIN_SENTENCES=4
AGENTIC_MAX_SENTENCES=20
AGENTIC_CONCURRENCY=4
AGENTIC_RETRIES=2
WORKING_DIR=./artifacts/stores/dickens_agentic_ibm
```

The LLM chooses contiguous topic boundaries within size-bounded macro-batches. It
returns sentence indexes rather than rewritten text, and the implementation
reconstructs chunks from the source sentences so canonical evidence offsets
remain valid. Structurally invalid or incomplete responses are retried. A
structurally valid response that violates the configured chunk sizes is
deterministically projected onto the constraints while minimizing total boundary
movement. Projection events are counted in the build result and stored with the
per-document chunking checkpoint for auditability. If structural validation
still fails after all attempts, construction stops instead of silently mixing
fallback chunks into an agentic experiment. Completed documents remain in the
chunking checkpoint and are reused when the same build is resumed.

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
