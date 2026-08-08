# rag-research

Research code and experiment artifacts for a thesis project on retrieval-augmented generation with graph-structured memory. The project started as a LightRAG replication, but it is now used as a broader experimental workspace for comparing chunking strategies, graph-based retrieval behavior, and evaluation traces.

## Current Focus

- Build a lightweight RAG pipeline with KV storage, vector indexes, and a knowledge graph.
- Compare fixed-size, sentence-window, and embedding-based semantic chunking.
- Evaluate retrieval quality across naive, local, global, and hybrid retrieval modes.
- Keep reusable experiment outputs for the *A Christmas Carol* corpus while the thesis method evolves.

## Pipeline

1. **Chunking** ([chunk.py](chunk.py))  
   Splits documents with one of three strategies:
   - `fixed`: character-based sliding windows.
   - `sentence_window`: sentence-based windows with overlap.
   - `semantic`: sentence-level semantic boundary detection using embeddings.

2. **Extraction** ([extract.py](extract.py))  
   Runs concurrent LLM extraction over chunks, parses JSON entities and relationships, repairs malformed JSON when possible, and retries failed chunk calls.

3. **Index Construction** ([core.py](core.py))  
   Merges duplicate entities and relations, stores chunks/entities/relations, embeds each retrieval unit, and builds a graph representation.

4. **Storage** ([storage.py](storage.py))  
   Persists the experiment state to disk:
   - `entities.json`, `relations.json`, `chunks.json`
   - `entity_vectors.*`, `relation_vectors.*`, `chunk_vectors.*`
   - `graph.json`

5. **Retrieval** ([core.py](core.py))  
   Supports four modes:
   - `naive`: chunk-vector retrieval only.
   - `local`: entity-vector retrieval plus one-hop graph expansion.
   - `global`: relation-vector retrieval plus endpoint entity lookup.
   - `hybrid`: deduplicated merge of local and global retrieval traces.

6. **Evaluation** ([eval/eval_retrieval.py](eval/eval_retrieval.py))  
   Runs all retrieval modes against a labeled question set and writes detailed results plus summary metrics under `eval/runs/`.

7. **Visualization** ([visual.py](visual.py))  
   Renders the selected experiment graph to `knowledge_graph.html` with `pyvis`.

## Project Layout

```text
backend.py                         LLM and embedding backend adapters
chunk.py                           fixed-size, sentence-window, and semantic chunkers
core.py                            LightRAG pipeline: construct, retrieve, retrieve_trace
extract.py                         concurrent entity/relation extraction
main.py                            small demo entry point for building and querying a store
prompt.py                          extraction and retrieval prompt templates
storage.py                         JSON/npy-backed KV, vector, and graph stores
visual.py                          graph visualization entry point
carol.txt                          sample corpus: A Christmas Carol
knowledge_graph.html               generated graph visualization output

dickens_fixed_size/                persisted store built with fixed-size chunks
dickens_sentence_window/           persisted store built with sentence-window chunks
dickens_semantic/                  persisted store built with semantic chunks

eval/eval_retrieval.py             retrieval evaluation script
eval/carol_eval_set_fixed_size.json
eval/carol_eval_set_sentence_window.json
eval/carol_eval_set_semantic.json
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

### Chunking

Set `CHUNKING_STRATEGY` to one of:

- `fixed`
- `sentence_window`
- `semantic`

Use a separate `WORKING_DIR` for each strategy. Construction is skipped when a store already exists, so changing chunking settings while reusing the same directory will not rebuild the stored chunks. For retrieval evaluation, set `EVAL_SET` to the matching labeled question file.

Example fixed-size run:

```env
CHUNKING_STRATEGY=fixed
FIXED_WINDOW_SIZE=2400
FIXED_WINDOW_OVERLAP=200
WORKING_DIR=./dickens_fixed_size
```

Example sentence-window run:

```env
CHUNKING_STRATEGY=sentence_window
SENTENCE_WINDOW_SIZE=8
SENTENCE_WINDOW_OVERLAP=2
WORKING_DIR=./dickens_sentence_window
```

Example semantic run:

```env
CHUNKING_STRATEGY=semantic
SEMANTIC_BREAKPOINT_PERCENTILE=92
SEMANTIC_MIN_SENTENCES=10
SEMANTIC_MAX_SENTENCES=32
SEMANTIC_BUFFER_SIZE=1
SEMANTIC_EMBEDDING_CONCURRENCY=4
WORKING_DIR=./dickens_semantic
```

## Usage

Run the demo query:

```bash
uv run python main.py
```

This loads or builds the store configured by `WORKING_DIR`, then asks a sample hybrid retrieval question.

Run retrieval evaluation:

```bash
uv run python eval/eval_retrieval.py
```

The script evaluates `naive`, `local`, `global`, and `hybrid` modes with the question file configured by `EVAL_SET`, then writes:

- `eval/runs/retrieval_eval_results.json`
- `eval/runs/retrieval_eval_results_summaries.json`

Render the graph for the configured `WORKING_DIR`:

```bash
uv run python visual.py
```

## Experiment Artifacts

The checked-in Dickens stores are snapshots for comparing chunking behavior:

- `dickens_fixed_size`: fixed character windows.
- `dickens_sentence_window`: sentence windows.
- `dickens_semantic`: embedding-based semantic boundaries.

Each store has the same file schema, so retrieval and visualization can be pointed at any of them by changing `WORKING_DIR`. For evaluation, pair it with the corresponding `eval/carol_eval_set_*.json` file.

## License

MIT - see [LICENSE](LICENSE).
