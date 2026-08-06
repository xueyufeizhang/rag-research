# rag-research

## Pipeline

1. **Chunking** (`chunk.py`) — configurable chunking with fixed-size character windows, sentence-window splitting, or embedding-based semantic splitting.
2. **Extraction** (`extract.py`) — concurrent LLM calls extract entities and binary relations from each chunk as JSON. Malformed model output is repaired with `json_repair` before parsing.
3. **Deduplication & merge** (`core.py`, `LightRAG.construct`) — entities sharing a name are merged (descriptions concatenated, source chunks unioned); relations are merged per unordered `(source, target)` pair.
4. **Storage** (`storage.py`) — three primitives, each JSON/npy-backed on disk:
   - `KVStore` — key/value store for entities, relations, and chunks
   - `VectorIndex` — flat numpy cosine-similarity index (separate indexes for entities, relations, chunks)
   - `GraphStore` — a `networkx` graph, persisted via `node_link_data`
5. **Retrieval** (`core.py`, `LightRAG.retrieve`) — four modes are implemented:
   - `naive` — chunk-vector retrieval only
   - `local` — entity-vector retrieval plus one-hop graph expansion
   - `global` — relation-vector retrieval plus endpoint entity lookup
   - `hybrid` — merges local and global retrieval results
6. **Visualization** (`visual.py`) — loads a persisted `graph.json` and renders an HTML graph with `pyvis`.

## Project layout

```
main.py               entry point — builds the store and runs one sample query
core.py                LightRAG class: construct() and retrieve()
chunk.py                fixed-size, sentence-window, and semantic chunkers
extract.py              concurrent entity/relation extraction over chunks
storage.py              KVStore / VectorIndex / GraphStore
prompt.py               extraction & retrieval prompt templates
visual.py               renders graph.json -> interactive HTML
dickens/                persisted store from a sample run (A Christmas Carol)
dickens_previous/       an earlier sample run, kept for comparison
knowledge_graph.html    pre-rendered visualization of dickens/graph.json
```

## Setup

```bash
git clone https://github.com/xueyufeizhang/rag-research.git
cd rag-research
uv sync
cp .env.example .env   # then fill in values for your setup
```

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

### LLM backend

Set `LLM_BACKEND` in `.env` to either:
- `ollama` — a local [Ollama](https://ollama.com) server (`OLLAMA_BASE_URL`, `LLM_MODEL`, `EMBED_MODEL`)
- `api` — any OpenAI-compatible endpoint (`API_BASE_URL`, `API_KEY`, `API_MODEL`)

See `.env.example` for the full list of configuration variables (chunking strategy, chunk size/overlap, retrieval top-k, concurrency, timeouts, working directory).

### Chunking strategies

Set `CHUNKING_STRATEGY` in `.env` to choose the chunking method used during indexing:

- `fixed` — character-based sliding windows. Controlled by `FIXED_WINDOW_SIZE` and `FIXED_WINDOW_OVERLAP`.
- `sentence_window` — sentence-based sliding windows. Controlled by `SENTENCE_WINDOW_SIZE` and `SENTENCE_WINDOW_OVERLAP`.
- `semantic` — sentence-based semantic boundary detection using embeddings. Controlled by `SEMANTIC_BREAKPOINT_PERCENTILE`, `SEMANTIC_MIN_SENTENCES`, `SEMANTIC_MAX_SENTENCES`, `SEMANTIC_BUFFER_SIZE`, and `SEMANTIC_EMBEDDING_CONCURRENCY`.

For chunking experiments, use a separate `WORKING_DIR` for each strategy. `LightRAG.construct()` skips indexing when a store already exists, so reusing the same directory will not rebuild chunks with the new strategy.

Example:

```env
CHUNKING_STRATEGY=sentence_window
SENTENCE_WINDOW_SIZE=8
SENTENCE_WINDOW_OVERLAP=2
WORKING_DIR=./dickens_sentence_window
```

Semantic chunking example:

```env
CHUNKING_STRATEGY=semantic
SEMANTIC_BREAKPOINT_PERCENTILE=90
SEMANTIC_MIN_SENTENCES=8
SEMANTIC_MAX_SENTENCES=24
SEMANTIC_BUFFER_SIZE=1
WORKING_DIR=./dickens_semantic_p90
```

### Sample corpus

`main.py` expects a `carol.txt` file in the repo root — the sample store checked into `dickens/` was built from the text of Charles Dickens' *A Christmas Carol* (public domain, e.g. via Project Gutenberg). Drop in any UTF-8 text file and adjust the filename in `main.py` to index your own corpus instead.

## Usage

```bash
uv run python main.py
```

This builds the KV/vector/graph stores under `WORKING_DIR` (skipping construction if a store already exists there) and prints an answer to a sample query ("Who is Scrooge?"). Retrieval mode can be selected in `main.py`.

To explore the resulting graph visually:

```bash
uv run python visual.py
```

## License

MIT — see [LICENSE](LICENSE).
