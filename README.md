# lightrag-replication

## Pipeline

1. **Chunking** (`chunk.py`) — fixed-size sliding-window text chunking with configurable overlap.
2. **Extraction** (`extract.py`) — concurrent LLM calls extract entities and binary relations from each chunk as JSON. Malformed model output is repaired with `json_repair` before parsing.
3. **Deduplication & merge** (`core.py`, `LightRAG.construct`) — entities sharing a name are merged (descriptions concatenated, source chunks unioned); relations are merged per unordered `(source, target)` pair.
4. **Storage** (`storage.py`) — three primitives, each JSON/npy-backed on disk:
   - `KVStore` — key/value store for entities, relations, and chunks
   - `VectorIndex` — flat numpy cosine-similarity index (separate indexes for entities, relations, chunks)
   - `GraphStore` — a `networkx` graph, persisted via `node_link_data`
5. **Retrieval** (`core.py`, `LightRAG.retrieve`) — **naive** mode is implemented: embed the query, take cosine top-k over the chunk vector index, and feed the retrieved chunks to the LLM as context. `local` and `global` (graph-aware retrieval over entities/relations) are under implementation.
6. **Visualization** (`visual.py`) — loads a persisted `graph.json` and renders an HTML graph with `pyvis`.

## Project layout

```
main.py               entry point — builds the store and runs one sample query
core.py                LightRAG class: construct() and retrieve()
chunk.py                sliding-window chunker
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
git clone https://github.com/xueyufeizhang/lightrag-replication.git
cd lightrag-replication
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in values for your setup
```

Requires Python 3.10+.

### LLM backend

Set `LLM_BACKEND` in `.env` to either:
- `ollama` — a local [Ollama](https://ollama.com) server (`OLLAMA_BASE_URL`, `LLM_MODEL`, `EMBED_MODEL`)
- `api` — any OpenAI-compatible endpoint (`API_BASE_URL`, `API_KEY`, `API_MODEL`)

See `.env.example` for the full list of configuration variables (chunking size/overlap, retrieval top-k, concurrency, timeouts, working directory).

### Sample corpus

`main.py` expects a `carol.txt` file in the repo root — the sample store checked into `dickens/` was built from the text of Charles Dickens' *A Christmas Carol* (public domain, e.g. via Project Gutenberg). Drop in any UTF-8 text file and adjust the filename in `main.py` to index your own corpus instead.

## Usage

```bash
python main.py
```

This builds the KV/vector/graph stores under `WORKING_DIR` (skipping construction if a store already exists there) and prints an answer to a sample query ("Who is Scrooge?") using naive retrieval.

To explore the resulting graph visually:

```bash
python visual.py
```

## Status

**Implemented:** chunking, concurrent LLM-based entity/relation extraction (with retry and partial-failure tolerance), dedup/merge, KV + vector + graph storage, naive (vector-only) retrieval, graph visualization.

**In progress:** `local` and `global` graph-aware retrieval modes (`core.py`, `LightRAG.retrieve`) are actively being developed, not yet functional; `hybrid` is planned once both land.

## License

MIT — see [LICENSE](LICENSE).
