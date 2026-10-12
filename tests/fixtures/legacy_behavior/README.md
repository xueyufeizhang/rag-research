# Legacy chunking baseline (pre-Jev)

This directory freezes the behavior of the existing `fixed`, `semantic`, and
`agentic` strategies before adding another strategy. Only tests and synthetic
fixtures are added; production code, dependencies, existing caches and indexes
are unchanged. No Jev integration is included.

## Run

From the `rag-research` repository root:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_legacy_*.py' -v
```

All model responses and embedding vectors are supplied by fixtures. Tests use
temporary working/cache directories and do not call model services. Configurations
explicitly provide every historical field so local `.env` settings cannot choose
the test inputs. Retry/fallback messages during the Agentic cases are expected.

## What is frozen

| File | Historical contract |
| --- | --- |
| `fingerprints.json` | Three strategies' complete explicit configs, serialized chunk config, complete build provenance, and exact chunking/extraction/build hashes for two synthetic documents. Unused legacy config fields remain in the captured payload. |
| `behavior.json` | Five cases' source text/metadata, configs, scripted LLM responses, embedding vectors, sentence spans, complete chunk records, embedding text, ordered provider calls, prompt hashes, and relative cache paths. |
| `caches/*.json` | Complete cache payloads emitted by the unchanged implementation, including schema version, source hashes, spans, generated metadata and ordered state events. |
| `capture_manifest.json` | Capture date, Git HEAD, dependency versions, source hashes and fixture hashes. Source hashes are historical provenance, not assertions preventing future source changes. |

The behavior scenarios are:

| Case | Coverage |
| --- | --- |
| `fixed_overlap` | Character windows, short final window and one uniform overlap expansion. |
| `semantic_topic_shift` | Buffered sentence input and embedding batches, semantic boundary, maximum size, short-tail rebalancing and overlap. Final core sentence ranges: `[1–3], [4–6], [7–8]`. |
| `semantic_uniform` | Equal-distance ties must not create semantic splits; maximum size still applies. Final core ranges: `[1–4], [5–8]`. |
| `agentic_routing_repair` | Mixed proposition sizes, forced new/append actions, semantic append/new decisions, metadata retry and dedicated repair, final tail merge, selective metadata refresh and overlap. The fake model deliberately proposes the opposite action on forced steps. Core ranges change from `[1–5], [6–8], [9–10]` to `[1–5], [6–10]`. There are 13 LLM calls: 1 proposition, 10 state, 2 metadata. |
| `agentic_projection_fallback` | Oversized proposition projection, source-derived metadata fallback, locally truncated summary with audit fields and overlap. Four LLM calls produce two chunks. |

Every output comparison checks static expected data. Chunk records include IDs,
source offsets, exact text (including whitespace), dataset/generated metadata and
`model_text`. The captured embedding text verifies that generated chunk titles and
summaries do not enter source embedding inputs. Serialized comparisons preserve
JSON type differences such as `1` versus `true`.

Cache compatibility is checked in both directions: newly emitted complete payloads
must match these historical files, and the historical files must load directly
without any model calls or cache rewrites. The latter does **not** create the cache
with today's writer first. Corrupted identity fields, source spans, metadata and
state traces must raise the existing errors instead of silently recomputing.

## Capture and maintenance

The fixture values were captured once from the existing working tree on
2026-10-12, before any production changes for this task. The working tree already
contained unrelated evaluation edits; the manifest records the actual source
hashes in addition to Git HEAD. Fixture hashes are checked by the tests, and the
tests also check that fixture bytes remain unchanged while running.

There is intentionally no update/regeneration mode. When a future change makes a
legacy test fail, inspect the changed contract first. Adding `jev` should normally
leave these legacy expectations unchanged. Any intentional break to a legacy
contract needs a separately reviewed compatibility/migration decision; do not
overwrite the historical fixtures just to make tests pass.

These cases freeze deterministic program behavior under recorded model replies.
They do not promise that a live LLM will always return the same answers, and they
do not exhaust every document or concurrency schedule. Existing unit tests remain
responsible for broader edge cases. These fixtures do not benchmark Jev or freeze
retrieval quality, extraction results, or vector-index contents.
