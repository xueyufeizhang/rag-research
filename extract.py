from dataclasses import dataclass
from prompt import PROMPTS
import re
import json
import time
import json_repair
import asyncio
from typing import Callable

@dataclass
class Entity:
    name: str
    type: str
    description: str
    source_id: list[str]

@dataclass
class Relation:
    source: str
    target: str
    keywords: list[str]
    description: str
    source_id: list[str]

def _parse_response(response: str, chunk_id: str, file_id: str) -> tuple[list[Entity], list[Relation]]:
    entities = []
    relations = []
    chunk_ref = file_id + "_chunk_" + chunk_id
    # json_str = re.search(r'\{.*\}', response, re.DOTALL).group()
    if not response:
        print(f"[parse] chunk {chunk_id}: empty response, skip!", flush=True)
        return entities, relations
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if not match:
        print(f"[parse] chunk {chunk_id}: no JSON object found, skip!\n{response[:200]}", flush=True)
        return entities, relations
    json_str = match.group()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        data = json_repair.loads(json_str)
    if not isinstance(data, dict):
        print(f"[parse] chunk {chunk_id}: JSON unable to repaire, skip!\n{json_str[:200]}", flush=True)
        return entities, relations

    for e in data.get("entities", []):
        if not isinstance(e, dict):
            continue
    
        name = e.get("name")
        if not name:
            continue
        
        entity = Entity(
            name=name,
            type=e.get("type") or "",
            description=e.get("description") or "",
            source_id=[chunk_ref]
        )
        entities.append(entity)
    
    for r in data.get("relationships", []):
        if not isinstance(r, dict):
            continue
        
        src, tgt = r.get("source"), r.get("target")
        if not src or not tgt:
            continue

        kw = r.get("keywords")
        if isinstance(kw, list):
            pass
        elif isinstance(kw, str):
            kw = kw.split(",")
        else:
            kw = []
        kw = [str(k).strip() for k in kw if str(k).strip()]

        relation = Relation(
            source=src,
            target=tgt,
            keywords = kw,
            description=r.get("description") or "",
            source_id=[chunk_ref]
        )
        relations.append(relation)
    return entities, relations


# async def extract(chunks: list[str], llm_func: Callable, file_id: str) -> tuple[list[Entity], list[Relation]]:
#     all_entites = []
#     all_relations = []
#     chunks_num = len(chunks)
#     start = time.time()
#     for idx, chunk in enumerate(chunks, start=1):
#         t0 = time.time()
#         response = await llm_func(
#             system=PROMPTS["entity_extraction_system_prompt"].format(
#                 entity_types_guidance=PROMPTS["default_entity_types_guidance"],
#                 examples=PROMPTS["entity_extraction_examples"],
#                 max_total_records=50, max_entity_records=20
#             ),
#             prompt=PROMPTS["entity_extraction_user_prompt"].format(
#                 entity_types_guidance=PROMPTS["default_entity_types_guidance"],
#                 input_text=chunk, max_total_records=50, max_entity_records=20
#             )
#         )
#         entities, relations = _parse_response(response, str(idx), file_id)
#         all_entites.extend(entities)
#         all_relations.extend(relations)

#         elapsed = time.time() - start
#         eta = elapsed / idx * (chunks_num - idx)
#         print(f"[extract] {idx}/{chunks_num}  "
#             f"+{len(entities)}ent +{len(relations)}rel  "
#             f"chunk {time.time()-t0:.1f}s  elapsed {elapsed:.0f}s  eta {eta:.0f}s",
#             flush=True)
        
#     return all_entites, all_relations



async def extract(chunks: list[str], llm_func: Callable, file_id: str, con_num: int) -> tuple[list[Entity], list[Relation]]:
    chunks_num = len(chunks)
    sem = asyncio.Semaphore(con_num)
    done_count = 0
    start = time.time()

    async def process_one(idx: int, chunk: str) -> tuple[list[Entity], list[Relation]] | None:
        nonlocal done_count
        async with sem:
            last_err = None
            for attempt in range(5):
                try:
                    t0 = time.time()
                    response = await llm_func(
                        system=PROMPTS["entity_extraction_system_prompt"].format(
                            entity_types_guidance=PROMPTS["default_entity_types_guidance"],
                            examples=PROMPTS["entity_extraction_examples"],
                            max_total_records=50, max_entity_records=20
                        ),
                        prompt=PROMPTS["entity_extraction_user_prompt"].format(
                            entity_types_guidance=PROMPTS["default_entity_types_guidance"],
                            input_text=chunk, max_total_records=50, max_entity_records=20
                        )
                    )
                    entities, relations = _parse_response(response, str(idx), file_id)

                    done_count += 1
                    elapsed = time.time() - start
                    eta = elapsed / done_count * (chunks_num - done_count)
                    print(f"[extract] {done_count}/{chunks_num} (chunk {idx})"
                          f"+{len(entities)}ent +{len(relations)}rel"
                          f"chunk {time.time()-t0:.1f}s  elapsed {elapsed:.0f}s  eta {eta:.0f}s",
                          flush=True)
                    return entities, relations

                except Exception as e:
                    last_err = e
                    wait = 2 ** attempt * 5
                    print(f"[extract] chunk {idx} failed ({type(e).__name__}: {e})"
                          f"Retry after {wait}s {attempt+1}/5", flush=True)
                    await asyncio.sleep(wait)

            print(f"[extract] chunk {idx} failed after 5 times retry, skipping it. "
                  f"last error: {last_err}", flush=True)
            return None

    results = await asyncio.gather(
        *[process_one(idx, c) for idx, c in enumerate(chunks, start=1)],
        return_exceptions=True,
    )

    all_entities: list[Entity] = []
    all_relations: list[Relation] = []
    failed = 0
    for result in results:
        if result is None or isinstance(result, Exception):
            failed += 1
            continue
        entities, relations = result
        all_entities.extend(entities)
        all_relations.extend(relations)

    if failed:
        print(f"[extract] {failed}/{chunks_num} chunk(s) failed and were skipped", flush=True)
    print(f"[extract] All done: {len(all_entities)} entities, {len(all_relations)} relations, "
          f"Total time: {time.time()-start:.0f}s", flush=True)
    return all_entities, all_relations