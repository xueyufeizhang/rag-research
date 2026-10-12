from typing import Any

def _count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.tokenize(text))

def select_chunks_by_token_budget(
    ordered_chunks: list[dict],
    token_budget: int,
    tokenizer: Any,
) -> dict:
    if token_budget < 0:
        raise ValueError("token_budget must be non-negative")

    selected_texts: list[str] = []
    selected_chunks: list[dict] = []
    selected_tokens = 0

    for chunk in ordered_chunks:
        if token_budget == 0:
            break

        text = chunk["text"]
        candidate_text = "\n\n".join(selected_texts + [text])
        candidate_tokens = _count_tokens(candidate_text, tokenizer)
        truncated = candidate_tokens > token_budget
        retained_text = text

        if truncated:
            prefix_length = len(candidate_text) - len(text)
            encoded = tokenizer(
                candidate_text,
                add_special_tokens=False,
                truncation=False,
                return_offsets_mapping=True,
            )

            possible_ends = sorted(
                {
                    end
                    for _, end in encoded["offset_mapping"][:token_budget]
                    if end > prefix_length
                },
                reverse=True,
            )

            retained_text = ""
            for end in possible_ends:
                count = _count_tokens(candidate_text[:end], tokenizer)
                if count <= token_budget:
                    local_end = end - prefix_length
                    retained_text = text[:local_end]
                    candidate_tokens = count
                    break

            if not retained_text:
                break

        selected_texts.append(retained_text)
        selected_tokens = candidate_tokens

        selected_chunks.append({
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "text": retained_text,
            "char_start": chunk["char_start"],
            "char_end": chunk["char_start"] + len(retained_text),
            "is_truncated": truncated,
        })

        if truncated:
            break

    return {
        "chunks": selected_chunks,
        "source_text": "\n\n".join(selected_texts),
        "source_tokens": selected_tokens,
        "token_budget": token_budget,
    }