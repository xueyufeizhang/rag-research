def chunk(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    if overlap >= size:
        raise ValueError(f"chunk overlap ({overlap}) must be smaller than chunk size ({size}), "
                          f"otherwise chunking never advances")

    chunk_result = []
    i = 0
    while i < len(text):
        chunk_result.append(text[i:i+size])
        i += size - overlap
    return chunk_result