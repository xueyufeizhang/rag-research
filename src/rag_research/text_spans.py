import pysbd

from rag_research.chunking_models import SentenceSpan


_SENTENCE_SEGMENTER = pysbd.Segmenter(
    language="en",
    clean=False,
    char_span=True,
)


def split_sentences(text: str) -> list[SentenceSpan]:
    """Return a lossless, continuous sentence partition of the source text."""
    if not text or not text.strip():
        return []

    raw_spans = _SENTENCE_SEGMENTER.segment(text)
    merged_intervals: list[list[int]] = []
    previous_start = -1

    for raw_span in raw_spans:
        start = int(raw_span.start)
        end = int(raw_span.end)

        if not (0 <= start <= end <= len(text)):
            raise ValueError(
                "Sentence segmenter returned an invalid span: "
                f"start={start}, end={end}, text_length={len(text)}"
            )
        if start < previous_start:
            raise ValueError(
                "Sentence spans are out of order: "
                f"previous_start={previous_start}, start={start}, end={end}"
            )
        previous_start = start

        if start == end or not text[start:end].strip():
            continue

        if merged_intervals and start < merged_intervals[-1][1]:
            merged_intervals[-1][1] = max(merged_intervals[-1][1], end)
        else:
            merged_intervals.append([start, end])

    if not merged_intervals:
        return [SentenceSpan(text=text, char_start=0, char_end=len(text))]

    boundaries = [0]
    boundaries.extend(interval[0] for interval in merged_intervals[1:])
    boundaries.append(len(text))

    sentences = [
        SentenceSpan(
            text=text[start:end],
            char_start=start,
            char_end=end,
        )
        for start, end in zip(boundaries, boundaries[1:])
        if start < end
    ]

    if "".join(sentence.text for sentence in sentences) != text:
        raise ValueError("Sentence spans do not reconstruct the original text")
    for left, right in zip(sentences, sentences[1:]):
        if left.char_end != right.char_start:
            raise ValueError("Sentence spans are not a continuous partition")

    return sentences


def sentence_context(
    text: str,
    sentences: list[SentenceSpan],
    index: int,
    buffer_size: int,
) -> str:
    start = max(0, index - buffer_size)
    end = min(len(sentences), index + buffer_size + 1)
    context_start = sentences[start].char_start
    context_end = sentences[end - 1].char_end
    return text[context_start:context_end].strip()
