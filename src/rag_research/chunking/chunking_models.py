from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkConfig:
    strategy: str = "fixed"
    fixed_size: int = 2400
    # One source-character overlap is applied after every strategy has chosen
    # its core boundaries.
    overlap_size: int = 200
    semantic_breakpoint_percentile: float = 90.0
    semantic_min_sentences: int = 8
    semantic_max_sentences: int = 24
    semantic_buffer_size: int = 1
    semantic_embedding_batch_size: int = 32
    semantic_embedding_concurrency: int = 4
    agentic_batch_max_sentences: int = 60
    agentic_batch_max_chars: int = 12000
    agentic_min_sentences: int = 4
    agentic_max_sentences: int = 20
    agentic_concurrency: int = 4
    agentic_retries: int = 2

    def __post_init__(self) -> None:
        if type(self.overlap_size) is not int or self.overlap_size < 0:
            raise ValueError("chunk overlap must be a non-negative integer")

    def fingerprint_dict(self) -> dict:
        """Return the canonical chunk configuration used in fingerprints."""
        return dict(self.__dict__)


@dataclass(frozen=True)
class ChunkSpan:
    text: str
    char_start: int
    char_end: int
    title: str | None = None
    summary: str | None = None


@dataclass(frozen=True)
class SentenceSpan:
    text: str
    char_start: int
    char_end: int
