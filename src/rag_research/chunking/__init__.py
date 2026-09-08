"""Public API for document chunking."""

from .agentic_chunking import AGENTIC_STATE_MODEL, agentic_chunk
from .chunking_models import ChunkConfig, ChunkSpan, SentenceSpan
from .dispatcher import (
    CHUNKING_PIPELINE_VERSION,
    CHUNKING_STRATEGY_VERSIONS,
    SEMANTIC_BOUNDARY_POLICY,
    SEMANTIC_DISTANCE_ABS_TOLERANCE,
    UNIFORM_OVERLAP_POLICY,
    apply_uniform_overlap,
    chunk_async,
    fixed_size_chunk,
    semantic_chunk,
)


__all__ = [
    "CHUNKING_PIPELINE_VERSION",
    "CHUNKING_STRATEGY_VERSIONS",
    "SEMANTIC_BOUNDARY_POLICY",
    "SEMANTIC_DISTANCE_ABS_TOLERANCE",
    "UNIFORM_OVERLAP_POLICY",
    "AGENTIC_STATE_MODEL",
    "ChunkConfig",
    "ChunkSpan",
    "SentenceSpan",
    "agentic_chunk",
    "chunk_async",
    "fixed_size_chunk",
    "apply_uniform_overlap",
    "semantic_chunk",
]
