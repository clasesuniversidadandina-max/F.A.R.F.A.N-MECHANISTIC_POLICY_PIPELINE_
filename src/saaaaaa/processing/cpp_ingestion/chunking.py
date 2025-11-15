"""
Chunking Module (Deprecated - Compatibility Stub)

This module provides backward compatibility for CPP chunking functionality.
The functionality has been migrated to semantic_chunking_policy and embedding_policy modules.

For new code, use:
- saaaaaa.processing.semantic_chunking_policy.SemanticChunkingProducer
- saaaaaa.processing.embedding_policy.AdvancedSemanticChunker
"""

from __future__ import annotations

import warnings
from typing import Any

from ..cpp_ingestion.models import Chunk

warnings.warn(
    "cpp_ingestion.chunking is deprecated. "
    "Use semantic_chunking_policy or embedding_policy modules instead.",
    DeprecationWarning,
    stacklevel=2,
)


class AdvancedChunker:
    """
    Advanced chunking with overlap detection (Compatibility Stub).

    This is a minimal compatibility implementation for backward compatibility.
    For full functionality, use AdvancedSemanticChunker from embedding_policy.
    """

    def __init__(self, overlap_threshold: float = 0.1, **kwargs: Any):
        """
        Initialize advanced chunker.

        Parameters
        ----------
        overlap_threshold : float
            Threshold for chunk overlap detection (0.0-1.0)
        **kwargs : Any
            Additional configuration parameters (ignored in stub)
        """
        self.overlap_threshold = overlap_threshold
        self.config = kwargs

    def chunk_text(self, text: str, resolution: str = "micro") -> list[Chunk]:
        """
        Chunk text into semantic units (Stub implementation).

        Parameters
        ----------
        text : str
            Text to chunk
        resolution : str
            Chunking resolution: "micro", "meso", or "macro"

        Returns
        -------
        list[Chunk]
            List of chunks (stub returns empty list)
        """
        warnings.warn(
            "AdvancedChunker.chunk_text is a stub. "
            "Use AdvancedSemanticChunker for full functionality.",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    def compute_overlap(self, chunk1: Chunk, chunk2: Chunk) -> float:
        """
        Compute semantic overlap between two chunks (Stub).

        Parameters
        ----------
        chunk1 : Chunk
            First chunk
        chunk2 : Chunk
            Second chunk

        Returns
        -------
        float
            Overlap score (0.0-1.0), stub returns 0.0
        """
        return 0.0


__all__ = ["AdvancedChunker"]
