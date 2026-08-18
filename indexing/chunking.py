from dataclasses import dataclass
from typing import Any


@dataclass
class Chunk:
    text: str
    chunk_id: str
    document_id: str
    metadata: dict[str, Any]


def normalize_text(text: str) -> str:
    """Normalize whitespace without changing the actual content."""
    return " ".join(text.split())


def sentence_chunks(
    text: str,
    document_id: str,
    max_words: int = 120,
) -> list[Chunk]:
    """
    Create sentence-aware chunks.

    Sentences are accumulated until adding another sentence would
    exceed max_words.
    """
    # We'll implement this after confirming the dataset's text format.
    raise NotImplementedError


def sliding_window_chunks(
    text: str,
    document_id: str,
    window_words: int = 120,
    overlap_words: int = 30,
) -> list[Chunk]:
    """
    Create overlapping fixed-word windows.
    """
    # We'll implement this after confirming the dataset's text format.
    raise NotImplementedError


def adaptive_chunks(
    text: str,
    document_id: str,
) -> list[Chunk]:
    """
    Adaptive chunking based on document structure and length.
    """
    # We'll implement this after confirming the dataset's text format.
    raise NotImplementedError