from dataclasses import dataclass
from typing import Any
import re


@dataclass
class Chunk:
    text: str
    chunk_id: str
    document_id: str
    metadata: dict[str, Any]


def normalize_text(text: str) -> str:
    """Normalize whitespace without changing the actual content."""
    return " ".join(str(text).split())


def _make_chunk(
    text: str,
    document_id: str,
    index: int,
    metadata: dict[str, Any] | None = None,
) -> Chunk:
    """Create a Chunk with a deterministic ID."""
    return Chunk(
        text=text,
        chunk_id=f"{document_id}_chunk_{index}",
        document_id=document_id,
        metadata=metadata or {},
    )


def sentence_chunks(
    text: str,
    document_id: str,
    max_words: int = 120,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """
    Create sentence-aware chunks.

    Sentences are accumulated until adding another sentence
    would exceed max_words.
    """
    text = normalize_text(text)

    if not text:
        return []

    if max_words <= 0:
        raise ValueError("max_words must be greater than 0")

    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks: list[Chunk] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence = normalize_text(sentence)

        if not sentence:
            continue

        words = sentence.split()
        word_count = len(words)

        if current and current_words + word_count > max_words:
            chunks.append(
                _make_chunk(
                    " ".join(current),
                    document_id,
                    len(chunks),
                    metadata,
                )
            )
            current = []
            current_words = 0

        # Keep a long sentence rather than dropping it.
        current.append(sentence)
        current_words += word_count

    if current:
        chunks.append(
            _make_chunk(
                " ".join(current),
                document_id,
                len(chunks),
                metadata,
            )
        )

    return chunks


def sliding_window_chunks(
    text: str,
    document_id: str,
    window_words: int = 120,
    overlap_words: int = 30,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """
    Create overlapping fixed-word windows.
    """
    text = normalize_text(text)

    if not text:
        return []

    if window_words <= 0:
        raise ValueError("window_words must be greater than 0")

    if overlap_words < 0 or overlap_words >= window_words:
        raise ValueError(
            "overlap_words must be >= 0 and smaller than window_words"
        )

    words = text.split()
    step = window_words - overlap_words

    chunks: list[Chunk] = []

    for start in range(0, len(words), step):
        window = words[start:start + window_words]

        if not window:
            break

        chunks.append(
            _make_chunk(
                " ".join(window),
                document_id,
                len(chunks),
                metadata,
            )
        )

        if start + window_words >= len(words):
            break

    return chunks


def adaptive_chunks(
    text: str,
    document_id: str,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """
    Choose a chunking strategy based on document length.

    Short documents use sentence-aware chunks.
    Longer documents use overlapping windows.
    """
    text = normalize_text(text)

    if not text:
        return []

    word_count = len(text.split())

    if word_count <= 240:
        return sentence_chunks(
            text,
            document_id,
            max_words=120,
            metadata=metadata,
        )

    return sliding_window_chunks(
        text,
        document_id,
        window_words=120,
        overlap_words=30,
        metadata=metadata,
    )