from collections.abc import Iterator
from typing import Any

from indexing.chunking import Chunk, adaptive_chunks
from indexing.document import Document, record_to_document


def record_to_chunks(record: dict[str, Any]) -> list[Chunk]:
    """
    Convert one dataset record into retrieval chunks.
    """
    document = record_to_document(record)

    if not document.text:
        return []

    return adaptive_chunks(
        document.text,
        document.document_id,
        document.metadata,
    )


def stream_chunks(
    dataset: Iterator[dict[str, Any]],
) -> Iterator[Chunk]:
    """
    Convert a streaming dataset into chunks lazily.

    Records are processed one at a time so the entire dataset
    does not need to fit in memory.
    """
    for record in dataset:
        yield from record_to_chunks(record)