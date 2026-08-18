from dataclasses import dataclass
from typing import Any


@dataclass
class Document:
    document_id: str
    text: str
    metadata: dict[str, Any]


def record_to_document(record: dict[str, Any]) -> Document:
    """
    Convert one MSMARCO-XI record into a searchable document.

    English query/answer fields are used for embeddings because
    the current embedding model is optimized for English.
    """
    query = str(record.get("Eng_Query") or "").strip()
    answer = str(record.get("Eng_Answer") or "").strip()

    parts = []

    if query:
        parts.append(f"Query: {query}")

    if answer:
        parts.append(f"Answer: {answer}")

    text = "\n\n".join(parts)

    document_id = str(record.get("query_id", ""))

    metadata = {
        "source_lang": record.get("source_lang"),
        "target_lang": record.get("target_lang"),
        "query_type": record.get("query_type"),
        "query_id": record.get("query_id"),
    }

    return Document(
        document_id=document_id,
        text=text,
        metadata=metadata,
    )