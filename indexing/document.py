from dataclasses import dataclass
from typing import Any


@dataclass
class Document:
    document_id: str
    text: str
    metadata: dict[str, Any]


def record_to_document(record: dict[str, Any]) -> Document:
    """
    Convert one MSMARCO-XI dataset record into a Document.
    """
    document_id = str(record.get("query_id", ""))

    query = str(record.get("query", "")).strip()
    answer = str(record.get("Answer", "")).strip()

    parts = []

    if query:
        parts.append(f"Query: {query}")

    if answer:
        parts.append(f"Answer: {answer}")

    text = "\n\n".join(parts)

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