"""
Retrieval Interface — Tech Doc §9.

This is the integration boundary between Member 2 (application layer) and
Member 1 (dataset, chunking, embeddings, FAISS, retrieval, reranking).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import List, Optional

from dotenv import load_dotenv

from app.schemas import RetrievedChunk

load_dotenv()

logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    """Raised when retrieval fails unexpectedly."""


class RetrievalInterface:
    """Adapter that connects Member 2's orchestrator to Member 1's retrieval engine."""

    def __init__(
        self,
        top_k: Optional[int] = None,
        retrieval_fn=None,
    ) -> None:
        self.default_top_k: int = top_k or int(os.environ.get("TOP_K", "5"))
        self._retriever_instance = None
        self._retrieval_fn = retrieval_fn

        if self._retrieval_fn is None:
            try:
                from retrieval.retriever import Retriever
                self._retriever_instance = Retriever()
                self._retrieval_fn = self._m1_wrapper
            except Exception as e:
                logger.warning("Could not initialize Member 1 FAISS Retriever: %s", e)

    def _m1_wrapper(self, query: str, top_k: int):
        if not self._retriever_instance:
            return []
        raw_tuples = self._retriever_instance.retrieve(query, top_k=top_k)
        parsed = []
        for chunk, score in raw_tuples:
            parsed.append(
                {
                    "chunk_id": str(getattr(chunk, "chunk_id", "")),
                    "document_id": str(getattr(chunk, "document_id", "")),
                    "text": str(getattr(chunk, "text", "")),
                    "score": float(score),
                    "source": getattr(chunk, "source", None),
                    "metadata": getattr(chunk, "metadata", {}),
                }
            )
        return parsed

    async def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """Retrieve top-k relevant chunks for the given query."""
        k = top_k or self.default_top_k

        if self._retrieval_fn is None:
            logger.warning(
                "No Member 1 retrieval function connected. Returning empty results."
            )
            return []

        try:
            raw_results = await asyncio.get_event_loop().run_in_executor(
                None, self._retrieval_fn, query, k
            )
            return self._parse_results(raw_results)
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            raise RetrievalError(f"Retrieval engine error: {exc}") from exc

    def _parse_results(self, raw_results) -> List[RetrievedChunk]:
        """Convert Member 1's output format to RetrievedChunk list."""
        chunks: List[RetrievedChunk] = []
        for item in (raw_results or []):
            if isinstance(item, dict):
                chunks.append(
                    RetrievedChunk(
                        chunk_id=str(item.get("chunk_id", item.get("id", ""))),
                        document_id=str(item.get("document_id", item.get("doc_id", ""))),
                        text=str(item.get("text", item.get("content", ""))),
                        score=float(item.get("score", 0.0)),
                        source=item.get("source") or item.get("url"),
                        metadata=item.get("metadata", {}),
                    )
                )
            elif isinstance(item, RetrievedChunk):
                chunks.append(item)
            else:
                logger.warning("Unexpected retrieval result type: %s. Skipping.", type(item))
        return chunks

