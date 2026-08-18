from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from indexing.chunking import Chunk


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class VectorStore:
    """Simple FAISS-backed vector store for retrieval chunks."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]) -> None:
        """Embed and add chunks to the FAISS index."""
        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = np.asarray(embeddings, dtype="float32")

        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Return the most similar chunks for a query."""
        if self.index is None or not self.chunks:
            return []

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        )

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, len(self.chunks)),
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            results.append(
                (self.chunks[index], float(score))
            )

        return results