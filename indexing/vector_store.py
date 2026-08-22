from pathlib import Path
import pickle

import faiss
import numpy as np

from indexing.chunking import Chunk


class VectorStore:
    """FAISS vector store with persistent storage."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[Chunk] = []

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        """Add embeddings and their corresponding chunks."""
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks.")

        if len(embeddings) == 0:
            return

        embeddings = np.asarray(embeddings, dtype="float32")

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """Return the most similar chunks."""
        if not self.chunks:
            return []

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        )

        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            results.append(
                (self.chunks[index], float(score))
            )

        return results

    def save(self, directory: str | Path) -> None:
        """Save the FAISS index and chunk metadata."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(directory / "index.faiss"),
        )

        with open(directory / "chunks.pkl", "wb") as file:
            pickle.dump(self.chunks, file)

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        """Load a previously saved vector store."""
        directory = Path(directory)

        index = faiss.read_index(
            str(directory / "index.faiss")
        )

        with open(directory / "chunks.pkl", "rb") as file:
            chunks = pickle.load(file)

        store = cls(index.d)
        store.index = index
        store.chunks = chunks

        return store