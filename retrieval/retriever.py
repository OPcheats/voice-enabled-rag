from pathlib import Path

from sentence_transformers import SentenceTransformer

from indexing.chunking import Chunk
from indexing.vector_store import VectorStore


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Retriever:
    """Retrieve relevant chunks from a persisted FAISS index."""

    def __init__(
        self,
        index_dir: str | Path = "data/vector_store",
        model_name: str = MODEL_NAME,
    ):
        self.index_dir = Path(index_dir)

        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)

        print("Loading vector store...")
        self.store = VectorStore.load(self.index_dir)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """Retrieve the most relevant chunks for a query."""
        if not query.strip():
            return []

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )

        return self.store.search(
            embedding,
            top_k=top_k,
        )


def main():
    retriever = Retriever()

    query = "What was the immediate impact of the success of the Manhattan Project?"

    print()
    print("=" * 60)
    print("QUERY")
    print("=" * 60)
    print(query)

    results = retriever.retrieve(query, top_k=3)

    print()
    print("=" * 60)
    print("RETRIEVAL RESULTS")
    print("=" * 60)

    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"\n[{rank}] Score: {score:.4f}")
        print(f"Document: {chunk.document_id}")
        print(chunk.text)


if __name__ == "__main__":
    main()