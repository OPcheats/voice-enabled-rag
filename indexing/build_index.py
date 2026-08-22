from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from indexing.dataset import get_dataset
from indexing.pipeline import stream_chunks
from indexing.vector_store import VectorStore


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = Path("data/vector_store")
BATCH_SIZE = 32
MAX_RECORDS = 100


def main():
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Opening dataset stream...")
    dataset = get_dataset()

    chunks = []
    records_processed = 0

    print(f"Building index from up to {MAX_RECORDS} records...")

    for chunk in stream_chunks(dataset):
        chunks.append(chunk)

        if len(chunks) >= BATCH_SIZE:
            texts = [chunk.text for chunk in chunks]

            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")

            if records_processed == 0:
                store = VectorStore(embeddings.shape[1])

            store.add(embeddings, chunks)

            records_processed += len(chunks)

            print(
                f"Indexed chunks: {records_processed}"
            )

            chunks = []

            if records_processed >= MAX_RECORDS:
                break

    # Add any remaining chunks.
    if chunks:
        texts = [chunk.text for chunk in chunks]

        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        if records_processed == 0:
            store = VectorStore(embeddings.shape[1])

        store.add(embeddings, chunks)

        records_processed += len(chunks)

    if records_processed == 0:
        print("No chunks were produced.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    store.save(OUTPUT_DIR)

    print()
    print("=" * 60)
    print("INDEX BUILD COMPLETE")
    print("=" * 60)
    print(f"Chunks indexed: {store.index.ntotal}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()