from indexing.chunking import Chunk


def build_context(
    results: list[tuple[Chunk, float]],
) -> str:
    """Format retrieved chunks into context for answer generation."""
    if not results:
        return ""

    parts = []

    for rank, (chunk, score) in enumerate(results, start=1):
        parts.append(
            f"[Source {rank} | Score: {score:.4f}]\n"
            f"{chunk.text}"
        )

    return "\n\n".join(parts)


if __name__ == "__main__":
    print("Context builder: OK")