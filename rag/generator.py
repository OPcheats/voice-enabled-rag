from retrieval.context import build_context
from retrieval.retriever import Retriever


class RAGGenerator:
    """Combine retrieval context with a question."""

    def __init__(self, retriever: Retriever | None = None):
        self.retriever = retriever or Retriever()

    def build_prompt(
        self,
        question: str,
        top_k: int = 3,
    ) -> str:
        """Build a grounded prompt from retrieved context."""
        results = self.retriever.retrieve(
            question,
            top_k=top_k,
        )

        context = build_context(results)

        return f"""Answer the question using only the provided context.

If the context does not contain enough information to answer,
say that the information is not available in the provided context.

Context:
{context}

Question:
{question}

Answer:"""


if __name__ == "__main__":
    generator = RAGGenerator()

    question = (
        "What was the immediate impact of the success "
        "of the Manhattan Project?"
    )

    print(generator.build_prompt(question))