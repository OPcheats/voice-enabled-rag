from rag.generator import RAGGenerator


class RAGApp:
    """Simple application interface for the RAG system."""

    def __init__(self):
        self.generator = RAGGenerator()

    def ask(self, question: str) -> str:
        """Answer a user question using retrieval-augmented generation."""
        question = question.strip()

        if not question:
            return "Please provide a question."

        return self.generator.generate(question)


if __name__ == "__main__":
    app = RAGApp()

    question = input("Question: ")
    answer = app.ask(question)

    print()
    print("Answer:", answer)